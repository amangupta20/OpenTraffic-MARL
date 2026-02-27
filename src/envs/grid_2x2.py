"""
Multi-agent Gymnasium environment for a 2×2 signalized grid.

Controls 4 traffic-light junctions (A0, A1, B0, B1) in a shared SUMO
simulation.  Each junction gets a LOCAL observation, action, and reward
identical to the single-intersection env — enabling zero-shot transfer
of a trained single-intersection PPO model.

Dual backend:
  - libsumo (fast, headless) — for evaluation / comparison
  - traci  (supports sumo-gui) — for visual demo
"""

from __future__ import annotations

import os
import pathlib
import subprocess
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# ---------------------------------------------------------------------------
# SUMO paths
# ---------------------------------------------------------------------------
_NET_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "sumo_net" / "grid_2x2"
_SUMOCFG = str(_NET_DIR / "grid.sumocfg")

# ---------------------------------------------------------------------------
# Junction topology  (matches netgenerate output)
# ---------------------------------------------------------------------------
TLS_IDS = ["A0", "A1", "B0", "B1"]

# Incoming edges for each junction (4 edges × 2 lanes = 8 lanes each)
INCOMING_EDGES = {
    "A0": ["A1A0", "B0A0", "bottom0A0", "left0A0"],
    "A1": ["top0A1", "B1A1", "A0A1", "left1A1"],
    "B0": ["B1B0", "right0B0", "bottom1B0", "A0B0"],
    "B1": ["top1B1", "right1B1", "B0B1", "A1B1"],
}

INCOMING_LANES = {
    tls_id: [f"{edge}_{i}" for edge in edges for i in range(2)]
    for tls_id, edges in INCOMING_EDGES.items()
}

# Phase indices (identical across all 4 junctions)
PHASE_NS_GREEN = 0
PHASE_NS_YELLOW = 1
PHASE_EW_GREEN = 2
PHASE_EW_YELLOW = 3

YELLOW_DURATION = 5


class SumoGrid2x2Env(gym.Env):
    """Multi-agent traffic control for a 2×2 signalized grid.

    Each junction exposes the same 10-dim observation and Discrete(2) action
    as the single-intersection env, enabling direct weight transfer.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        use_gui: bool = False,
        max_steps: int = 3600,
        delta_time: int = 5,
        yellow_time: int = YELLOW_DURATION,
        switch_penalty: float = 2.0,
        sumo_seed: int | str = "random",
        render_mode: Optional[str] = None,
        gui_delay: int = 200,
    ) -> None:
        super().__init__()

        self.use_gui = use_gui
        self.max_steps = max_steps
        self.delta_time = delta_time
        self.yellow_time = yellow_time
        self.switch_penalty = switch_penalty
        self.sumo_seed = sumo_seed
        self.render_mode = render_mode
        self.gui_delay = gui_delay

        self._sumo = None
        self._step_count = 0

        # Per-junction state
        self._current_green_phase: dict[str, int] = {}
        self._is_yellow: dict[str, bool] = {}
        self._yellow_timer: dict[str, int] = {}
        self._time_since_switch: dict[str, int] = {}

        # Spaces: Dict of per-junction spaces
        single_obs = spaces.Box(low=0.0, high=np.inf, shape=(10,), dtype=np.float32)
        single_act = spaces.Discrete(2)

        self.observation_space = spaces.Dict(
            {tls_id: single_obs for tls_id in TLS_IDS}
        )
        self.action_space = spaces.Dict(
            {tls_id: single_act for tls_id in TLS_IDS}
        )

    # ------------------------------------------------------------------
    # SUMO lifecycle
    # ------------------------------------------------------------------
    def _get_sumo_module(self):
        if self.use_gui:
            import traci
            return traci
        else:
            import libsumo
            return libsumo

    def _start_sumo(self) -> None:
        sumo_binary = "sumo-gui" if self.use_gui else "sumo"
        cmd = [
            sumo_binary,
            "-c", _SUMOCFG,
            "--no-step-log", "true",
            "--waiting-time-memory", "1000",
            "--no-warnings", "true",
        ]
        if self.sumo_seed != "random":
            cmd += ["--seed", str(self.sumo_seed)]

        if self.use_gui:
            cmd += [
                "--start", "--quit-on-end",
                "--delay", str(self.gui_delay),
                "--window-size", "1280,720",
                "--window-pos", "0,0",
            ]

        self._sumo.start(cmd)

    def _close_sumo(self) -> None:
        try:
            self._sumo.close()
        except Exception:
            pass
        if self.use_gui:
            for name in ("sumo", "sumo-gui"):
                try:
                    subprocess.run(
                        ["pkill", "-f", name],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)

        if self._sumo is not None:
            self._close_sumo()

        self._sumo = self._get_sumo_module()
        self._start_sumo()

        self._step_count = 0
        for tls_id in TLS_IDS:
            self._current_green_phase[tls_id] = PHASE_NS_GREEN
            self._is_yellow[tls_id] = False
            self._yellow_timer[tls_id] = 0
            self._time_since_switch[tls_id] = 0
            self._sumo.trafficlight.setPhase(tls_id, PHASE_NS_GREEN)

        obs = self._get_all_obs()
        info = self._get_info(
            rewards={t: 0.0 for t in TLS_IDS},
            switched={t: False for t in TLS_IDS},
        )
        return obs, info

    def step(
        self, actions: dict[str, int]
    ) -> tuple[dict[str, np.ndarray], dict[str, float], bool, bool, dict[str, Any]]:
        switched = {}

        # Apply each agent's action to its junction
        for tls_id in TLS_IDS:
            action = actions.get(tls_id, 0)
            switched[tls_id] = False

            if self._is_yellow[tls_id]:
                self._yellow_timer[tls_id] -= self.delta_time
                if self._yellow_timer[tls_id] <= 0:
                    self._is_yellow[tls_id] = False
                    self._current_green_phase[tls_id] = (
                        PHASE_EW_GREEN
                        if self._current_green_phase[tls_id] == PHASE_NS_GREEN
                        else PHASE_NS_GREEN
                    )
                    self._sumo.trafficlight.setPhase(
                        tls_id, self._current_green_phase[tls_id]
                    )
                    self._time_since_switch[tls_id] = 0
            else:
                if action == 1:
                    yellow_phase = (
                        PHASE_NS_YELLOW
                        if self._current_green_phase[tls_id] == PHASE_NS_GREEN
                        else PHASE_EW_YELLOW
                    )
                    self._sumo.trafficlight.setPhase(tls_id, yellow_phase)
                    self._is_yellow[tls_id] = True
                    self._yellow_timer[tls_id] = self.yellow_time
                    switched[tls_id] = True

        # Advance simulation
        arrived = 0
        for _ in range(self.delta_time):
            self._sumo.simulationStep()
            arrived += self._sumo.simulation.getArrivedNumber()
        self._step_count += self.delta_time
        for tls_id in TLS_IDS:
            self._time_since_switch[tls_id] += self.delta_time

        # Per-junction rewards
        rewards = {}
        for tls_id in TLS_IDS:
            queue = self._junction_queue(tls_id)
            penalty = self.switch_penalty if switched[tls_id] else 0.0
            rewards[tls_id] = -queue - penalty

        obs = self._get_all_obs()
        terminated = self._step_count >= self.max_steps
        info = self._get_info(rewards=rewards, switched=switched, throughput=arrived)

        # Gymnasium expects a single scalar reward — use global sum
        global_reward = sum(rewards.values())

        return obs, global_reward, terminated, False, info

    def close(self) -> None:
        self._close_sumo()

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------
    def _get_all_obs(self) -> dict[str, np.ndarray]:
        return {tls_id: self._get_obs(tls_id) for tls_id in TLS_IDS}

    def _get_obs(self, tls_id: str) -> np.ndarray:
        queues = [
            float(self._sumo.lane.getLastStepHaltingNumber(lane))
            for lane in INCOMING_LANES[tls_id]
        ]
        phase_flag = (
            0.0 if self._current_green_phase[tls_id] == PHASE_NS_GREEN else 1.0
        )
        return np.array(
            queues + [phase_flag, float(self._time_since_switch[tls_id])],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    def _junction_queue(self, tls_id: str) -> float:
        return sum(
            self._sumo.lane.getLastStepHaltingNumber(lane)
            for lane in INCOMING_LANES[tls_id]
        )

    def _junction_wait_time(self, tls_id: str) -> float:
        return sum(
            self._sumo.lane.getWaitingTime(lane)
            for lane in INCOMING_LANES[tls_id]
        )

    def _get_info(
        self,
        rewards: dict[str, float],
        switched: dict[str, bool],
        throughput: int = 0,
    ) -> dict[str, Any]:
        total_queue = sum(self._junction_queue(t) for t in TLS_IDS)
        total_wait = sum(self._junction_wait_time(t) for t in TLS_IDS)

        info: dict[str, Any] = {
            # Global metrics (compatible with existing Prometheus/compare)
            "queue_length": total_queue,
            "wait_time_total": total_wait,
            "reward": sum(rewards.values()),
            "switch_penalty": sum(
                -self.switch_penalty if switched[t] else 0.0 for t in TLS_IDS
            ),
            "throughput": throughput,
            "step": self._step_count,
            # Per-junction breakdown
            "per_junction": {
                tls_id: {
                    "queue_length": self._junction_queue(tls_id),
                    "wait_time": self._junction_wait_time(tls_id),
                    "reward": rewards[tls_id],
                    "switched": switched[tls_id],
                }
                for tls_id in TLS_IDS
            },
        }
        return info
