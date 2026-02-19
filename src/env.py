"""
Custom Gymnasium environment for SUMO traffic signal control.

Dual backend:
  - libsumo  (fast, headless) – for training / evaluation
  - traci    (slower, supports sumo-gui) – for visual demo

The env controls a single traffic-light junction ("center") in a
4-way intersection and exposes:
  Observation: queue lengths (8), current phase (1), time since switch (1)
  Action:      0 = keep current green, 1 = switch
  Reward:      -(total_queue) - α · I(switched)
"""

from __future__ import annotations

import os
import pathlib
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# ---------------------------------------------------------------------------
# SUMO paths
# ---------------------------------------------------------------------------
_NET_DIR = pathlib.Path(__file__).resolve().parent.parent / "sumo_net"
_SUMOCFG = str(_NET_DIR / "intersection.sumocfg")

# Incoming (towards-center) edges and their two lanes
INCOMING_EDGES = [
    "north_to_center",
    "south_to_center",
    "east_to_center",
    "west_to_center",
]
INCOMING_LANES = [f"{e}_{i}" for e in INCOMING_EDGES for i in range(2)]

TLS_ID = "center"

# Phase indices in the generated tlLogic
PHASE_NS_GREEN = 0   # state: GGGggrrrrrGGGggrrrrr
PHASE_NS_YELLOW = 1  # state: yyyyyrrrrryyyyyrrrrr
PHASE_EW_GREEN = 2   # state: rrrrrGGGggrrrrrGGGgg
PHASE_EW_YELLOW = 3  # state: rrrrryyyyyrrrrryyyyy

GREEN_PHASES = {PHASE_NS_GREEN, PHASE_EW_GREEN}
YELLOW_DURATION = 5  # seconds of yellow before switching


class SumoEnv(gym.Env):
    """Single-agent traffic signal control environment."""

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
        self.delta_time = delta_time      # seconds between agent decisions
        self.yellow_time = yellow_time
        self.switch_penalty = switch_penalty
        self.sumo_seed = sumo_seed
        self.render_mode = render_mode
        self.gui_delay = gui_delay        # ms delay per sim step in GUI

        # Will be set in reset()
        self._sumo = None  # libsumo or traci module
        self._step_count = 0
        self._time_since_switch = 0
        self._current_green_phase = PHASE_NS_GREEN
        self._is_yellow = False
        self._yellow_timer = 0

        # Spaces
        # obs: 8 queue lengths + 1 phase (0/1 for NS/EW) + 1 time-since-switch
        self.observation_space = spaces.Box(
            low=0.0, high=np.inf, shape=(10,), dtype=np.float32
        )
        # action: 0 = keep, 1 = switch
        self.action_space = spaces.Discrete(2)

    # ------------------------------------------------------------------
    # SUMO lifecycle
    # ------------------------------------------------------------------
    def _get_sumo_module(self):
        """Return the correct SUMO API module."""
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
            cmd += ["--start", "--quit-on-end", "--delay", str(self.gui_delay)]
            self._sumo.start(cmd)
        else:
            self._sumo.start(cmd)

    def _close_sumo(self) -> None:
        try:
            self._sumo.close()
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
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)

        # Close previous run if any
        if self._sumo is not None:
            self._close_sumo()

        self._sumo = self._get_sumo_module()
        self._start_sumo()

        self._step_count = 0
        self._time_since_switch = 0
        self._current_green_phase = PHASE_NS_GREEN
        self._is_yellow = False
        self._yellow_timer = 0

        # Set initial phase
        self._sumo.trafficlight.setPhase(TLS_ID, self._current_green_phase)

        obs = self._get_obs()
        info = self._get_info(reward=0.0, switched=False)
        return obs, info

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        switched = False

        if self._is_yellow:
            # Continue yellow countdown
            self._yellow_timer -= self.delta_time
            if self._yellow_timer <= 0:
                # Yellow done → switch to the other green
                self._is_yellow = False
                self._current_green_phase = (
                    PHASE_EW_GREEN
                    if self._current_green_phase == PHASE_NS_GREEN
                    else PHASE_NS_GREEN
                )
                self._sumo.trafficlight.setPhase(
                    TLS_ID, self._current_green_phase
                )
                self._time_since_switch = 0
        else:
            if action == 1:
                # Initiate yellow phase
                yellow_phase = (
                    PHASE_NS_YELLOW
                    if self._current_green_phase == PHASE_NS_GREEN
                    else PHASE_EW_YELLOW
                )
                self._sumo.trafficlight.setPhase(TLS_ID, yellow_phase)
                self._is_yellow = True
                self._yellow_timer = self.yellow_time
                switched = True
            # else: keep current green, do nothing

        # Advance simulation by delta_time seconds
        arrived = 0
        for _ in range(self.delta_time):
            self._sumo.simulationStep()
            arrived += self._sumo.simulation.getArrivedNumber()
        self._step_count += self.delta_time
        self._time_since_switch += self.delta_time

        # Compute reward
        queue = self._total_queue_length()
        penalty = self.switch_penalty if switched else 0.0
        reward = -queue - penalty

        obs = self._get_obs()
        terminated = self._step_count >= self.max_steps
        truncated = False
        info = self._get_info(
            reward=reward, switched=switched, throughput=arrived
        )

        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        self._close_sumo()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _total_queue_length(self) -> float:
        return sum(
            self._sumo.lane.getLastStepHaltingNumber(lane)
            for lane in INCOMING_LANES
        )

    def _total_wait_time(self) -> float:
        return sum(
            self._sumo.lane.getWaitingTime(lane) for lane in INCOMING_LANES
        )

    def _get_obs(self) -> np.ndarray:
        queues = [
            float(self._sumo.lane.getLastStepHaltingNumber(lane))
            for lane in INCOMING_LANES
        ]
        phase_flag = (
            0.0 if self._current_green_phase == PHASE_NS_GREEN else 1.0
        )
        return np.array(
            queues + [phase_flag, float(self._time_since_switch)],
            dtype=np.float32,
        )

    def _get_info(
        self, reward: float, switched: bool, throughput: int = 0
    ) -> dict[str, Any]:
        return {
            "queue_length": self._total_queue_length(),
            "wait_time_total": self._total_wait_time(),
            "reward": reward,
            "switch_penalty": -self.switch_penalty if switched else 0.0,
            "throughput": throughput,
            "step": self._step_count,
        }
