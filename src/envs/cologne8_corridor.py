"""
Multi-agent Gymnasium environment for the Cologne8 RESCO corridor.
Enforces a 15-second minimum green phase to preserve physical traffic dynamics.
"""

from __future__ import annotations

import pathlib
import subprocess
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# SUMO paths
_NET_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "sumo_net" / "cologne8"
_SUMOCFG = str(_NET_DIR / "cologne8.sumocfg")

YELLOW_DURATION = 4

class SumoCologne8Corridor(gym.Env):
    """
    Multi-agent traffic control for Cologne8 corridor.
    Heterogeneous action and observation spaces auto-discovered.
    Enforces minimum green phase for Markov property hold.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        use_gui: bool = False,
        max_steps: int = 3600,
        delta_time: int = 15,
        yellow_time: int = 4,
        min_green: int = 15,
        switch_penalty: float = 2.0,
        sumo_seed: int | str = "random",
        scale: float = 1.0,
        render_mode: Optional[str] = None,
        gui_delay: int = 200,
    ) -> None:
        super().__init__()

        self.use_gui = use_gui
        self.max_steps = max_steps
        self.delta_time = delta_time
        self.yellow_time = yellow_time
        self.min_green = min_green
        self.switch_penalty = switch_penalty
        self.sumo_seed = sumo_seed
        self.scale = scale
        self.render_mode = render_mode
        self.gui_delay = gui_delay

        self._sumo = None
        self._step_count = 0

        # Discovered topology
        self.tls_ids: list[str] = []
        self.tls_phases: dict[str, list[int]] = {}  # Green phase indices
        self.tls_incoming_lanes: dict[str, list[str]] = {}

        # Per-junction state
        self._current_green_phase_idx: dict[str, int] = {}
        self._is_yellow: dict[str, bool] = {}
        self._target_green_phase_idx: dict[str, int] = {}
        self._time_since_switch: dict[str, int] = {}

        self.observation_space = spaces.Dict()
        self.action_space = spaces.Dict()

        # Initialize network info
        self._discover_network()

    def _discover_network(self) -> None:
        """Starts SUMO briefly just to inspect the network topology."""
        self._sumo = self._get_sumo_module()
        self._start_sumo(discover_only=True)

        self.tls_ids = list(self._sumo.trafficlight.getIDList())
        
        obs_spaces = {}
        act_spaces = {}

        for tls_id in self.tls_ids:
            # Get controlled lanes
            links = self._sumo.trafficlight.getControlledLinks(tls_id)
            lanes = [link[0][0] for link in links if link]
            # Unique incoming lanes, preserving order
            unique_lanes = []
            for lane in lanes:
                if lane not in unique_lanes:
                    unique_lanes.append(lane)
            self.tls_incoming_lanes[tls_id] = unique_lanes

            # Get phases
            logic = self._sumo.trafficlight.getAllProgramLogics(tls_id)[0]
            green_phases = []
            for i, phase in enumerate(logic.phases):
                # A phase is considered green if it has 'G' or 'g' and no 'y' or 'Y'
                if ('G' in phase.state or 'g' in phase.state) and 'y' not in phase.state and 'Y' not in phase.state:
                    green_phases.append(i)
            # Fallback if no clean green phase (e.g. very weird logic)
            if not green_phases:
                green_phases = [0]
            
            self.tls_phases[tls_id] = green_phases

            # Observation dimension: queue per lane + current phase index + time since switch
            obs_dim = len(unique_lanes) + 2
            obs_spaces[tls_id] = spaces.Box(low=0.0, high=np.inf, shape=(obs_dim,), dtype=np.float32)
            act_spaces[tls_id] = spaces.Discrete(len(green_phases))

        self.observation_space = spaces.Dict(obs_spaces)
        self.action_space = spaces.Dict(act_spaces)

        self._close_sumo()

    def _get_sumo_module(self):
        if self.use_gui:
            import traci
            return traci
        else:
            import libsumo
            return libsumo

    def _start_sumo(self, discover_only: bool = False) -> None:
        sumo_binary = "sumo-gui" if self.use_gui and not discover_only else "sumo"
        cmd = [
            sumo_binary,
            "-c", _SUMOCFG,
            "--no-step-log", "true",
            "--waiting-time-memory", "1000",
            "--no-warnings", "true",
            "--ignore-route-errors", "true",
            "--time-to-teleport", "-1",  # Disable teleportations
        ]
        
        # Don't try to add non-existent polygon files for cologne8
        # We also skip tripinfo/stats outputs here unless specifically requested, but let's standardise
        cmd.extend([
            "--tripinfo-output", "/tmp/tripinfo.xml",
            "--stop-output", "/tmp/stopinfo.xml",
            "--emission-output", "/tmp/emission.xml",
            "--statistic-output", "/tmp/stats.xml",
        ])

        if not discover_only:
            cmd.extend(["--scale", str(self.scale)])
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

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)

        # Propagate Gymnasium seed to SUMO so traffic demand varies per episode.
        if seed is not None:
            self.sumo_seed = seed
        else:
            self.sumo_seed = "random"

        if self._sumo is not None:
            self._close_sumo()

        self._sumo = self._get_sumo_module()
        self._start_sumo()

        self._step_count = 0
        for tls_id in self.tls_ids:
            self._current_green_phase_idx[tls_id] = 0
            self._is_yellow[tls_id] = False
            self._target_green_phase_idx[tls_id] = 0
            self._time_since_switch[tls_id] = 0
            
            real_phase = self.tls_phases[tls_id][0]
            self._sumo.trafficlight.setPhase(tls_id, real_phase)

        obs = self._get_all_obs()
        info = self._get_info(
            rewards={t: 0.0 for t in self.tls_ids},
            switched={t: False for t in self.tls_ids},
        )
        return obs, info

    def step(
        self, actions: dict[str, int]
    ) -> tuple[dict[str, np.ndarray], dict[str, float], bool, bool, dict[str, Any]]:
        switched = {t: False for t in self.tls_ids}

        for tls_id in self.tls_ids:
            # Validate action
            num_phases = len(self.tls_phases[tls_id])
            action = actions.get(tls_id, 0)
            if action >= num_phases:
                action = num_phases - 1

            # 15-SECOND MINIMUM GREEN ENFORCEMENT
            if not self._is_yellow[tls_id] and self._time_since_switch[tls_id] < self.min_green:
                action = self._current_green_phase_idx[tls_id]  # Force hold

            if self._is_yellow[tls_id]:
                # Switch to target phase (assuming delta_time > yellow_time)
                # But here, we step `delta_time` at once. If we were in yellow before this step,
                # we should switch to green now.
                self._is_yellow[tls_id] = False
                self._current_green_phase_idx[tls_id] = self._target_green_phase_idx[tls_id]
                real_phase = self.tls_phases[tls_id][self._current_green_phase_idx[tls_id]]
                self._sumo.trafficlight.setPhase(tls_id, real_phase)
                self._time_since_switch[tls_id] = 0
            else:
                if action != self._current_green_phase_idx[tls_id]:
                    # Need transition
                    self._target_green_phase_idx[tls_id] = action
                    current_real_phase = self.tls_phases[tls_id][self._current_green_phase_idx[tls_id]]
                    # In SUMO, the yellow phase is usually immediately after the green phase
                    yellow_phase = current_real_phase + 1
                    self._sumo.trafficlight.setPhase(tls_id, yellow_phase)
                    self._is_yellow[tls_id] = True
                    self._time_since_switch[tls_id] = 0
                    switched[tls_id] = True
                else:
                    self._time_since_switch[tls_id] += self.delta_time

        # Advance simulation
        arrived = 0
        for _ in range(self.delta_time):
            self._sumo.simulationStep()
            arrived += self._sumo.simulation.getArrivedNumber()
        self._step_count += self.delta_time

        # Per-junction rewards
        rewards = {}
        for tls_id in self.tls_ids:
            queue = self._junction_queue(tls_id)
            switch_pen = self.switch_penalty if switched[tls_id] else 0.0
            starve_pen = self._junction_starvation_penalty(tls_id)
            rewards[tls_id] = -queue - switch_pen - starve_pen

        obs = self._get_all_obs()
        terminated = self._step_count >= self.max_steps
        info = self._get_info(rewards=rewards, switched=switched, throughput=arrived)

        global_reward = sum(rewards.values())

        return obs, global_reward, terminated, False, info

    def close(self) -> None:
        self._close_sumo()

    def _get_all_obs(self) -> dict[str, np.ndarray]:
        return {tls_id: self._get_obs(tls_id) for tls_id in self.tls_ids}

    def _get_obs(self, tls_id: str) -> np.ndarray:
        queues = []
        for lane in self.tls_incoming_lanes[tls_id]:
            try:
                queues.append(float(self._sumo.lane.getLastStepHaltingNumber(lane)))
            except Exception:
                queues.append(0.0)
                
        phase_flag = float(self._current_green_phase_idx[tls_id])
        return np.array(
            queues + [phase_flag, float(self._time_since_switch[tls_id])],
            dtype=np.float32,
        )

    def _junction_features(self, tls_id: str) -> np.ndarray:
        """
        Extracts a compact 5-dim handcrafted feature vector for a specific junction.
        Features:
        0: total_queue (normalized / 20.0)
        1: max_lane_queue (normalized / 10.0)
        2: mean_wait_time (normalized / 300.0s)
        3: phase_index_norm (idx / (n_phases - 1))
        4: time_since_switch_norm (time / max_steps)
        """
        queues = []
        waits = []
        for lane in self.tls_incoming_lanes[tls_id]:
            try:
                queues.append(float(self._sumo.lane.getLastStepHaltingNumber(lane)))
                waits.append(float(self._sumo.lane.getWaitingTime(lane)))
            except Exception:
                queues.append(0.0)
                waits.append(0.0)

        total_queue = sum(queues) / 20.0  # Normalize around 20 vehicles
        max_queue = max(queues) if queues else 0.0
        max_queue = max_queue / 10.0  # Normalize around 10 vehicles
        
        mean_wait = (sum(waits) / len(waits)) if waits else 0.0
        mean_wait = mean_wait / 300.0  # Normalize around 300s (5 mins)
        
        n_phases = len(self.tls_phases[tls_id])
        phase_idx = float(self._current_green_phase_idx.get(tls_id, 0.0))
        phase_norm = phase_idx / max(1, n_phases - 1)
        
        time_norm = float(self._time_since_switch.get(tls_id, 0.0)) / float(self.max_steps)

        return np.array(
            [total_queue, max_queue, mean_wait, phase_norm, time_norm],
            dtype=np.float32,
        )

    def get_global_state(self) -> np.ndarray:
        """
        Constructs a compact global state vector by concatenating the 5-dim feature
        vectors of all junctions in a fixed deterministic order.
        Returns:
            np.ndarray of shape (5 * n_junctions,)
        """
        features_list = [self._junction_features(tls_id) for tls_id in self.tls_ids]
        return np.concatenate(features_list).astype(np.float32)

    def _junction_queue(self, tls_id: str) -> float:
        total = 0.0
        for lane in self.tls_incoming_lanes[tls_id]:
            try:
                total += self._sumo.lane.getLastStepHaltingNumber(lane)
            except Exception:
                pass
        return total

    def _junction_wait_time(self, tls_id: str) -> float:
        total = 0.0
        for lane in self.tls_incoming_lanes[tls_id]:
            try:
                total += self._sumo.lane.getWaitingTime(lane)
            except Exception:
                pass
        return total

    def _junction_starvation_penalty(self, tls_id: str) -> float:
        penalty = 0.0
        for lane in self.tls_incoming_lanes[tls_id]:
            try:
                wait = self._sumo.lane.getWaitingTime(lane)
                # Quadratic/exponential penalty to severely punish starving a specific lane
                # Scaling by 100 keeps it from drowning out the queue metric immediately.
                penalty += (wait / 100.0) ** 2
            except Exception:
                pass
        return penalty

    def _get_info(
        self,
        rewards: dict[str, float],
        switched: dict[str, bool],
        throughput: int = 0,
    ) -> dict[str, Any]:
        total_queue = sum(self._junction_queue(t) for t in self.tls_ids)
        total_wait = sum(self._junction_wait_time(t) for t in self.tls_ids)

        return {
            "queue_length": total_queue,
            "wait_time_total": total_wait,
            "reward": sum(rewards.values()),
            "switch_penalty": sum(-self.switch_penalty if switched[t] else 0.0 for t in self.tls_ids),
            "throughput": throughput,
            "step": self._step_count,
            "scale": self.scale,
            "per_junction": {
                tls_id: {
                    "queue_length": self._junction_queue(tls_id),
                    "wait_time": self._junction_wait_time(tls_id),
                    "reward": rewards[tls_id],
                    "switched": switched[tls_id],
                }
                for tls_id in self.tls_ids
            },
        }
