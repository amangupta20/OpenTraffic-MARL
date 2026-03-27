"""
Multi-agent Gymnasium environment for the Bangalore MG Road corridor.
"""

from __future__ import annotations

import pathlib
import subprocess
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# SUMO paths
_NET_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "sumo_net" / "bangalore_mg_road"
_SUMOCFG = str(_NET_DIR / "osm.sumocfg")

YELLOW_DURATION = 5

class SumoBangaloreCorridor(gym.Env):
    """
    Multi-agent traffic control for Bangalore MG Road corridor.
    Heterogeneous action and observation spaces auto-discovered.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        use_gui: bool = False,
        max_steps: int = 3600,
        delta_time: int = 5,
        yellow_time: int = 5,
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
            "--additional-files", f"{_NET_DIR}/osm.poly.xml.gz,{_NET_DIR}/osm_stops.add.xml",
            "--tripinfo-output", "/tmp/tripinfo.xml",
            "--stop-output", "/tmp/stopinfo.xml",
            "--emission-output", "/tmp/emission.xml",
            "--statistic-output", "/tmp/stats.xml",
            "--time-to-teleport", "-1",  # Disable teleportations
        ]
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

            if self._is_yellow[tls_id]:
                # Assuming delta_time == yellow_time (5s) for simplicity
                # Switch to target phase
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
                    # time_since_switch remains the same during yellow, or reset? Reset makes sense.
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
            penalty = self.switch_penalty if switched[tls_id] else 0.0
            rewards[tls_id] = -queue - penalty

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
