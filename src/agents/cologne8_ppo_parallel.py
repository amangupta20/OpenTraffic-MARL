"""
Parallel Heterogeneous multi-agent training for the Cologne8 RESCO corridor.

Uses multiprocessing to spin up 4 parallel SUMO instances. 8 independent PPO 
agents communicate via a Coordinator wrapper with `MultiAgentSharedSubproc`.
"""

import argparse
import json
import os
import pathlib
import threading
import time
from typing import Any

import gymnasium as gym
import numpy as np
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv

from src.envs import make_env
from src.envs.multi_agent_subproc import MultiAgentSharedSubproc
from src.utils.metrics import start_metrics_server

MODELS_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "models"


class SubprocCoordinator:
    """Synchronizes 5 top-level PPO threads into 4 background multiprocessing SUMO pipes."""

    def __init__(self, multi_env: MultiAgentSharedSubproc, tls_ids: list[str], total_timesteps: int, delta_time: float, num_envs: int):
        self.multi_env = multi_env
        self.tls_ids = tls_ids
        self.num_agents = len(tls_ids)
        self.total_timesteps = total_timesteps
        self.delta_time = delta_time
        self.num_envs = num_envs
        
        # Action buffer: Dict[tls_id, np.ndarray(num_envs,)]
        self.actions = {}
        
        # State buffers tailored for SB3 VecEnv Return arrays
        self.obs = {t: None for t in tls_ids}
        self.rewards = {t: None for t in tls_ids}
        self.dones = {t: None for t in tls_ids}
        self.infos = {t: None for t in tls_ids}
        
        # Thread sync: PPO learn() uses threading barriers
        # Barrier action callback executes sequentially IN the context of the last thread to arrive
        self.step_barrier = threading.Barrier(self.num_agents, action=self._execute_step)
        self.reset_barrier = threading.Barrier(self.num_agents, action=self._execute_reset)
        
        # Curriculum tracking
        self.current_grade_idx = 0
        self.total_master_steps = 0
        
        # FPS Tracking
        self._wall_start = time.time()
        self._step_count_for_fps = 0
        self._last_scale = 0.6
        
    def set_action(self, tls_id: str, actions: np.ndarray):
        """Called by AgentFacadeVecEnv.step_async() in a PPO thread."""
        self.actions[tls_id] = actions
        
    def wait_for_step(self, tls_id: str):
        """Called by AgentFacadeVecEnv.step_wait() in a PPO thread."""
        self.step_barrier.wait()
        # Barrier triggers _execute_step, which fully resolves self.obs/rewards etc.
        return self.obs[tls_id], self.rewards[tls_id], self.dones[tls_id], self.infos[tls_id]
        
    def _execute_step(self):
        """Called exclusively & once by the step barrier just before unblocking threads."""
        
        # Multiplex: Convert {J1: [env0_act, env1_act], J2: [env0_act, env1_act]} 
        # Into list: [ {J1: env0_act, J2: env0_act}, {J1: env1_act, J2: env1_act} ]
        worker_actions = []
        for i in range(self.num_envs):
            
            # Action array handling - discrete actions might be (4,1) or (4,) from PPO
            # we need atomic int types for sumo
            worker_actions.append({
                t: int(np.squeeze(self.actions[t][i])) for t in self.tls_ids
            })
            
        # Dispatch to 4 processes
        self.multi_env.step_async(worker_actions)
        results = self.multi_env.step_wait()
        
        self._step_count_for_fps += self.num_envs
        self.total_master_steps += self.delta_time * self.num_envs
        
        total_queue_this_step = 0
        total_reward_this_step = 0
        wait_time_this_step = 0
        
        # Re-scatter dict lists into vertical numpy arrays per agent
        for t in self.tls_ids:
            self.obs[t] = np.stack([res[0][t] for res in results])
            self.rewards[t] = np.array([res[4].get("per_junction", {}).get(t, {}).get("reward", 0.0) for res in results], dtype=np.float32)
            
            # Done if terminated or truncated (SB3 requirement)
            dones_array = np.array([res[2] or res[3] for res in results], dtype=np.bool_)
            self.dones[t] = dones_array
            
            infos_list = []
            for i, res in enumerate(results):
                info_dict = res[4].get("per_junction", {}).get(t, {}).copy()
                if dones_array[i] and "terminal_observation" in res[4]:
                    info_dict["terminal_observation"] = res[4]["terminal_observation"][t]
                infos_list.append(info_dict)
            self.infos[t] = infos_list
            
        progress_pct = self._step_count_for_fps / self.total_timesteps
        
        target_scale = 0.6
        target_grade = 0
        if progress_pct >= 0.40:
            target_scale = 1.0
            target_grade = 2
        elif progress_pct >= 0.15:
            target_scale = 0.8
            target_grade = 1
            
        if self.current_grade_idx != target_grade:
            self.current_grade_idx = target_grade
            print(f"\n{'='*50}\n[CURRICULUM] Promoting to Grade {target_grade + 1}: scale={target_scale} (Progress: {progress_pct*100:.1f}%)")
            self.multi_env.set_scale(target_scale)
            self._last_scale = target_scale
            
        # Logging Telemetry (every 100 simulation seconds equivalent)
        # Because we parallelize 4 envs, we'll check if master_step % 100 == 0 approx
        if self._step_count_for_fps % (100 // self.delta_time) == 0:
            for res in results:
                total_queue_this_step += sum(res[4]["per_junction"][t]["queue_length"] for t in self.tls_ids)
                total_reward_this_step += res[1]
                wait_time_this_step += res[4]["wait_time_total"]
                
            elapsed = time.time() - self._wall_start
            fps = self._step_count_for_fps / elapsed if elapsed > 0 else 0
            
            wandb.log({
                "env/queue_length_avg": total_queue_this_step / self.num_envs,
                "env/wait_time_avg": wait_time_this_step / self.num_envs,
                "env/reward_avg": total_reward_this_step / self.num_envs,
                "perf/fps": fps,
                "env/scale": self._last_scale
            })
            
            print(
                f"[{self._step_count_for_fps:>6d}/{self.total_timesteps}] "
                f"scale={self._last_scale:.1f}  "
                f"fps={fps:.0f}  "
                f"queue={total_queue_this_step/self.num_envs:.0f}  "
                f"wait={wait_time_this_step/self.num_envs:.0f}  "
                f"reward={total_reward_this_step/self.num_envs:.1f}  "
                f"progress={progress_pct*100:.1f}%"
            )
            
        self.actions.clear()
        
    def _execute_reset(self):
        """Called by reset barrier on very first rollout loop collection initialization."""
        results = self.multi_env.reset()
        for t in self.tls_ids:
            self.obs[t] = np.stack([res[0][t] for res in results])
            self.infos[t] = [res[1].get("per_junction", {}).get(t, {}) for res in results]


class AgentFacadeVecEnv(VecEnv):
    """Facade exposing a Vectorized gym interface blocking on a multiprocess coordinator."""

    def __init__(self, coordinator: SubprocCoordinator, tls_id: str, num_envs: int, obs_space: gym.Space, act_space: gym.Space):
        super().__init__(num_envs, obs_space, act_space)
        self.coordinator = coordinator
        self.tls_id = tls_id
        self.render_mode = None

    def step_async(self, actions: np.ndarray) -> None:
        self.coordinator.set_action(self.tls_id, actions)

    def step_wait(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        return self.coordinator.wait_for_step(self.tls_id)

    def reset(self) -> np.ndarray:
        self.coordinator.reset_barrier.wait()
        return self.coordinator.obs[self.tls_id]

    def close(self) -> None:
        pass

    def get_attr(self, attr_name, indices=None):
        return [None for _ in range(self.num_envs)]

    def set_attr(self, attr_name, value, indices=None):
        pass

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        pass

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False for _ in range(self.num_envs)]


def train_parallel(
    total_timesteps: int = 500000,
    port: int = 8000,
    run_name: str = "c8-parallel-ppo",
    num_envs: int = 4
):
    """Entry point for parallel independent multi-agent training."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    start_metrics_server(port)
    
    wandb_key = os.environ.get("WANDB_API_KEY")
    has_netrc = pathlib.Path("~/.netrc").expanduser().exists()
    has_credentials = bool(wandb_key) or has_netrc
    
    is_sanity = total_timesteps < 1000 or (run_name and run_name.lower().startswith("sanity"))
    
    if is_sanity:
        mode = "disabled"
    elif not has_credentials:
        mode = "offline"
    else:
        mode = "online"

    print(f"[wandb] Initializing in {mode} mode (sanity={is_sanity}, key_present={bool(wandb_key)})")

    wandb.init(
        project="marl-traffic",
        name=run_name,
        mode=mode,
        config={
            "total_timesteps": total_timesteps, 
            "curriculum": True, 
            "scale_stages": [0.6, 0.8, 1.0],
            "num_envs": num_envs,
            "architecture": "parallel_multiprocess"
        },
    )

    # 1. Instantiate Dummy to extract API signatures natively 
    print("[Master] Extracting dynamic junction signatures...")
    dummy_env = make_env("cologne8_corridor", use_gui=False, max_steps=3600, scale=0.6)
    tls_ids = dummy_env.tls_ids
    obs_spaces = dummy_env.observation_space.spaces
    act_spaces = dummy_env.action_space.spaces
    delta_time = dummy_env.delta_time
    dummy_env.close()
    
    print(f"Discovered {len(tls_ids)} Junctions. Spinning up {num_envs} Parallel Workers...")
    
    # 2. Build Subproc Manager
    def create_env_fn():
        return make_env("cologne8_corridor", use_gui=False, max_steps=3600, scale=0.6)
        
    multi_env = MultiAgentSharedSubproc([create_env_fn for _ in range(num_envs)])
    
    # 3. Build Coordinator and VecEnv Facades
    coordinator = SubprocCoordinator(multi_env, tls_ids, total_timesteps, delta_time, num_envs)
    facades = {
        t: AgentFacadeVecEnv(coordinator, t, num_envs, obs_spaces[t], act_spaces[t]) for t in tls_ids
    }
    
    # 4. Attach independent PPO models to VecEnvs
    models = {
        t: PPO(
            "MlpPolicy", 
            facades[t], 
            verbose=0, 
            n_steps=240,     # Max steps is 3600 delta=15 -> 240 agent decisions per episode
            batch_size=60, 
            learning_rate=3e-4,
            gamma=0.99
        )
        for t in tls_ids
    }
    
    threads = []
    
    saved_paths: list[str] = []

    def learn_agent(t: str, model: PPO):
        print(f"[{t}] PPO Thread Started")
        model.learn(total_timesteps=total_timesteps)
        save_path = str(MODELS_DIR / f"ppo_c8_parallel_{t}.zip")
        model.save(save_path)
        saved_paths.append(save_path)
        print(f"[{t}] PPO Thread Finished & Saved")

    start_t = time.time()
    for t in tls_ids:
        th = threading.Thread(target=learn_agent, args=(t, models[t]))
        threads.append(th)
        th.start()
        
    for th in threads:
        th.join()
        
    wall_clock = time.time() - start_t
    run_name = wandb.run.name
    run_id   = wandb.run.id
    print(f"\n[{run_name}] Complete! {total_timesteps} steps in {wall_clock:.1f}s (Overall FPS: {total_timesteps/wall_clock:.1f})")
    
    # ── Upload model weights as a versioned W&B artifact ──────────────────
    print("[wandb] Uploading model weights as artifact...")
    artifact = wandb.Artifact(
        name=f"c8-ppo-models-{run_name}",
        type="model",
        description=f"Parallel heterogeneous PPO weights — {len(tls_ids)} agents, {total_timesteps} steps",
        metadata={"num_agents": len(tls_ids), "num_envs": num_envs, "total_timesteps": total_timesteps},
    )
    for p in saved_paths:
        artifact.add_file(p)
    wandb.log_artifact(artifact)
    print(f"[wandb] Artifact logged: c8-ppo-models-{run_name}")

    # ── Persist run metadata so c8-compare can resume this run ──────────
    metadata = {"run_id": run_id, "run_name": run_name, "project": "marl-traffic"}
    meta_path = MODELS_DIR / "c8_run_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[wandb] Run metadata saved to {meta_path}")

    multi_env.close()
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--timesteps", type=int, default=10000)
    parser.add_argument("--run-name", type=str, default="c8-parallel-ppo")
    parser.add_argument("--num-envs", type=int, default=4)
    args = parser.parse_args()
    
    if args.train:
        train_parallel(
            total_timesteps=args.timesteps, 
            run_name=args.run_name,
            num_envs=args.num_envs
        )
