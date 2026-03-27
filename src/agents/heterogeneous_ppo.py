"""
Heterogeneous multi-agent training for the Bangalore MG Road corridor using PPO.

Spawns independent PPO agents per junction, communicating with a central 
SUMO simulation via a synchronization barrier (Coordinator pattern).
"""

import argparse
import os
import pathlib
import threading
import time
from typing import Any

import gymnasium as gym
import numpy as np
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from src.envs import make_env
from src.utils.metrics import start_metrics_server, update

MODELS_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "models"


class Coordinator:
    """Synchronizes multi-agent stepping and resetting for independent PPO loops."""

    def __init__(self, master_env: gym.Env, tls_ids: list[str], total_timesteps: int):
        self.master_env = master_env
        self.tls_ids = tls_ids
        self.num_agents = len(tls_ids)
        self.total_timesteps = total_timesteps
        
        # State buffers
        self.actions = {}
        self.obs = {}
        self.rewards = {}
        self.dones = {t: False for t in tls_ids}
        self.truncs = {t: False for t in tls_ids}
        self.infos = {}
        
        # We need two barriers: one for step(), one for reset()
        self.step_barrier = threading.Barrier(self.num_agents, action=self._step_master)
        self.reset_barrier = threading.Barrier(self.num_agents, action=self._reset_master)
        
        # Curriculum tracking
        self.curriculum_grades = [0.2, 0.4, 0.6]
        self.current_grade_idx = 0
        self.episodes_completed = 0
        self.total_master_steps = 0
        
    def _step_master(self):
        """Called by the last agent to reach the step barrier; steps the SUMO simulation."""
        new_obs, global_reward, terminated, truncated, info = self.master_env.step(self.actions)
        
        self.total_master_steps += self.master_env.delta_time
        
        # Dispatch rewards correctly to agents
        for t in self.tls_ids:
            self.obs[t] = new_obs[t]
            self.rewards[t] = info["per_junction"][t]["reward"]
            self.dones[t] = terminated
            self.truncs[t] = truncated
            self.infos[t] = info

        # Clear actions
        self.actions = {}
        
        # Log to wandb from master thread hook
        if "step" in info and info["step"] % 100 == 0:
            total_queue = sum(info["per_junction"][t]["queue_length"] for t in self.tls_ids)
            wandb.log({
                "env/queue_length": total_queue,
                "env/wait_time": info["wait_time_total"],
                "env/reward": sum(self.rewards.values()),
                "env/scale": info.get("scale", self.master_env.scale),
                "env/master_step": self.total_master_steps,
            })
            
            # Print to stdout occasionally
            if info["step"] % 300 == 0:
                print(f"[Master] scale={self.master_env.scale:.1f} queue={total_queue:.0f} "
                      f"wait={info['wait_time_total']:.0f} reward={sum(self.rewards.values()):.1f}")

    def _reset_master(self):
        """Called by the last agent to reach the reset barrier; resets the SUMO simulation."""
        # Curriculum promotion logic based on training progress
        self.episodes_completed += 1
        
        # Approximate agent steps completed 
        # (max_steps // delta_time = 360 steps per episode)
        steps_per_episode = self.master_env.max_steps // self.master_env.delta_time
        agent_steps_completed = self.episodes_completed * steps_per_episode
        progress = agent_steps_completed / self.total_timesteps
        
        target_scale = 0.2
        target_grade = 0
        
        if progress >= 0.40:    # Last 60% of steps runs on 0.6
            target_scale = 0.6
            target_grade = 2
        elif progress >= 0.15:  # Next 25% of steps runs on 0.4
            target_scale = 0.4
            target_grade = 1
            
        if self.current_grade_idx != target_grade:
            self.current_grade_idx = target_grade
            print(f"\n{'='*50}\n[CURRICULUM] Promoting to Grade {target_grade + 1}: scale={target_scale} (Progress: {progress*100:.1f}%)\n{'='*50}\n")
            self.master_env.scale = target_scale
            
        new_obs, new_info = self.master_env.reset()
        
        for t in self.tls_ids:
            self.obs[t] = new_obs[t]
            self.dones[t] = False
            self.truncs[t] = False
            self.infos[t] = new_info
            
        print(f"[Master] Reset complete. Starting episode {self.episodes_completed} at scale {self.master_env.scale}.")


class AgentFacadeEnv(gym.Env):
    """Facade exposing a single-agent Gymnasium API, blocking on the Coordinator."""

    def __init__(self, coordinator: Coordinator, tls_id: str):
        super().__init__()
        self.coordinator = coordinator
        self.tls_id = tls_id
        
        self.observation_space = self.coordinator.master_env.observation_space.spaces[tls_id]
        self.action_space = self.coordinator.master_env.action_space.spaces[tls_id]

    def reset(self, *, seed=None, options=None) -> tuple[np.ndarray, dict[str, Any]]:
        self.coordinator.reset_barrier.wait()
        return self.coordinator.obs[self.tls_id], self.coordinator.infos[self.tls_id]

    def step(self, action: np.ndarray | int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        # SB3 might pass action as a numpy array with 1 element
        act_val = int(action.item()) if isinstance(action, np.ndarray) else int(action)
        
        self.coordinator.actions[self.tls_id] = act_val
        self.coordinator.step_barrier.wait()
        
        return (
            self.coordinator.obs[self.tls_id],
            self.coordinator.rewards[self.tls_id],
            self.coordinator.dones[self.tls_id],
            self.coordinator.truncs[self.tls_id],
            self.coordinator.infos[self.tls_id],
        )


def train_heterogeneous(
    total_timesteps: int = 50000,
    port: int = 8000,
    run_name: str = "blr-hetero-ppo",
):
    """Entry point for training."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    start_metrics_server(port)
    
    wandb.init(
        project="marl-traffic",
        name=run_name,
        config={
            "total_timesteps": total_timesteps, 
            "curriculum": True, 
            "scale_stages": [0.2, 0.4, 0.6]
        },
    )

    master_env = make_env(
        "bangalore_corridor", 
        use_gui=False, 
        max_steps=1800,  # 30 mins per episode
        scale=0.2        # Initial grade
    )
    
    tls_ids = master_env.tls_ids
    print(f"Discovered {len(tls_ids)} Junctions: {tls_ids}")
    
    coordinator = Coordinator(master_env, tls_ids, total_timesteps)
    facades = {t: AgentFacadeEnv(coordinator, t) for t in tls_ids}
    
    # We must seed the coordinator with the initial reset before training starts, 
    # but the very first thing `model.learn()` does is call `env.reset()`.
    # Therefore, we just let the models call reset immediately when they start.
    
    models = {
        t: PPO(
            "MlpPolicy", 
            facades[t], 
            verbose=0, 
            n_steps=360,     # Max steps is 1800 delta=5 -> 360 steps per episode. Align n_steps so PPO updates per episode.
            batch_size=60, 
            learning_rate=3e-4,
            gamma=0.99
        )
        for t in tls_ids
    }
    
    threads = []
    
    def learn_agent(t: str, model: PPO):
        print(f"[{t}] PPO Thread Started")
        model.learn(total_timesteps=total_timesteps)
        model.save(str(MODELS_DIR / f"ppo_blr_{t}.zip"))
        print(f"[{t}] PPO Thread Finished & Saved")

    for t in tls_ids:
        th = threading.Thread(target=learn_agent, args=(t, models[t]))
        threads.append(th)
        th.start()
        
    for th in threads:
        th.join()
        
    master_env.close()
    wandb.finish()
    print("Training Complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--timesteps", type=int, default=10000)
    parser.add_argument("--run-name", type=str, default="blr-hetero-ppo")
    args = parser.parse_args()
    
    if args.train:
        train_heterogeneous(total_timesteps=args.timesteps, run_name=args.run_name)
