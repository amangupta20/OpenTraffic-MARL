"""
Feudal MARL Network Components.

Contains:
  - ManagerActorContinuous: Global State → Continuous goal vectors for all agents (shape: N_AGENTS * K_GOAL_DIM)
  - ManagerCritic:          Global State → Centralized V(s) for the Manager
  - FeudalWorkerActor:      Local Obs + Goal → Action logits
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal

# ─────────────────────────────────────────────────────────────────────────────
# Feudal Manager Networks (Macro-level)
# ─────────────────────────────────────────────────────────────────────────────

class ManagerActorContinuous(nn.Module):
    """
    Centralized macro-policy network.
    Input:  Global State (shape: global_dim,)
    Output: Continuous subgoals (shape: n_agents * k_goal_dim,)
    """
    def __init__(self, global_dim: int, n_agents: int, k_goal_dim: int = 3, hidden: int = 128) -> None:
        super().__init__()
        self.output_dim = n_agents * k_goal_dim
        
        self.net = nn.Sequential(
            nn.Linear(global_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, self.output_dim),
            nn.Tanh() # Bounding goals between -1 and 1
        )
        # Log std is independent of state, a standard continuous PPO practice
        self.log_std = nn.Parameter(torch.zeros(self.output_dim))

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        return self.net(global_state)

    def get_action_and_logprob(
        self, global_state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample goals and return log-prob and entropy."""
        mu = self.forward(global_state)
        std = self.log_std.exp().expand_as(mu)
        dist = Normal(mu, std)
        
        action = dist.sample()
        # Bound actions between -1 and 1 if desired, but our mean is bounded.
        # Log prob is the sum across dimensions
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return action, log_prob, entropy

    def evaluate_actions(
        self, global_state: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate log-prob and entropy of given goals."""
        mu = self.forward(global_state)
        std = self.log_std.exp().expand_as(mu)
        dist = Normal(mu, std)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy

class ManagerCritic(nn.Module):
    """
    Centralized value network for the Manager.
    Input: Global State
    Output: Single Global V(s)
    """
    def __init__(self, global_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1)
        )

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        return self.net(global_state).squeeze(-1)

# ─────────────────────────────────────────────────────────────────────────────
# Feudal Worker Networks (Micro-level)
# ─────────────────────────────────────────────────────────────────────────────

class FeudalWorkerActor(nn.Module):
    """
    Decentralized actor for a single junction under a Manager.
    Input:  Local observation o_i (shape: obs_dim) + Goal g_i (shape: k_goal_dim)
    Output: Action logits (shape: n_actions)
    """
    def __init__(self, obs_dim: int, k_goal_dim: int, n_actions: int, hidden: int = 64) -> None:
        super().__init__()
        # Input is concatenation of obs and goal
        self.net = nn.Sequential(
            nn.Linear(obs_dim + k_goal_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, obs: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, goal], dim=-1)
        return self.net(x)

    def get_action_and_logprob(
        self, obs: torch.Tensor, goal: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.forward(obs, goal)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy()

    def evaluate_actions(
        self, obs: torch.Tensor, goal: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.forward(obs, goal)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy()

# ─────────────────────────────────────────────────────────────────────────────
# Buffers
# ─────────────────────────────────────────────────────────────────────────────

class ManagerRolloutBuffer:
    """Stores transitions for the Manager (macro-level steps)."""
    def __init__(self, n_steps: int, global_dim: int, output_dim: int) -> None:
        self.n_steps = n_steps
        self.global_dim = global_dim
        self.output_dim = output_dim
        self.reset()

    def reset(self) -> None:
        self.global_state = np.zeros((self.n_steps, self.global_dim), dtype=np.float32)
        self.actions      = np.zeros((self.n_steps, self.output_dim), dtype=np.float32)
        self.log_probs    = np.zeros(self.n_steps, dtype=np.float32)
        self.rewards      = np.zeros(self.n_steps, dtype=np.float32)
        self.values       = np.zeros(self.n_steps, dtype=np.float32)
        self.dones        = np.zeros(self.n_steps, dtype=np.float32)
        self.ptr = 0

    def add(self, gs, a, lp, r, v, d) -> None:
        if self.ptr >= self.n_steps: return
        i = self.ptr
        self.global_state[i] = gs
        self.actions[i]      = a
        self.log_probs[i]    = lp
        self.rewards[i]      = r
        self.values[i]       = v
        self.dones[i]        = float(d)
        self.ptr += 1

    def is_full(self) -> bool:
        return self.ptr >= self.n_steps

    def compute_returns_and_advantages(self, last_values: list[float], gamma: float, gae_lambda: float, num_envs: int = 1):
        T = self.ptr
        advantages = np.zeros(T, dtype=np.float32)
        last_gae = np.zeros(num_envs, dtype=np.float32)
        for t in reversed(range(T)):
            env_idx = t % num_envs
            is_last_step = (t >= T - num_envs)
            
            next_non_terminal = 1.0 - self.dones[t] if is_last_step else 1.0 - self.dones[t + num_envs]
            next_value = last_values[env_idx] if is_last_step else self.values[t + num_envs]
            
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            last_gae[env_idx] = delta + gamma * gae_lambda * next_non_terminal * last_gae[env_idx]
            advantages[t] = last_gae[env_idx]
        returns = advantages + self.values[:T]
        return advantages, returns

    def get_tensors(self, device: torch.device):
        T = self.ptr
        return {
            "global_state": torch.tensor(self.global_state[:T], device=device),
            "actions":      torch.tensor(self.actions[:T], device=device),
            "log_probs":    torch.tensor(self.log_probs[:T], device=device),
            "rewards":      torch.tensor(self.rewards[:T], device=device),
            "values":       torch.tensor(self.values[:T], device=device),
            "dones":        torch.tensor(self.dones[:T], device=device),
        }

class FeudalWorkerRolloutBuffer:
    """Stores transitions for a Worker (micro-level steps)."""
    def __init__(self, n_steps: int, obs_dim: int, k_goal_dim: int, global_dim: int) -> None:
        self.n_steps = n_steps
        self.obs_dim = obs_dim
        self.k_goal_dim = k_goal_dim
        self.global_dim = global_dim
        self.reset()

    def reset(self) -> None:
        self.local_obs    = np.zeros((self.n_steps, self.obs_dim), dtype=np.float32)
        self.goals        = np.zeros((self.n_steps, self.k_goal_dim), dtype=np.float32)
        self.global_state = np.zeros((self.n_steps, self.global_dim), dtype=np.float32)
        self.actions      = np.zeros(self.n_steps, dtype=np.int64)
        self.log_probs    = np.zeros(self.n_steps, dtype=np.float32)
        self.rewards      = np.zeros(self.n_steps, dtype=np.float32)
        self.values       = np.zeros(self.n_steps, dtype=np.float32)
        self.dones        = np.zeros(self.n_steps, dtype=np.float32)
        self.ptr = 0

    def add(self, obs, goal, gs, a, lp, r, v, d) -> None:
        if self.ptr >= self.n_steps: return
        i = self.ptr
        self.local_obs[i]    = obs
        self.goals[i]        = goal
        self.global_state[i] = gs
        self.actions[i]      = a
        self.log_probs[i]    = lp
        self.rewards[i]      = r
        self.values[i]       = v
        self.dones[i]        = float(d)
        self.ptr += 1

    def is_full(self) -> bool:
        return self.ptr >= self.n_steps

    def compute_returns_and_advantages(self, last_values: list[float], gamma: float, gae_lambda: float, num_envs: int = 1):
        T = self.ptr
        advantages = np.zeros(T, dtype=np.float32)
        last_gae = np.zeros(num_envs, dtype=np.float32)
        for t in reversed(range(T)):
            env_idx = t % num_envs
            is_last_step = (t >= T - num_envs)
            
            next_non_terminal = 1.0 - self.dones[t] if is_last_step else 1.0 - self.dones[t + num_envs]
            next_value = last_values[env_idx] if is_last_step else self.values[t + num_envs]
            
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            last_gae[env_idx] = delta + gamma * gae_lambda * next_non_terminal * last_gae[env_idx]
            advantages[t] = last_gae[env_idx]
        returns = advantages + self.values[:T]
        return advantages, returns

    def get_tensors(self, device: torch.device):
        T = self.ptr
        return {
            "local_obs":    torch.tensor(self.local_obs[:T], device=device),
            "goals":        torch.tensor(self.goals[:T], device=device),
            "global_state": torch.tensor(self.global_state[:T], device=device),
            "actions":      torch.tensor(self.actions[:T], device=device),
            "log_probs":    torch.tensor(self.log_probs[:T], device=device),
            "rewards":      torch.tensor(self.rewards[:T], device=device),
            "values":       torch.tensor(self.values[:T], device=device),
            "dones":        torch.tensor(self.dones[:T], device=device),
        }
