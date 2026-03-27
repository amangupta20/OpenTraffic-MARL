"""
MAPPO Network Components for Centralized Training, Decentralized Execution (CTDE).

Contains:
  - MAPPOActor:        Per-agent local policy network (obs_i → action logits)
  - MAPPOCritic:       Shared centralized value network (global_state → V(s))
  - MAPPORolloutBuffer: Per-agent trajectory storage with GAE computation

Design choices:
  - Each junction gets its own Actor (heterogeneous obs/action dims)
  - All junctions share ONE Critic seeing global state (all obs zero-padded & concatenated)
  - Per-junction LOCAL rewards drive per-actor advantages (not global sum)
  - Zero-padding to max_obs_dim handles heterogeneous obs dimensions cleanly
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


# ─────────────────────────────────────────────────────────────────────────────
# Networks
# ─────────────────────────────────────────────────────────────────────────────

class MAPPOActor(nn.Module):
    """
    Decentralized actor for a single junction.
    Input:  local observation  o_i  (shape: obs_dim,)
    Output: action logits           (shape: n_actions,)
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    def get_action_and_logprob(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action + return log-prob and entropy."""
        logits = self.forward(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy()

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate log-prob and entropy of given actions (for PPO update)."""
        logits = self.forward(obs)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy()


class MAPPOCritic(nn.Module):
    """
    Centralized value network — shared across all agents.
    Input:  global state  s = concat(pad(o_1), ..., pad(o_N))  (shape: global_dim,)
    Output: scalar value  V(s)
    """

    def __init__(self, global_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden // 2),
            nn.Tanh(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        return self.net(global_state).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# Rollout Buffer
# ─────────────────────────────────────────────────────────────────────────────

class MAPPORolloutBuffer:
    """
    Stores one rollout of transitions for a SINGLE agent, plus global state
    for the shared centralized critic.

    Fields per-step:
      local_obs     : (T, obs_dim)        — agent-local observation
      global_state  : (T, global_dim)     — all-agent concat obs
      actions       : (T,)                — discrete action taken
      log_probs     : (T,)                — log π(a|o) at collection time
      rewards       : (T,)                — per-junction local reward
      values        : (T,)                — V(s) from centralized critic
      dones         : (T,)                — episode-done flag
    """

    def __init__(self, n_steps: int, obs_dim: int, global_dim: int) -> None:
        self.n_steps = n_steps
        self.obs_dim = obs_dim
        self.global_dim = global_dim
        self.reset()

    def reset(self) -> None:
        self.local_obs    = np.zeros((self.n_steps, self.obs_dim),    dtype=np.float32)
        self.global_state = np.zeros((self.n_steps, self.global_dim), dtype=np.float32)
        self.actions      = np.zeros(self.n_steps,                    dtype=np.int64)
        self.log_probs    = np.zeros(self.n_steps,                    dtype=np.float32)
        self.rewards      = np.zeros(self.n_steps,                    dtype=np.float32)
        self.values       = np.zeros(self.n_steps,                    dtype=np.float32)
        self.dones        = np.zeros(self.n_steps,                    dtype=np.float32)
        self.ptr = 0

    def add(
        self,
        local_obs: np.ndarray,
        global_state: np.ndarray,
        action: int,
        log_prob: float,
        reward: float,
        value: float,
        done: bool,
    ) -> None:
        assert self.ptr < self.n_steps, "Buffer is full — call compute_returns() first"
        i = self.ptr
        self.local_obs[i]    = local_obs
        self.global_state[i] = global_state
        self.actions[i]      = action
        self.log_probs[i]    = log_prob
        self.rewards[i]      = reward
        self.values[i]       = value
        self.dones[i]        = float(done)
        self.ptr += 1

    def is_full(self) -> bool:
        return self.ptr >= self.n_steps

    def compute_returns_and_advantages(
        self,
        last_value: float,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute GAE advantages and discounted returns.
        Returns: (advantages, returns)  — both shape (T,)
        """
        T = self.ptr
        advantages = np.zeros(T, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(T)):
            if t == T - 1:
                next_non_terminal = 1.0 - self.dones[t]
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_value = self.values[t + 1]

            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + self.values[:T]
        return advantages, returns

    def get_tensors(
        self, device: torch.device
    ) -> dict[str, torch.Tensor]:
        """Return all stored data as PyTorch tensors on the requested device."""
        T = self.ptr
        return {
            "local_obs":    torch.tensor(self.local_obs[:T],    device=device),
            "global_state": torch.tensor(self.global_state[:T], device=device),
            "actions":      torch.tensor(self.actions[:T],      device=device),
            "log_probs":    torch.tensor(self.log_probs[:T],    device=device),
            "rewards":      torch.tensor(self.rewards[:T],      device=device),
            "values":       torch.tensor(self.values[:T],       device=device),
            "dones":        torch.tensor(self.dones[:T],        device=device),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_global_state(
    obs_dict: dict[str, np.ndarray],
    tls_ids: list[str],
    max_obs_dim: int,
) -> np.ndarray:
    """
    Concatenate all agents' observations into one global state vector,
    zero-padding each obs to max_obs_dim to handle heterogeneous dims.
    Shape: (N * max_obs_dim,)
    """
    parts = []
    for tls_id in tls_ids:
        obs = obs_dict[tls_id]
        pad = max_obs_dim - len(obs)
        parts.append(np.concatenate([obs, np.zeros(pad, dtype=np.float32)]))
    return np.concatenate(parts)
