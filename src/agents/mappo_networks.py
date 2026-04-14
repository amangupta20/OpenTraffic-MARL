"""
MAPPO Network Components for Centralized Training, Decentralized Execution (CTDE).

Contains:
  - MAPPOActor:        Per-agent local policy network (obs_i → action logits)
  - MAPPOCritic:       Shared-trunk centralized value network with per-agent heads
                       (global_state → V_i(s) for each agent i)
  - MAPPORolloutBuffer: Per-agent trajectory storage with GAE computation

Design choices:
  - Each junction gets its own Actor (heterogeneous obs/action dims)
  - All junctions share ONE Critic trunk seeing global state, but each junction
    has its OWN value head V_i(s). This prevents reward-scale contamination
    when junctions have wildly different reward magnitudes (a 5-way complex
    junction gets ~10x more reward signal than a T-junction).
  - Per-junction LOCAL rewards → per-agent advantages via their own V_i(s)
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
    Centralized value network — shared trunk, per-agent output heads.

    Architecture:
        global_state  (global_dim,)
            → Shared Trunk: Linear(global_dim, 128) → Tanh → Linear(128, 64) → Tanh
                → Per-agent heads: Linear(64, 1) × N_agents

    Each agent i gets its own V_i(s) estimate so different reward scales across
    heterogeneous junctions don't contaminate each other's value targets.

    Args:
        global_dim:  Length of the concatenated+padded global state vector.
        agent_ids:   Ordered list of agent IDs (defines head ordering/indexing).
        hidden:      Width of the shared trunk (default 128).
    """

    def __init__(self, global_dim: int, agent_ids: list[str], hidden: int = 128) -> None:
        super().__init__()
        self.agent_ids = agent_ids
        self.n_agents  = len(agent_ids)

        # Shared encoder trunk — sees the full global state
        self.trunk = nn.Sequential(
            nn.Linear(global_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden // 2),
            nn.Tanh(),
        )

        # One independent value head per agent
        self.heads = nn.ModuleDict({
            agent_id: nn.Linear(hidden // 2, 1)
            for agent_id in agent_ids
        })

    def forward(self, global_state: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            global_state: Tensor of shape (batch, global_dim) or (global_dim,)
        Returns:
            dict mapping agent_id → value tensor of shape (batch,)
        """
        features = self.trunk(global_state)
        return {
            agent_id: self.heads[agent_id](features).squeeze(-1)
            for agent_id in self.agent_ids
        }

    def get_value(self, global_state: torch.Tensor, agent_id: str) -> torch.Tensor:
        """Convenience: get V_i(s) for a single agent. Shape: scalar or (batch,)."""
        features = self.trunk(global_state)
        return self.heads[agent_id](features).squeeze(-1)


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
