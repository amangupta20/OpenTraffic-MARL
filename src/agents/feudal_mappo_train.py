"""
Feudal MARL Training for Cologne8 Corridor — Parallel Edition.

Dual PPO Architecture:
- Macro Manager: Acts every `C` steps (e.g. 60s), observing Global State, outputting continuous Goal Priorities `g_i`.
                 Receives purely Extrinsic Environment Reward (Global).
- Micro Worker:  Acts every step (15s), observing Local State + `g_i`.
                 Receives Extrinsic Reward + Intrinsic Reward (alignment with Manager goal priorities).

Parallelism:
  N independent SUMO processes (MultiAgentSharedSubproc) run in parallel subprocesses.
  The Manager + Worker networks remain on the main process.
  Each logical step sends actions to all N envs simultaneously and receives N transitions,
  filling the rollout buffers N× faster with no change to the update logic.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import time

import numpy as np
import torch
import torch.nn as nn
import wandb

from src.envs import make_env
from src.envs.multi_agent_subproc import MultiAgentSharedSubproc
# Import Feudal specific networks
from src.agents.feudal_networks import (
    ManagerActorContinuous,
    ManagerCritic,
    ManagerRolloutBuffer,
    FeudalWorkerActor,
    FeudalWorkerRolloutBuffer,
)
from src.agents.mappo_networks import MAPPOCritic
from src.utils.metrics import start_metrics_server

MODELS_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "models"

# ─────────────────────────────────────────────────────────────────────────────
# Hyper-parameters
# ─────────────────────────────────────────────────────────────────────────────
# General PPO Params
GAMMA          = 0.99
GAE_LAMBDA     = 0.95
CLIP_EPS       = 0.2
MAX_GRAD_NORM  = 0.5
REWARD_CLIP    = 500.0

# Hierarchy Scaling
MANAGER_C       = 4            # Manager operates every 4 environment steps (60s)
WORKER_N_STEPS  = 480          # Worker rollout steps before update (per env)
MANAGER_N_STEPS = WORKER_N_STEPS // MANAGER_C  # Manager rollout steps
N_EPOCHS        = 10           # PPO update epochs

# Manager Learning Rates & Coefs
MANGER_LR       = 1e-4
MANAGER_ENTROPY = 0.05
MANAGER_VALUE   = 0.5
MANGER_BATCH    = MANAGER_N_STEPS // 4  # e.g., 120//4 = 30

# Worker Learning Rates & Coefs
WORKER_LR       = 3e-4
WORKER_ENTROPY  = 0.05
WORKER_VALUE    = 0.5
WORKER_BATCH    = WORKER_N_STEPS // 8   # e.g., 480//8 = 60

# Goal Definition
K_GOAL_DIM      = 3            # Priorities: Queue Minimization, Wait Minimization, Coordination/Starvation
GOAL_DROPOUT_P  = 0.1          # 10% chance to drop Manager's goal, fostering worker autonomy
ALPHA           = 0.5          # Mixing parameter: R_worker = ALPHA * R_ext + (1 - ALPHA) * R_int

# Curriculum: grades for the Cologne8 network
CURRICULUM = [
    (0.00, 0.60),   # Grade 1
    (0.15, 0.80),   # Grade 2
    (0.40, 1.00),   # Grade 3
    (0.70, 1.30),   # Grade 4
]


class FeudalTrainer:
    def __init__(
        self,
        total_timesteps: int,
        run_name: str = "feudal-cologne8",
        num_envs: int = 4,
        port: int = 8000,
    ) -> None:
        self.total_timesteps = total_timesteps
        self.run_name = run_name
        self.num_envs = num_envs
        self.device = torch.device("cpu")

        start_metrics_server(port)

        # ── Discover network topology ──────────────────────────────────────
        print("[Feudal] Discovering network topology...")
        _probe = make_env("cologne8_corridor", use_gui=False, max_steps=3600, scale=0.6)
        self.tls_ids: list[str] = _probe.tls_ids
        self.obs_dims: dict[str, int] = {
            t: _probe.observation_space.spaces[t].shape[0] for t in self.tls_ids
        }
        self.act_dims: dict[str, int] = {
            t: int(_probe.action_space.spaces[t].n) for t in self.tls_ids
        }
        self.delta_time: int = _probe.delta_time
        _sample_gs = _probe.get_global_state()
        self.global_dim = _sample_gs.shape[0]
        _probe.close()

        print(f"[Feudal] {len(self.tls_ids)} junctions | global_dim={self.global_dim} | num_envs={num_envs}")

        # ── Build Feudal Manager ──────────────────────────────────────────
        self.n_agents = len(self.tls_ids)
        self.manager_actor = ManagerActorContinuous(
            global_dim=self.global_dim,
            n_agents=self.n_agents,
            k_goal_dim=K_GOAL_DIM
        ).to(self.device)
        self.manager_critic = ManagerCritic(self.global_dim).to(self.device)

        self.manager_optim = torch.optim.Adam(
            list(self.manager_actor.parameters()) + list(self.manager_critic.parameters()),
            lr=MANGER_LR
        )
        # Buffer holds steps from ALL envs: MANAGER_N_STEPS * num_envs capacity
        self.manager_buffer = ManagerRolloutBuffer(
            MANAGER_N_STEPS * num_envs, self.global_dim, self.n_agents * K_GOAL_DIM
        )

        # ── Build Workers (CTDE) ──────────────────────────────────────────
        # Worker Critic sees Global State + Concatenated Goals
        worker_global_dim = self.global_dim + (self.n_agents * K_GOAL_DIM)

        self.worker_actors: dict[str, FeudalWorkerActor] = {
            t: FeudalWorkerActor(self.obs_dims[t], K_GOAL_DIM, self.act_dims[t]).to(self.device)
            for t in self.tls_ids
        }
        self.worker_critic = MAPPOCritic(worker_global_dim, self.tls_ids).to(self.device)

        self.worker_optims: dict[str, torch.optim.Adam] = {
            t: torch.optim.Adam(self.worker_actors[t].parameters(), lr=WORKER_LR)
            for t in self.tls_ids
        }
        self.worker_critic_optim = torch.optim.Adam(self.worker_critic.parameters(), lr=WORKER_LR)

        # Buffer capacity = WORKER_N_STEPS * num_envs
        self.worker_buffers: dict[str, FeudalWorkerRolloutBuffer] = {
            t: FeudalWorkerRolloutBuffer(
                WORKER_N_STEPS * num_envs, self.obs_dims[t], K_GOAL_DIM, worker_global_dim
            )
            for t in self.tls_ids
        }

        # ── State keeping ──────────────────────────────────────────────────
        self.current_scale = 0.6
        self.steps_collected = 0        # counts logical steps (each += num_envs)
        self._wall_start = time.time()
        self._last_log_time = self._wall_start
        self._last_log_steps = 0

        # Zero-Order Hold state: one goal vector per env
        self.zoh_goals = np.zeros(
            (num_envs, self.n_agents * K_GOAL_DIM), dtype=np.float32
        )
        # Manager step buffer values (stored per env until manager buffer is written)
        self._mgr_lp = np.zeros(num_envs, dtype=np.float32)
        self._mgr_v  = np.zeros(num_envs, dtype=np.float32)
        self._ep_mgr_reward = np.zeros(num_envs, dtype=np.float32)

    def _get_target_scale(self) -> float:
        progress = self.steps_collected / self.total_timesteps
        target = CURRICULUM[0][1]
        for thresh, scale in CURRICULUM:
            if progress >= thresh:
                target = scale
        return target

    def _update_manager(self):
        """Update Manager PPO networks"""
        with torch.no_grad():
            last_value = 0.0
        adv, ret = self.manager_buffer.compute_returns_and_advantages(last_value, GAMMA, GAE_LAMBDA)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        tensors = self.manager_buffer.get_tensors(self.device)
        gs_ts     = tensors["global_state"]
        a_ts      = tensors["actions"]
        old_lp_ts = tensors["log_probs"]
        ret_ts    = torch.tensor(ret, device=self.device)
        adv_ts    = torch.tensor(adv, device=self.device)

        T = self.manager_buffer.ptr
        indices = np.arange(T)

        metrics = {"fl/mgr_actor_loss": 0.0, "fl/mgr_critic_loss": 0.0, "fl/mgr_entropy": 0.0}
        n_updates = 0

        for _ in range(N_EPOCHS):
            np.random.shuffle(indices)
            for start in range(0, T, MANGER_BATCH):
                mb_idxs = indices[start:start + MANGER_BATCH]
                mb_gs    = gs_ts[mb_idxs]
                mb_a     = a_ts[mb_idxs]
                mb_ret   = ret_ts[mb_idxs]
                mb_adv   = adv_ts[mb_idxs]
                mb_old_lp = old_lp_ts[mb_idxs]

                new_lp, entropy = self.manager_actor.evaluate_actions(mb_gs, mb_a)
                ratio = torch.exp(new_lp - mb_old_lp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * mb_adv
                actor_loss  = -torch.min(surr1, surr2).mean()
                v_pred      = self.manager_critic(mb_gs)
                critic_loss = 0.5 * ((v_pred - mb_ret) ** 2).mean()
                loss        = actor_loss - MANAGER_ENTROPY * entropy.mean() + MANAGER_VALUE * critic_loss

                self.manager_optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.manager_actor.parameters(), MAX_GRAD_NORM)
                nn.utils.clip_grad_norm_(self.manager_critic.parameters(), MAX_GRAD_NORM)
                self.manager_optim.step()

                metrics["fl/mgr_actor_loss"]  += actor_loss.item()
                metrics["fl/mgr_critic_loss"]  += critic_loss.item()
                metrics["fl/mgr_entropy"]      += entropy.mean().item()
                n_updates += 1

        self.manager_buffer.reset()
        if n_updates > 0:
            for k in metrics:
                metrics[k] /= n_updates
        return metrics

    def _update_workers(self):
        """Update Worker PPO networks"""
        adv_dict, ret_dict = {}, {}
        for t in self.tls_ids:
            adv, ret = self.worker_buffers[t].compute_returns_and_advantages(0.0, GAMMA, GAE_LAMBDA)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            adv_dict[t] = adv
            ret_dict[t] = ret

        tensors  = {t: self.worker_buffers[t].get_tensors(self.device) for t in self.tls_ids}
        metrics  = {"fl/wk_actor_loss": 0.0, "fl/wk_critic_loss": 0.0, "fl/wk_entropy": 0.0}
        n_updates = 0

        T       = self.worker_buffers[self.tls_ids[0]].ptr
        indices = np.arange(T)

        for _ in range(N_EPOCHS):
            np.random.shuffle(indices)
            for start in range(0, T, WORKER_BATCH):
                mb_idxs = indices[start:start + WORKER_BATCH]
                self.worker_critic_optim.zero_grad()
                critic_loss_total = 0.0

                mb_w_gs       = tensors[self.tls_ids[0]]["global_state"][mb_idxs]
                v_preds_all   = self.worker_critic(mb_w_gs)

                for t in self.tls_ids:
                    mb_obs    = tensors[t]["local_obs"][mb_idxs]
                    mb_g      = tensors[t]["goals"][mb_idxs]
                    mb_a      = tensors[t]["actions"][mb_idxs]
                    mb_old_lp = tensors[t]["log_probs"][mb_idxs]
                    mb_ret    = torch.tensor(ret_dict[t][mb_idxs], device=self.device)
                    mb_adv    = torch.tensor(adv_dict[t][mb_idxs], device=self.device)

                    new_lp, entropy = self.worker_actors[t].evaluate_actions(mb_obs, mb_g, mb_a)
                    ratio  = torch.exp(new_lp - mb_old_lp)
                    surr1  = ratio * mb_adv
                    surr2  = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * mb_adv
                    actor_loss = -torch.min(surr1, surr2).mean()

                    self.worker_optims[t].zero_grad()
                    al_total = actor_loss - WORKER_ENTROPY * entropy.mean()
                    al_total.backward()
                    nn.utils.clip_grad_norm_(self.worker_actors[t].parameters(), MAX_GRAD_NORM)
                    self.worker_optims[t].step()

                    v_pred = v_preds_all[t]
                    c_loss = 0.5 * ((v_pred - mb_ret) ** 2).mean()
                    critic_loss_total = critic_loss_total + c_loss

                    metrics["fl/wk_actor_loss"] += actor_loss.item()
                    metrics["fl/wk_entropy"]    += entropy.mean().item()

                critic_loss_total = WORKER_VALUE * critic_loss_total
                critic_loss_total.backward()
                nn.utils.clip_grad_norm_(self.worker_critic.parameters(), MAX_GRAD_NORM)
                self.worker_critic_optim.step()

                metrics["fl/wk_critic_loss"] += critic_loss_total.item()
                n_updates += len(self.tls_ids)

        for t in self.tls_ids:
            self.worker_buffers[t].reset()

        if n_updates > 0:
            for k in metrics:
                metrics[k] /= n_updates
        return metrics

    def run(self):
        print(f"[Feudal] Starting Parallel Training ({self.num_envs} envs). W&B: {wandb.run.name}")

        def _make_env_fn():
            return make_env("cologne8_corridor", use_gui=False, max_steps=3600, scale=self.current_scale)

        multi_env = MultiAgentSharedSubproc([_make_env_fn for _ in range(self.num_envs)])

        # Initial reset — returns list of (obs_dict, info_dict) per env
        reset_results = multi_env.reset()
        obs_list = [r[0] for r in reset_results]    # list[dict[tls_id, np.ndarray]]

        ep_ext_rewards = [{t: 0.0 for t in self.tls_ids} for _ in range(self.num_envs)]

        while self.steps_collected < self.total_timesteps:

            # ── Curriculum ────────────────────────────────────────────────────
            target_scale = self._get_target_scale()
            if abs(target_scale - self.current_scale) > 0.01:
                print(f"[Feudal] Curriculum advanced! Scale {self.current_scale:.2f} → {target_scale:.2f}")
                self.current_scale = target_scale
                multi_env.set_scale(target_scale)

            # ── Manager Step (every MANAGER_C steps) — one decision per env ──
            is_manager_step = (self.steps_collected % MANAGER_C == 0)

            # We need per-env global states. Since envs are in subprocesses we
            # approximate with the obs from the last step (consistent with ZOH design).
            # Full global_state is returned inside the info dict on each step; we
            # cache it below.  On the very first step we use zeros (ZOH holds fine).
            for env_idx in range(self.num_envs):
                if is_manager_step:
                    # Use the cached global state for this env (set after first real step)
                    gs = getattr(self, "_cached_gs", [np.zeros(self.global_dim)] * self.num_envs)[env_idx]
                    gs_ts = torch.tensor(gs, dtype=torch.float32, device=self.device)
                    with torch.no_grad():
                        mgr_a, mgr_lp, _ = self.manager_actor.get_action_and_logprob(gs_ts)
                        mgr_v = self.manager_critic(gs_ts)
                    self.zoh_goals[env_idx] = mgr_a.cpu().numpy()
                    self._mgr_lp[env_idx]   = mgr_lp.item()
                    self._mgr_v[env_idx]    = mgr_v.item()

            # ── Build actions for all envs in one batched forward pass ────────
            # effective_goals[env_idx] = goals with 10% dropout
            effective_goals = self.zoh_goals.copy()
            dropout_mask = np.random.rand(self.num_envs) < GOAL_DROPOUT_P
            effective_goals[dropout_mask] = 0.0

            priority_weights = (effective_goals + 1.0) / 2.0  # [-1,1] → [0,1]

            # Batch worker forward pass across all envs for each junction
            # actions_list[env_idx] = {tls_id: action_int}
            actions_list = [{} for _ in range(self.num_envs)]
            w_lp_list    = [{} for _ in range(self.num_envs)]
            w_goal_list  = [{} for _ in range(self.num_envs)]

            for i, t in enumerate(self.tls_ids):
                # Stack obs from all envs for this junction: (num_envs, obs_dim)
                obs_batch = np.stack([obs_list[e][t] for e in range(self.num_envs)])
                goal_batch = effective_goals[:, i * K_GOAL_DIM:(i + 1) * K_GOAL_DIM]

                o_ts = torch.tensor(obs_batch, dtype=torch.float32, device=self.device)
                g_ts = torch.tensor(goal_batch, dtype=torch.float32, device=self.device)

                with torch.no_grad():
                    # worker actor handles (B, obs_dim) batches
                    a_batch, lp_batch, _ = self.worker_actors[t].get_action_and_logprob(o_ts, g_ts)

                for e in range(self.num_envs):
                    actions_list[e][t]   = int(a_batch[e].item()) if a_batch.dim() > 0 else int(a_batch.item())
                    w_lp_list[e][t]      = lp_batch[e].item() if lp_batch.dim() > 0 else lp_batch.item()
                    w_goal_list[e][t]    = goal_batch[e]

            # ── Step all envs in parallel ─────────────────────────────────────
            multi_env.step_async(actions_list)
            results = multi_env.step_wait()
            # results: list[(obs_dict, reward, terminated, truncated, info_dict)]

            self.steps_collected += self.num_envs

            # ── Process results from each env ─────────────────────────────────
            cached_gs = []
            for env_idx, res in enumerate(results):
                next_obs, _global_rew, terminated, truncated, info = res
                done = terminated or truncated

                # Real global state piggybacked from subprocess via info dict
                gs = info["global_state"]
                cached_gs.append(gs)

                worker_global_input = np.concatenate([gs, effective_goals[env_idx]])
                w_gs_ts = torch.tensor(worker_global_input, dtype=torch.float32, device=self.device)

                with torch.no_grad():
                    w_v_dict = self.worker_critic(w_gs_ts)

                # ── Manager reward accumulation & buffer write ────────────────
                global_ext_reward = info.get("reward", 0.0)
                clipped_mgr_r = float(np.clip(global_ext_reward / 100.0, -REWARD_CLIP, REWARD_CLIP))
                self._ep_mgr_reward[env_idx] += clipped_mgr_r

                if is_manager_step:
                    self.manager_buffer.add(
                        gs=gs,
                        a=self.zoh_goals[env_idx],
                        lp=self._mgr_lp[env_idx],
                        r=self._ep_mgr_reward[env_idx],
                        v=self._mgr_v[env_idx],
                        d=done
                    )
                    self._ep_mgr_reward[env_idx] = 0.0

                # ── Worker buffer write ───────────────────────────────────────
                for i, t in enumerate(self.tls_ids):
                    r_ext = info.get("per_junction", {}).get(t, {}).get("reward", 0.0)
                    pw    = priority_weights[env_idx, i * K_GOAL_DIM:(i + 1) * K_GOAL_DIM]
                    q_pen = float(info.get("per_junction", {}).get(t, {}).get("queue_length", 0.0)) / 20.0
                    w_pen = float(info.get("per_junction", {}).get(t, {}).get("wait_time", 0.0)) / 300.0
                    r_int = -(pw[0] * q_pen + pw[1] * w_pen)
                    r_w   = float(np.clip(ALPHA * r_ext + (1.0 - ALPHA) * r_int, -REWARD_CLIP, REWARD_CLIP))

                    ep_ext_rewards[env_idx][t] += r_ext

                    self.worker_buffers[t].add(
                        obs=obs_list[env_idx][t],
                        goal=w_goal_list[env_idx][t],
                        gs=worker_global_input,
                        a=actions_list[env_idx][t],
                        lp=w_lp_list[env_idx][t],
                        r=r_w,
                        v=w_v_dict[t].item(),
                        d=done
                    )

                # ── Episode end ───────────────────────────────────────────────
                if done:
                    wandb.log({
                        "env/global_reward":      sum(ep_ext_rewards[env_idx].values()),
                        "env/curriculum_scale":   self.current_scale,
                        "env/throughput":         info.get("throughput", 0),
                    }, step=self.steps_collected)
                    ep_ext_rewards[env_idx] = {t: 0.0 for t in self.tls_ids}
                    self._ep_mgr_reward[env_idx] = 0.0
                    # next_obs is the auto-reset new obs from the worker
                    obs_list[env_idx] = next_obs
                else:
                    obs_list[env_idx] = next_obs

            self._cached_gs = cached_gs

            # ── PPO Update when buffers are full ──────────────────────────────
            if self.worker_buffers[self.tls_ids[0]].is_full():
                m_metrics = self._update_manager()
                w_metrics = self._update_workers()

                wall_time = time.time() - self._wall_start
                dt        = time.time() - self._last_log_time
                d_steps   = self.steps_collected - self._last_log_steps
                fps       = int(d_steps / dt) if dt > 0 else 0
                self._last_log_time  = time.time()
                self._last_log_steps = self.steps_collected

                progress_pct = (self.steps_collected / self.total_timesteps) * 100.0

                wandb.log({
                    **m_metrics, **w_metrics,
                    "perf/wall_time_s": wall_time,
                    "perf/fps": fps,
                }, step=self.steps_collected)

                print(
                    f"[{self.steps_collected:08d} | {progress_pct:5.1f}% | FPS: {fps:5d}] "
                    f"M_Loss: {m_metrics['fl/mgr_actor_loss']:.3f} | "
                    f"W_Loss: {w_metrics['fl/wk_actor_loss']:.3f}"
                )

        multi_env.close()

        # Save models
        os.makedirs(MODELS_DIR, exist_ok=True)
        torch.save({
            "manager_actor":  self.manager_actor.state_dict(),
            "manager_critic": self.manager_critic.state_dict(),
            "worker_actors":  {t: a.state_dict() for t, a in self.worker_actors.items()},
        }, MODELS_DIR / f"{self.run_name}.pt")
        print("[Feudal] Training Complete. Model saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",     action="store_true")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--run-name",  type=str, default="c8-feudal-mappo")
    parser.add_argument("--num-envs",  type=int, default=4)
    parser.add_argument("--port",      type=int, default=8000)
    args = parser.parse_args()

    if args.train:
        wandb.init(
            project="marl-traffic",
            name=args.run_name,
            config={
                "gamma":       GAMMA,
                "gae_lambda":  GAE_LAMBDA,
                "clip_eps":    CLIP_EPS,
                "mgr_lr":      MANGER_LR,
                "wk_lr":       WORKER_LR,
                "num_envs":    args.num_envs,
                "architecture": "feudal_parallel",
            }
        )
        trainer = FeudalTrainer(
            total_timesteps=args.timesteps,
            run_name=args.run_name,
            num_envs=args.num_envs,
            port=args.port,
        )
        trainer.run()
        wandb.finish()
