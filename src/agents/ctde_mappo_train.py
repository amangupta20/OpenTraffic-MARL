"""
CTDE (Centralized Training, Decentralized Execution) training for Bangalore corridor.

Algorithm: MAPPO — Multi-Agent PPO with a shared centralized critic.
  - Training: Central Critic sees global state (all agents' obs concatenated).
  - Execution: Each actor uses only its local observation. Critic is discarded at eval.

Architecture:
  - N heterogeneous Actors  (one per junction, separate weights, local obs input)
  - 1 shared Central Critic (global state = zero-padded concat of all obs)

Curriculum: 0.75 → 1.5 → 2.0 (re-calibrated for 83%-reduced cleaned network)

Usage:
  docker compose run --rm -e MODE=ctde-train agent --timesteps 500000
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
from src.agents.mappo_networks import (
    MAPPOActor,
    MAPPOCritic,
    MAPPORolloutBuffer,
    build_global_state,
)
from src.utils.metrics import start_metrics_server

MODELS_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "models"

# ─────────────────────────────────────────────────────────────────────────────
# Hyper-parameters
# ─────────────────────────────────────────────────────────────────────────────
GAMMA        = 0.99
GAE_LAMBDA   = 0.95
CLIP_EPS     = 0.2
ENTROPY_COEF = 0.05
VALUE_COEF   = 0.5
LR_ACTOR     = 3e-4
LR_CRITIC    = 3e-4
N_STEPS      = 720        # Steps per rollout — sub-episode so rollouts see diverse traffic states
N_EPOCHS     = 10         # PPO update epochs per rollout
BATCH_SIZE   = 90         # keeps N_STEPS/BATCH_SIZE ratio at 8
MAX_GRAD_NORM = 0.5
REWARD_CLIP  = 500.0      # per-step reward clip — prevents exponential wait penalty from
                           # exploding to -millions on gridlock episodes at high scales
# Entropy annealing: start high (exploration) → end low (commit to good policy)
ENTROPY_COEF_START = 0.05
ENTROPY_COEF_END   = 0.003

# Curriculum: grades for the cleaned (83%-reduced) network
CURRICULUM = [
    (0.00, 0.75),   # Grade 1: 0–15%  of training
    (0.15, 1.50),   # Grade 2: 15–40%
    (0.40, 1.75),   # Grade 3: 40–70% — bridging step avoids cliff-edge jump to 2.0
    (0.70, 2.00),   # Grade 4: 70–100%
]


# ─────────────────────────────────────────────────────────────────────────────
# CTDE Trainer
# ─────────────────────────────────────────────────────────────────────────────

class CTDETrainer:
    """
    Manages the MAPPO training loop on the Bangalore corridor:
    - Runs rollouts in a single SUMO environment (libsumo, headless)
    - Computes GAE with the centralized critic for each agent
    - Updates actors + critic via PPO clipped surrogate loss
    - Manages curriculum scaling
    """

    def __init__(
        self,
        total_timesteps: int,
        run_name: str = "ctde-mappo",
        port: int = 8000,
    ) -> None:
        self.total_timesteps = total_timesteps
        self.run_name = run_name
        self.device = torch.device("cpu")

        start_metrics_server(port)

        # ── Discover network topology ──────────────────────────────────────
        print("[CTDE] Discovering network topology...")
        _probe = make_env("bangalore_corridor", use_gui=False, max_steps=1800, scale=0.75)
        self.tls_ids: list[str] = _probe.tls_ids
        self.obs_dims: dict[str, int] = {
            t: _probe.observation_space.spaces[t].shape[0] for t in self.tls_ids
        }
        self.act_dims: dict[str, int] = {
            t: int(_probe.action_space.spaces[t].n) for t in self.tls_ids
        }
        self.delta_time: int = _probe.delta_time
        _probe.close()

        self.max_obs_dim = max(self.obs_dims.values())
        self.global_dim  = self.max_obs_dim * len(self.tls_ids)

        print(f"[CTDE] {len(self.tls_ids)} junctions | global_dim={self.global_dim}")
        for t in self.tls_ids:
            print(f"  {t}: obs={self.obs_dims[t]}  actions={self.act_dims[t]}")

        # ── Build actors + shared critic (multi-head) ──────────────────────
        self.actors: dict[str, MAPPOActor] = {
            t: MAPPOActor(self.obs_dims[t], self.act_dims[t]).to(self.device)
            for t in self.tls_ids
        }
        # Multi-head critic: shared trunk + one V_i(s) head per junction
        # Prevents cross-agent reward-scale contamination
        self.critic = MAPPOCritic(self.global_dim, self.tls_ids).to(self.device)

        # Separate optimizers: one per actor + one for critic
        self.actor_optims: dict[str, torch.optim.Adam] = {
            t: torch.optim.Adam(self.actors[t].parameters(), lr=LR_ACTOR)
            for t in self.tls_ids
        }
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=LR_CRITIC)

        # ── Rollout buffers (one per agent) ───────────────────────────────
        self.buffers: dict[str, MAPPORolloutBuffer] = {
            t: MAPPORolloutBuffer(N_STEPS, self.obs_dims[t], self.global_dim)
            for t in self.tls_ids
        }

        # ── Curriculum + stats ─────────────────────────────────────────────
        self.current_scale = 0.75
        self.steps_collected = 0
        self._wall_start = time.time()

    # ──────────────────────────────────────────────────────────────────────────
    # Curriculum
    # ──────────────────────────────────────────────────────────────────────────

    def _get_target_scale(self) -> float:
        progress = self.steps_collected / self.total_timesteps
        target = CURRICULUM[0][1]
        for thresh, scale in CURRICULUM:
            if progress >= thresh:
                target = scale
        return target

    def _get_entropy_coef(self) -> float:
        """Linearly anneal entropy coefficient: high early (explore) → low late (exploit)."""
        progress = min(self.steps_collected / self.total_timesteps, 1.0)
        return ENTROPY_COEF_START + (ENTROPY_COEF_END - ENTROPY_COEF_START) * progress

    # ──────────────────────────────────────────────────────────────────────────
    # PPO update
    # ──────────────────────────────────────────────────────────────────────────

    def _update(self) -> dict[str, float]:
        """Run N_EPOCHS of PPO updates for all actors + critic on collected rollout."""

        # First compute last values for GAE bootstrap
        with torch.no_grad():
            # We don't have the very last obs here — GAE bootstrap = 0 for done envs
            # The env auto-resets, so we use V=0 as conservative bootstrap
            last_value_per_agent = {t: 0.0 for t in self.tls_ids}

        # Compute advantages per agent
        advantages_dict = {}
        returns_dict    = {}
        for t in self.tls_ids:
            adv, ret = self.buffers[t].compute_returns_and_advantages(
                last_value  = last_value_per_agent[t],
                gamma       = GAMMA,
                gae_lambda  = GAE_LAMBDA,
            )
            advantages_dict[t] = adv
            returns_dict[t]    = ret

        # Get all tensors
        tensors: dict[str, dict[str, torch.Tensor]] = {
            t: self.buffers[t].get_tensors(self.device) for t in self.tls_ids
        }

        metrics = {
            "ctde/actor_loss": 0.0,
            "ctde/critic_loss": 0.0,
            "ctde/entropy": 0.0,
            "ctde/approx_kl": 0.0,
        }
        n_updates = 0

        T = self.buffers[self.tls_ids[0]].ptr
        indices = np.arange(T)

        for _ in range(N_EPOCHS):
            np.random.shuffle(indices)
            for start in range(0, T - BATCH_SIZE + 1, BATCH_SIZE):
                batch_idx = torch.tensor(indices[start:start + BATCH_SIZE], device=self.device)

                # ── Critic update (multi-head: one loss per agent, summed) ──
                # All agents store identical global_state — pick any one
                gs = tensors[self.tls_ids[0]]["global_state"][batch_idx]
                all_v_preds = self.critic(gs)  # dict[agent_id → (batch,)]

                # Sum MSE losses across per-agent heads — each head fits its
                # own scale, trunk gradients are the average of all signals
                critic_loss = sum(
                    nn.functional.mse_loss(
                        all_v_preds[t],
                        torch.tensor(returns_dict[t], device=self.device)[batch_idx]
                    )
                    for t in self.tls_ids
                )

                self.critic_optim.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), MAX_GRAD_NORM)
                self.critic_optim.step()

                # ── Per-actor update ────────────────────────────────────
                for t in self.tls_ids:
                    batch = {k: v[batch_idx] for k, v in tensors[t].items()}
                    adv_batch = torch.tensor(advantages_dict[t], device=self.device)[batch_idx]

                    # Normalize advantages per-batch
                    adv_batch = (adv_batch - adv_batch.mean()) / (adv_batch.std() + 1e-8)

                    new_logprobs, entropy = self.actors[t].evaluate_actions(
                        batch["local_obs"], batch["actions"]
                    )
                    ratio = torch.exp(new_logprobs - batch["log_probs"])

                    # PPO clipped surrogate
                    surr1 = ratio * adv_batch
                    surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv_batch
                    actor_loss = -torch.min(surr1, surr2).mean()
                    entropy_coef = self._get_entropy_coef()  # annealed
                    entropy_loss = -entropy_coef * entropy.mean()
                    total_loss = actor_loss + entropy_loss

                    self.actor_optims[t].zero_grad()
                    total_loss.backward()
                    nn.utils.clip_grad_norm_(self.actors[t].parameters(), MAX_GRAD_NORM)
                    self.actor_optims[t].step()

                    # Approx KL (diagnostic)
                    with torch.no_grad():
                        approx_kl = ((ratio - 1) - torch.log(ratio)).mean().item()

                    metrics["ctde/actor_loss"]  += actor_loss.item()
                    metrics["ctde/entropy"]     += entropy.mean().item()
                    metrics["ctde/approx_kl"]   += approx_kl

                metrics["ctde/critic_loss"] += critic_loss.item()
                n_updates += 1

        if n_updates > 0:
            for k in metrics:
                metrics[k] /= n_updates

        # Reset all buffers
        for t in self.tls_ids:
            self.buffers[t].reset()

        return metrics

    # ──────────────────────────────────────────────────────────────────────────
    # Main training loop
    # ──────────────────────────────────────────────────────────────────────────

    def train(self) -> None:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        env = make_env(
            "bangalore_corridor", use_gui=False,
            max_steps=1800, scale=self.current_scale
        )
        obs_dict, _ = env.reset()

        rollout_rewards = {t: 0.0 for t in self.tls_ids}
        rollout_steps   = 0
        n_rollouts      = 0
        last_log_step   = 0

        print(f"[CTDE] Starting training for {self.total_timesteps} steps...")
        print(f"[CTDE] Curriculum: {[f'{s}×' for _, s in CURRICULUM]}")

        while self.steps_collected < self.total_timesteps:
            # ── Curriculum promotion ───────────────────────────────────────
            new_scale = self._get_target_scale()
            if abs(new_scale - self.current_scale) > 1e-6:
                self.current_scale = new_scale
                progress_pct = self.steps_collected / self.total_timesteps * 100
                print(f"\n{'='*55}")
                print(f"[CURRICULUM] Grade → scale={new_scale}  ({progress_pct:.1f}% complete)")
                print(f"{'='*55}")
                env.close()
                env = make_env(
                    "bangalore_corridor", use_gui=False,
                    max_steps=1800, scale=self.current_scale
                )
                obs_dict, _ = env.reset()

            # ── Rollout collection ─────────────────────────────────────────
            global_state = build_global_state(obs_dict, self.tls_ids, self.max_obs_dim)

            actions_dict: dict[str, int] = {}
            log_probs_dict: dict[str, float] = {}
            values_dict: dict[str, float] = {}

            with torch.no_grad():
                critic_input = torch.tensor(global_state, device=self.device).unsqueeze(0)
                # Multi-head: each agent gets its own V_i(s)
                all_values = self.critic(critic_input)

                for t in self.tls_ids:
                    obs_t = torch.tensor(obs_dict[t], device=self.device).unsqueeze(0)
                    action, log_prob, _ = self.actors[t].get_action_and_logprob(obs_t)
                    actions_dict[t]   = action.item()
                    log_probs_dict[t] = log_prob.item()
                    values_dict[t]    = all_values[t].item()  # agent-specific V_i(s)

            next_obs_dict, global_reward, terminated, _, info = env.step(actions_dict)
            self.steps_collected += self.delta_time
            rollout_steps += 1

            per_j = info.get("per_junction", {})
            done = terminated

            for t in self.tls_ids:
                local_reward = per_j.get(t, {}).get("reward", 0.0)
                # Clip to guard against exponential wait-penalty explosion on gridlock
                # episodes — without this, scale=2.0 gridlocks produce rewards of -12M
                # which corrupt the critic target and destroy the gradient signal.
                local_reward = max(local_reward, -REWARD_CLIP)
                self.buffers[t].add(
                    local_obs    = obs_dict[t],
                    global_state = global_state,
                    action       = actions_dict[t],
                    log_prob     = log_probs_dict[t],
                    reward       = local_reward,
                    value        = values_dict[t],
                    done         = done,
                )
                rollout_rewards[t] += local_reward

            obs_dict = next_obs_dict

            if done:
                obs_dict, _ = env.reset()

            # ── PPO update when buffer full ────────────────────────────────
            if self.buffers[self.tls_ids[0]].is_full():
                update_metrics = self._update()
                n_rollouts += 1

                elapsed = time.time() - self._wall_start
                fps = self.steps_collected / elapsed if elapsed > 0 else 0
                progress_pct = self.steps_collected / self.total_timesteps * 100

                avg_reward = sum(rollout_rewards.values()) / len(self.tls_ids)
                queue      = info.get("queue_length", 0)
                wait       = info.get("wait_time_total", 0)

                print(
                    f"[{self.steps_collected:>7d}/{self.total_timesteps}] "
                    f"scale={self.current_scale:.2f}  "
                    f"fps={fps:.0f}  "
                    f"queue={queue:.0f}  wait={wait:.0f}  "
                    f"reward={avg_reward:.1f}  "
                    f"kl={update_metrics['ctde/approx_kl']:.4f}  "
                    f"({progress_pct:.1f}%)"
                )

                wandb.log({
                    **update_metrics,
                    "env/queue_length_avg": queue,
                    "env/wait_time_avg":    wait,
                    "env/reward_avg":       avg_reward,
                    "env/scale":            self.current_scale,
                    "perf/fps":             fps,
                    "train/progress_pct":   progress_pct,
                    "train/entropy_coef":   self._get_entropy_coef(),
                })

                rollout_rewards = {t: 0.0 for t in self.tls_ids}
                rollout_steps   = 0

        env.close()
        print(f"\n[CTDE] Training complete. Saving models...")

        # ── Save actor weights + critic ────────────────────────────────────
        saved = []
        for t in self.tls_ids:
            path = str(MODELS_DIR / f"ctde_actor_{t}.pt")
            torch.save(self.actors[t].state_dict(), path)
            saved.append(path)
            print(f"  Saved actor: {path}")

        critic_path = str(MODELS_DIR / "ctde_critic.pt")
        torch.save(self.critic.state_dict(), critic_path)
        print(f"  Saved critic: {critic_path}")

        # Save topology metadata for eval loading
        meta = {
            "tls_ids":     self.tls_ids,
            "obs_dims":    self.obs_dims,
            "act_dims":    self.act_dims,
            "max_obs_dim": self.max_obs_dim,
            "global_dim":  self.global_dim,
            "run_id":      wandb.run.id,
            "run_name":    wandb.run.name,
        }
        meta_path = MODELS_DIR / "ctde_run_metadata.json"
        meta_path.write_text(json.dumps(meta, indent=2))
        print(f"  Saved metadata: {meta_path}")

        # ── Upload to W&B as versioned artifact ───────────────────────────
        artifact = wandb.Artifact(
            name=f"ctde-mappo-models-{wandb.run.name}",
            type="model",
            description=f"CTDE MAPPO actor+critic weights — {len(self.tls_ids)} junctions",
            metadata=meta,
        )
        for p in saved:
            artifact.add_file(p)
        artifact.add_file(critic_path)
        wandb.log_artifact(artifact)
        print(f"[wandb] Artifact logged.")

        wandb.finish()

    # ──────────────────────────────────────────────────────────────────────────
    # Decentralized evaluation helper (used by ctde_mappo_eval)
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def load_actors(
        models_dir: pathlib.Path,
    ) -> tuple[dict[str, MAPPOActor], dict]:
        """
        Load all actor weights from `models_dir`.
        Returns (actors_dict, metadata_dict).
        """
        meta_path = models_dir / "ctde_run_metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"CTDE metadata not found at {meta_path}.\n"
                "  Run: make ctde-train ARGS='--timesteps 500000'"
            )
        meta = json.loads(meta_path.read_text())
        tls_ids   = meta["tls_ids"]
        obs_dims  = meta["obs_dims"]
        act_dims  = meta["act_dims"]

        actors: dict[str, MAPPOActor] = {}
        for t in tls_ids:
            path = models_dir / f"ctde_actor_{t}.pt"
            if not path.exists():
                raise FileNotFoundError(f"Actor weights not found: {path}")
            actor = MAPPOActor(obs_dims[t], act_dims[t])
            actor.load_state_dict(torch.load(str(path), map_location="cpu"))
            actor.eval()
            actors[t] = actor
        return actors, meta


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="CTDE MAPPO trainer — Bangalore corridor")
    parser.add_argument("--train",      action="store_true")
    parser.add_argument("--timesteps",  type=int, default=500_000)
    parser.add_argument("--run-name",   type=str, default="ctde-mappo")
    parser.add_argument("--port",       type=int, default=8000)
    args = parser.parse_args()

    if not args.train:
        parser.error("Specify --train to begin training.")

    # W&B init
    wandb_key  = os.environ.get("WANDB_API_KEY")
    has_netrc  = pathlib.Path("~/.netrc").expanduser().exists()
    is_sanity  = args.timesteps < 2000 or args.run_name.lower().startswith("sanity")
    mode       = "disabled" if is_sanity else ("online" if (wandb_key or has_netrc) else "offline")

    print(f"[wandb] mode={mode}  sanity={is_sanity}")
    wandb.init(
        project  = "marl-traffic",
        name     = args.run_name,
        mode     = mode,
        config   = {
            "algorithm":        "MAPPO-CTDE",
            "total_timesteps":  args.timesteps,
            "curriculum":       True,
            "scale_stages":     [s for _, s in CURRICULUM],
            "gamma":            GAMMA,
            "gae_lambda":       GAE_LAMBDA,
            "clip_eps":         CLIP_EPS,
            "lr_actor":         LR_ACTOR,
            "lr_critic":        LR_CRITIC,
            "n_steps":          N_STEPS,
            "n_epochs":         N_EPOCHS,
            "batch_size":       BATCH_SIZE,
            "entropy_coef":     ENTROPY_COEF,
        },
    )

    trainer = CTDETrainer(
        total_timesteps = args.timesteps,
        run_name        = args.run_name,
        port            = args.port,
    )
    trainer.train()


if __name__ == "__main__":
    main()
