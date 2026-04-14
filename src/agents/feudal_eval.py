"""
Feudal MARL Evaluation & 4-Way Comparison — Cologne8 RESCO.

Modes:
  --evaluate : Headless eval of Feudal workers (actors only, no critic)
  --compare  : 4-way comparison: Static Timer vs Ind. PPO vs CTDE MAPPO vs Feudal MARL
               Logs results and plot to W&B (Feudal training run).

Usage (via Docker):
  docker compose run --rm -e MODE=feudal-eval   agent --evaluate
  docker compose run --rm -e MODE=feudal-eval   agent --compare --model feudal-exp-v2
"""
from __future__ import annotations

import argparse
import pathlib
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from stable_baselines3 import PPO

from src.envs import make_env
from src.agents.mappo_networks import MAPPOActor
from src.agents.feudal_networks import FeudalWorkerActor, ManagerActorContinuous
from src.utils.metrics import start_metrics_server

# Reuse helpers from CTDE eval (static timer, collection loops, PPO step fn)
from src.agents.ctde_mappo_eval import (
    _collect_episode,
    _collect_multi_episode_with_reset,
    _make_static_step_fn_resettable,
    _make_indep_ppo_step_fn,
    _make_ctde_step_fn,
)
from src.agents.ctde_mappo_train import CTDETrainer

MODELS_DIR  = pathlib.Path(__file__).resolve().parent.parent.parent / "models"
RESULTS_DIR = MODELS_DIR.parent / "results"

EVAL_SCALE      = 1.3
MAX_STEPS       = 3600
DELTA_TIME      = 15
GREEN_DUR       = 30
N_EVAL_EPISODES = 10


# ─────────────────────────────────────────────────────────────────────────────
# Feudal Hierarchy Constants (must match training)
# ─────────────────────────────────────────────────────────────────────────────
MANAGER_C  = 4    # Manager decides every 4 env steps (60s)
K_GOAL_DIM = 3    # Priority vector dimensionality


# ─────────────────────────────────────────────────────────────────────────────
# Load full Feudal hierarchy from .pt checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def _load_feudal_hierarchy(
    model_name: str,
) -> tuple[ManagerActorContinuous, dict[str, FeudalWorkerActor], list[str], dict[str, int]]:
    """Load Manager actor + Worker actors from models/<model_name>.pt"""
    ckpt_path = MODELS_DIR / f"{model_name}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Feudal model not found: {ckpt_path}\n"
            f"  Run: make feudal-train ARGS='--run-name {model_name} --timesteps 500000'"
        )

    # Probe env to discover topology
    _probe = make_env("cologne8_corridor", use_gui=False, max_steps=MAX_STEPS, scale=EVAL_SCALE)
    tls_ids  = _probe.tls_ids
    obs_dims = {t: _probe.observation_space.spaces[t].shape[0] for t in tls_ids}
    act_dims = {t: int(_probe.action_space.spaces[t].n) for t in tls_ids}
    global_dim = _probe.get_global_state().shape[0]
    _probe.close()

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    n_agents = len(tls_ids)

    # Manager
    manager = ManagerActorContinuous(global_dim, n_agents, K_GOAL_DIM)
    manager.load_state_dict(ckpt["manager_actor"])
    manager.eval()

    # Workers
    workers: dict[str, FeudalWorkerActor] = {}
    for t in tls_ids:
        actor = FeudalWorkerActor(obs_dims[t], K_GOAL_DIM, act_dims[t])
        actor.load_state_dict(ckpt["worker_actors"][t])
        actor.eval()
        workers[t] = actor

    print(f"[Feudal Eval] Loaded Manager + {len(workers)} Workers from {ckpt_path.name}")
    return manager, workers, tls_ids, act_dims


# ─────────────────────────────────────────────────────────────────────────────
# Feudal step function — full Manager→Worker hierarchy, live
# ─────────────────────────────────────────────────────────────────────────────

def _make_feudal_step_fn(
    manager: ManagerActorContinuous,
    workers: dict[str, FeudalWorkerActor],
    env,
) -> tuple[callable, callable]:
    """
    Returns (step_fn, reset_fn) that runs the full Feudal hierarchy live.

    - Manager runs deterministically every MANAGER_C steps using env.get_global_state().
    - Goals are held via Zero-Order Hold between Manager decisions.
    - Workers receive their per-junction goal slice and act deterministically (argmax).
    - reset_fn resets the internal step counter and cached goals for multi-episode eval.
    """
    tls_list = list(workers.keys())
    n_agents = len(tls_list)

    # Mutable state for the closure
    step_count   = [0]
    cached_goals = [np.zeros(n_agents * K_GOAL_DIM, dtype=np.float32)]

    def step_fn(obs_dict: dict[str, np.ndarray]) -> dict[str, int]:
        # ── Manager decision every MANAGER_C steps ────────────────────────
        if step_count[0] % MANAGER_C == 0:
            gs = env.get_global_state()
            gs_ts = torch.tensor(gs, dtype=torch.float32).unsqueeze(0)  # (1, global_dim)
            with torch.no_grad():
                # Deterministic: use mean (forward = tanh-bounded mu), not sampled
                goals = manager(gs_ts).squeeze(0).cpu().numpy()  # (n_agents * K_GOAL_DIM,)
            cached_goals[0] = goals

        goals = cached_goals[0]
        step_count[0] += 1

        # ── Worker actions (deterministic argmax) ─────────────────────────
        actions = {}
        for i, t in enumerate(tls_list):
            g_i = goals[i * K_GOAL_DIM : (i + 1) * K_GOAL_DIM]
            obs_t  = torch.tensor(obs_dict[t], dtype=torch.float32).unsqueeze(0)
            goal_t = torch.tensor(g_i, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits = workers[t](obs_t, goal_t)
                action = torch.argmax(logits, dim=-1).item()
            actions[t] = action
        return actions

    def reset_fn():
        """Reset between episodes for multi-episode eval."""
        step_count[0] = 0
        cached_goals[0] = np.zeros(n_agents * K_GOAL_DIM, dtype=np.float32)

    return step_fn, reset_fn


# ─────────────────────────────────────────────────────────────────────────────
# Headless eval
# ─────────────────────────────────────────────────────────────────────────────

def run_feudal_eval(model_name: str, port: int = 8000) -> None:
    start_metrics_server(port)
    manager, workers, tls_ids, _ = _load_feudal_hierarchy(model_name)
    env = make_env("cologne8_corridor", use_gui=False, max_steps=MAX_STEPS, scale=EVAL_SCALE)

    step_fn, reset_fn = _make_feudal_step_fn(manager, workers, env)
    print(f"[Feudal Eval] Running {N_EVAL_EPISODES} episodes (scale={EVAL_SCALE})...")
    results = _collect_multi_episode_with_reset(env, step_fn, N_EVAL_EPISODES, reset_fn)
    env.close()

    print(f"\n[Feudal Eval] Results ({results['n_episodes']} episodes):")
    print(f"  avg_queue    = {results['avg_queue']:.2f} ± {results['avg_queue_std']:.2f}")
    print(f"  avg_wait     = {results['avg_wait']:.2f} ± {results['avg_wait_std']:.2f}")
    print(f"  total_reward = {results['total_reward']:.2f} ± {results['total_reward_std']:.2f}")
    print(f"  throughput   = {results['throughput']:.0f} ± {results['throughput_std']:.0f}")


# ─────────────────────────────────────────────────────────────────────────────
# 4-way comparison
# ─────────────────────────────────────────────────────────────────────────────

def run_feudal_compare(model_name: str, port: int = 8000) -> None:
    """
    4-way comparison: Static Timer → Ind. PPO → CTDE MAPPO → Feudal MARL.
    Logs results + plot to W&B.
    """
    start_metrics_server(port)

    # Load all models
    manager, feudal_workers, tls_ids, act_dims = _load_feudal_hierarchy(model_name)

    print("[Feudal Compare] Loading CTDE actors...")
    ctde_actors, ctde_meta = CTDETrainer.load_actors(MODELS_DIR)

    # W&B — start a new compare run linked to the feudal model name
    has_creds = (
        pathlib.Path("~/.netrc").expanduser().exists()
        or bool(__import__("os").environ.get("WANDB_API_KEY"))
    )
    wandb.init(
        project="marl-traffic",
        name=f"{model_name}-compare",
        mode="online" if has_creds else "offline",
        config={"model": model_name, "eval_scale": EVAL_SCALE, "n_episodes": N_EVAL_EPISODES},
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Probe env for phase counts (needed by static timer)
    n_phases = {t: act_dims[t] for t in tls_ids}

    print(f"\n[Feudal Compare] Config: scale={EVAL_SCALE}×, episodes={N_EVAL_EPISODES}")
    print("=" * 70)

    static_step_fn, static_reset_fn = _make_static_step_fn_resettable(tls_ids, n_phases)

    # NOTE: Feudal step_fn needs a live env reference for get_global_state().
    #       We create a dedicated env for it below in the run loop.
    #       Other systems use their own fresh env per run.
    systems_simple = [
        ("Static Timer",    static_step_fn,                   static_reset_fn),
        ("Independent PPO", _make_indep_ppo_step_fn(tls_ids), None),
        ("CTDE MAPPO",      _make_ctde_step_fn(ctde_actors),  None),
    ]
    colors = {
        "Static Timer":    "#e74c3c",
        "Independent PPO": "#f39c12",
        "CTDE MAPPO":      "#2ecc71",
        "Feudal MARL":     "#9b59b6",
    }

    # Run simple systems (no env dependency in step_fn)
    all_results: dict[str, dict] = {}
    for name, step_fn, reset_fn in systems_simple:
        print(f"\n[Feudal Compare] → {name} ({N_EVAL_EPISODES} episodes)...")
        env = make_env("cologne8_corridor", use_gui=False, max_steps=MAX_STEPS, scale=EVAL_SCALE)
        all_results[name] = _collect_multi_episode_with_reset(env, step_fn, N_EVAL_EPISODES, reset_fn)
        env.close()
        r = all_results[name]
        print(f"  queue={r['avg_queue']:6.1f}±{r['avg_queue_std']:.1f}  "
              f"wait={r['avg_wait']:6.1f}±{r['avg_wait_std']:.1f}  "
              f"reward={r['total_reward']:8.1f}±{r['total_reward_std']:.1f}  "
              f"tp={r['throughput']:5.0f}±{r['throughput_std']:.0f}")

    # Run Feudal — step_fn is bound to a live env for get_global_state()
    print(f"\n[Feudal Compare] → Feudal MARL ({N_EVAL_EPISODES} episodes)...")
    feudal_env = make_env("cologne8_corridor", use_gui=False, max_steps=MAX_STEPS, scale=EVAL_SCALE)
    feudal_step_fn, feudal_reset_fn = _make_feudal_step_fn(manager, feudal_workers, feudal_env)
    all_results["Feudal MARL"] = _collect_multi_episode_with_reset(
        feudal_env, feudal_step_fn, N_EVAL_EPISODES, feudal_reset_fn
    )
    feudal_env.close()
    r = all_results["Feudal MARL"]
    print(f"  queue={r['avg_queue']:6.1f}±{r['avg_queue_std']:.1f}  "
          f"wait={r['avg_wait']:6.1f}±{r['avg_wait_std']:.1f}  "
          f"reward={r['total_reward']:8.1f}±{r['total_reward_std']:.1f}  "
          f"tp={r['throughput']:5.0f}±{r['throughput_std']:.0f}")

    names = [n for n, _, _ in systems_simple] + ["Feudal MARL"]

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    fig.patch.set_facecolor("#0f1117")
    for ax in axes.flat:
        ax.set_facecolor("#1a1d27")
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(colors="#aaaaaa")
        ax.xaxis.label.set_color("#aaaaaa")
        ax.yaxis.label.set_color("#aaaaaa")
        ax.title.set_color("#ffffff")

    fig.suptitle(
        f"4-Way Comparison — Cologne8 RESCO  (scale={EVAL_SCALE}×, {N_EVAL_EPISODES} eps)",
        fontsize=14, fontweight="bold", color="#ffffff"
    )

    step_axis = np.arange(len(all_results["Static Timer"]["mean_queues"])) * DELTA_TIME

    for name in names:
        res = all_results[name]
        c = colors[name]
        n = min(len(res["mean_queues"]), len(step_axis))
        axes[0, 0].plot(step_axis[:n], res["mean_queues"][:n],              label=name, color=c, alpha=0.9)
        axes[0, 1].plot(step_axis[:n], res["mean_waits"][:n],               label=name, color=c, alpha=0.9)
        axes[1, 0].plot(step_axis[:n], np.cumsum(res["mean_rewards"][:n]),  label=name, color=c, alpha=0.9)

    # Throughput bar chart
    tps         = [all_results[n]["throughput"] for n in names]
    bar_colors  = [colors[n] for n in names]
    bars = axes[1, 1].bar(names, tps, color=bar_colors, alpha=0.85, edgecolor="#0f1117", width=0.5)
    axes[1, 1].set_ylabel("Total vehicles arrived", color="#aaaaaa")
    axes[1, 1].set_title("Throughput")
    axes[1, 1].tick_params(axis="x", rotation=15)

    # ── Summary text box ──────────────────────────────────────────────────────
    ref = all_results["Static Timer"]
    def pct(val, ref_val, higher=False):
        if abs(ref_val) < 1e-8: return 0.0
        return ((ref_val - val) / abs(ref_val) * 100) if not higher else ((val - ref_val) / abs(ref_val) * 100)

    lines = ["Δ vs Static Timer (queue reduction %)"]
    for nm in ["Independent PPO", "CTDE MAPPO", "Feudal MARL"]:
        q_imp = pct(all_results[nm]["avg_queue"], ref["avg_queue"])
        w_imp = pct(all_results[nm]["avg_wait"],  ref["avg_wait"])
        lines.append(f"  {nm:<16}: {q_imp:+.1f}% Q | {w_imp:+.1f}% W")

    # Feudal vs CTDE
    feudal_vs_ctde_q = pct(all_results["Feudal MARL"]["avg_queue"], all_results["CTDE MAPPO"]["avg_queue"])
    feudal_vs_ctde_w = pct(all_results["Feudal MARL"]["avg_wait"],  all_results["CTDE MAPPO"]["avg_wait"])
    lines.append(f"\nFeudalMARL vs CTDE: {feudal_vs_ctde_q:+.1f}% Q | {feudal_vs_ctde_w:+.1f}% W")

    textstr = "\n".join(lines)
    props = dict(boxstyle="round", facecolor="#1a1d27", alpha=0.9, edgecolor="#9b59b6")
    axes[1, 1].text(0.02, 0.96, textstr, transform=axes[1, 1].transAxes, fontsize=9,
                    verticalalignment="top", bbox=props, color="#ffffff", zorder=5)
    axes[1, 1].set_ylim(0, max(tps) * 2.2)

    axes[0, 0].set(title="Queue Length (Mean)", ylabel="Vehicles waiting",     xlabel="Simulation time (s)")
    axes[0, 1].set(title="Total Wait Time (Mean)", ylabel="Cumulative wait (s)", xlabel="Simulation time (s)")
    axes[1, 0].set(title="Cumulative Reward (Mean)", ylabel="Reward",            xlabel="Simulation time (s)")

    for ax in axes.flat[:3]:
        ax.legend(fontsize=8, facecolor="#2a2d37", labelcolor="#dddddd")
        ax.grid(alpha=0.15, color="#444444")

    plt.tight_layout()
    plot_path = RESULTS_DIR / f"feudal_4way_comparison_{model_name}.png"
    plt.savefig(str(plot_path), dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n[Feudal Compare] Plot saved → {plot_path}")

    # ── W&B logging ───────────────────────────────────────────────────────────
    summary: dict[str, float] = {"compare/n_eval_episodes": N_EVAL_EPISODES}
    for name in names:
        r = all_results[name]
        key = name.replace(" ", "_")
        summary[f"compare/{key}/avg_queue"]        = r["avg_queue"]
        summary[f"compare/{key}/avg_queue_std"]    = r["avg_queue_std"]
        summary[f"compare/{key}/avg_wait"]         = r["avg_wait"]
        summary[f"compare/{key}/avg_wait_std"]     = r["avg_wait_std"]
        summary[f"compare/{key}/total_reward"]     = r["total_reward"]
        summary[f"compare/{key}/total_reward_std"] = r["total_reward_std"]
        summary[f"compare/{key}/throughput"]       = r["throughput"]

    # Improvement metrics (all vs static)
    for nm in ["Independent PPO", "CTDE MAPPO", "Feudal MARL"]:
        r = all_results[nm]
        key = nm.replace(" ", "_")
        summary[f"compare/{key}/queue_reduction_pct"] = pct(r["avg_queue"], ref["avg_queue"])
        summary[f"compare/{key}/wait_reduction_pct"]  = pct(r["avg_wait"],  ref["avg_wait"])

    summary["compare/Feudal_vs_CTDE_queue_pct"] = feudal_vs_ctde_q
    summary["compare/Feudal_vs_CTDE_wait_pct"]  = feudal_vs_ctde_w

    wandb.log(summary)
    wandb.log({"compare/4way_plot": wandb.Image(str(plot_path))})

    # ── Print results table ────────────────────────────────────────────────────
    print(f"\n{'='*78}")
    print(f"  {N_EVAL_EPISODES}-Episode Results  (scale={EVAL_SCALE}×)")
    print(f"{'='*78}")
    print(f"{'System':<20} {'Avg Queue':>14} {'Avg Wait':>14} {'Reward':>16} {'Throughput':>12}")
    print(f"{'-'*78}")
    for name in names:
        r = all_results[name]
        print(f"{name:<20} "
              f"{r['avg_queue']:>6.1f}±{r['avg_queue_std']:<5.1f} "
              f"{r['avg_wait']:>6.1f}±{r['avg_wait_std']:<5.1f} "
              f"{r['total_reward']:>8.1f}±{r['total_reward_std']:<5.1f} "
              f"{r['throughput']:>6.0f}±{r['throughput_std']:<5.0f}")
    print(f"{'='*78}")
    print(f"\nΔ vs Static Timer:")
    for nm in ["Independent PPO", "CTDE MAPPO", "Feudal MARL"]:
        qi = summary[f"compare/{nm.replace(' ','_')}/queue_reduction_pct"]
        wi = summary[f"compare/{nm.replace(' ','_')}/wait_reduction_pct"]
        print(f"  {nm:<18}: {qi:+.1f}% queue  {wi:+.1f}% wait")
    print(f"\nFeudalMARL vs CTDE: {feudal_vs_ctde_q:+.1f}% queue  {feudal_vs_ctde_w:+.1f}% wait")

    wandb.finish()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Feudal MARL evaluation — Cologne8 corridor")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--evaluate", action="store_true", help="Headless eval of Feudal workers")
    group.add_argument("--compare",  action="store_true", help="4-way comparison: Static / IndePPO / CTDE / Feudal")
    parser.add_argument("--model",   type=str, default="feudal-exp-v2", help="Model filename (without .pt)")
    parser.add_argument("--port",    type=int, default=8000)
    args = parser.parse_args()

    if args.evaluate:
        run_feudal_eval(args.model, args.port)
    elif args.compare:
        run_feudal_compare(args.model, args.port)


if __name__ == "__main__":
    main()
