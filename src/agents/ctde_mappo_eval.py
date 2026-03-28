"""
CTDE MAPPO Evaluation, Demo & 3-Way Comparison — Bangalore MG Road.

Modes:
  --evaluate : Decentralized eval (actors only, no critic)
  --demo     : Visual sumo-gui run via noVNC
  --compare  : 3-way comparison: Static Timer vs Independent PPO vs CTDE MAPPO
               Logs results to the CTDE training W&B run.

Usage (via Docker):
  docker compose run --rm -e MODE=ctde-eval    agent
  docker compose run --rm -e MODE=ctde-compare agent
  docker compose run --rm -p 6080:6080 -e MODE=ctde-demo agent
"""
from __future__ import annotations

import argparse
import json
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
from src.agents.mappo_networks import MAPPOActor, build_global_state
from src.agents.ctde_mappo_train import CTDETrainer
from src.utils.metrics import start_metrics_server

MODELS_DIR  = pathlib.Path(__file__).resolve().parent.parent.parent / "models"
RESULTS_DIR = MODELS_DIR.parent / "results"

# Eval configuration — match end-of-training stress level
EVAL_SCALE  = 2.0
MAX_STEPS   = 3600
DELTA_TIME  = 5
GREEN_DUR   = 30   # static-timer green phase duration (seconds)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _collect_episode(
    env,
    step_fn,       # callable(obs_dict) → actions_dict
    max_steps: int = MAX_STEPS,
) -> dict[str, Any]:
    """Run one full episode under step_fn action policy. Returns summary metrics."""
    obs_dict, _ = env.reset()
    total_queue = 0.0
    total_wait  = 0.0
    total_reward = 0.0
    total_throughput = 0
    n_steps = 0

    while True:
        actions = step_fn(obs_dict)
        obs_dict, reward, terminated, _, info = env.step(actions)
        total_queue      += info.get("queue_length", 0)
        total_wait       += info.get("wait_time_total", 0)
        total_reward     += reward
        total_throughput += info.get("throughput", 0)
        n_steps += 1
        if terminated:
            break

    return {
        "avg_queue":     total_queue / max(n_steps, 1),
        "avg_wait":      total_wait / max(n_steps, 1),
        "total_reward":  total_reward,
        "throughput":    total_throughput,
        "n_steps":       n_steps,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Static timer baseline
# ─────────────────────────────────────────────────────────────────────────────

def _make_static_step_fn(tls_ids: list[str], n_phases: dict[str, int]):
    """Returns a step function that cycles through ALL phases every GREEN_DUR seconds.

    n_phases: dict mapping tls_id → number of green phases (from action_space.n)
    Previously this was hard-coded to binary (% 2), which left 4-6 phase junctions
    permanently stuck cycling only their first two phases.
    """
    phase_idx: dict[str, int] = {t: 0 for t in tls_ids}
    green_time: dict[str, int] = {t: 0 for t in tls_ids}

    def step_fn(obs_dict: dict[str, np.ndarray]) -> dict[str, int]:
        actions = {}
        for t in tls_ids:
            green_time[t] += DELTA_TIME
            if green_time[t] >= GREEN_DUR:
                phase_idx[t] = (phase_idx[t] + 1) % n_phases[t]  # cycle ALL phases
                green_time[t] = 0
            actions[t] = phase_idx[t]
        return actions

    return step_fn


# ─────────────────────────────────────────────────────────────────────────────
# Independent PPO step function
# ─────────────────────────────────────────────────────────────────────────────

def _make_indep_ppo_step_fn(tls_ids: list[str]) -> callable:
    """Load existing parallel PPO weights and return a step function."""
    models: dict[str, PPO] = {}
    missing = []
    for t in tls_ids:
        path = MODELS_DIR / f"ppo_blr_parallel_{t}.zip"
        if path.exists():
            models[t] = PPO.load(str(path), device="cpu")
        else:
            missing.append(str(path))

    if missing:
        raise FileNotFoundError(
            "Independent PPO models not found:\n" + "\n".join(f"  {p}" for p in missing) +
            "\n  Run: make blr-train ARGS='--timesteps 500000'"
        )

    def step_fn(obs_dict: dict[str, np.ndarray]) -> dict[str, int]:
        return {
            t: int(models[t].predict(obs_dict[t], deterministic=True)[0])
            for t in tls_ids
        }

    return step_fn


# ─────────────────────────────────────────────────────────────────────────────
# CTDE step function (decentralized — actors only)
# ─────────────────────────────────────────────────────────────────────────────

def _make_ctde_step_fn(actors: dict[str, MAPPOActor]) -> callable:
    """Return deterministic step function using trained CTDE actors (no critic)."""

    def step_fn(obs_dict: dict[str, np.ndarray]) -> dict[str, int]:
        actions = {}
        for t, actor in actors.items():
            obs_t = torch.tensor(obs_dict[t], dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits = actor(obs_t)
                action = torch.argmax(logits, dim=-1).item()
            actions[t] = action
        return actions

    return step_fn


# ─────────────────────────────────────────────────────────────────────────────
# Evaluate (headless)
# ─────────────────────────────────────────────────────────────────────────────

def run_ctde_eval(port: int = 8000) -> None:
    print("[CTDE Eval] Loading trained actors...")
    actors, meta = CTDETrainer.load_actors(MODELS_DIR)
    tls_ids = meta["tls_ids"]

    start_metrics_server(port)
    env = make_env("bangalore_corridor", use_gui=False, max_steps=MAX_STEPS, scale=EVAL_SCALE)

    step_fn = _make_ctde_step_fn(actors)
    results = _collect_episode(env, step_fn)
    env.close()

    print("\n[CTDE Eval] Results:")
    print(f"  avg_queue   = {results['avg_queue']:.2f}")
    print(f"  avg_wait    = {results['avg_wait']:.2f}")
    print(f"  total_reward= {results['total_reward']:.2f}")
    print(f"  throughput  = {results['throughput']}")


# ─────────────────────────────────────────────────────────────────────────────
# Visual demo
# ─────────────────────────────────────────────────────────────────────────────

def run_ctde_demo(port: int = 8000) -> None:
    print("[CTDE Demo] Loading trained actors for GUI demo...")
    actors, meta = CTDETrainer.load_actors(MODELS_DIR)
    tls_ids = meta["tls_ids"]

    start_metrics_server(port)
    env = make_env(
        "bangalore_corridor", use_gui=True,
        max_steps=MAX_STEPS, scale=EVAL_SCALE, gui_delay=150
    )

    obs_dict, _ = env.reset()
    step_fn = _make_ctde_step_fn(actors)

    step = 0
    while True:
        actions = step_fn(obs_dict)
        obs_dict, _, terminated, _, info = env.step(actions)
        step += 1
        if step % 60 == 0:
            q = info.get("queue_length", 0)
            w = info.get("wait_time_total", 0)
            print(f"[step {step}] queue={q:.0f}  wait={w:.0f}")
        if terminated:
            print("[CTDE Demo] Episode ended. Resetting...")
            obs_dict, _ = env.reset()
            step = 0

    env.close()


# ─────────────────────────────────────────────────────────────────────────────
# 3-Way comparison
# ─────────────────────────────────────────────────────────────────────────────

def run_ctde_compare(port: int = 8000) -> None:
    """
    Run Static Timer → Independent PPO → CTDE MAPPO on identical conditions.
    Generate 3-way comparison plot and log summary metrics to W&B (CTDE run).
    """
    print("[CTDE Compare] Loading CTDE actors...")
    actors, ctde_meta = CTDETrainer.load_actors(MODELS_DIR)
    tls_ids = ctde_meta["tls_ids"]

    start_metrics_server(port)

    # ── Resume CTDE W&B run ─────────────────────────────────────────────────
    meta_path = MODELS_DIR / "ctde_run_metadata.json"
    wandb_key  = pathlib.Path("~/.netrc").expanduser().exists() or bool(
        __import__("os").environ.get("WANDB_API_KEY")
    )
    run_id  = ctde_meta.get("run_id")
    run_name = ctde_meta.get("run_name", "ctde-mappo")

    if wandb_key and run_id:
        print(f"[wandb] Resuming run {run_id}...")
        wandb.init(project="marl-traffic", id=run_id, resume="allow",
                   name=f"{run_name}-compare")
    else:
        wandb.init(project="marl-traffic", name=f"{run_name}-compare",
                   mode="offline" if not wandb_key else "online")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Helper: run a full episode and collect per-step data for plotting ──
    def _run_episode_with_trace(env, step_fn):
        obs_dict, _ = env.reset()
        queues, waits, rewards = [], [], []
        throughput_total = 0
        while True:
            actions = step_fn(obs_dict)
            obs_dict, reward, terminated, _, info = env.step(actions)
            queues.append(info.get("queue_length", 0))
            waits.append(info.get("wait_time_total", 0))
            rewards.append(reward)
            throughput_total += info.get("throughput", 0)
            if terminated:
                break
        return {
            "queues": queues, "waits": waits, "rewards": rewards,
            "avg_queue": float(np.mean(queues)),
            "avg_wait":  float(np.mean(waits)),
            "total_reward": float(sum(rewards)),
            "throughput": throughput_total,
        }

    # ── Run all 3 systems ──────────────────────────────────────────────────
    # Build n_phases from a probe env so static timer cycles ALL phases
    _probe = make_env("bangalore_corridor", use_gui=False, max_steps=MAX_STEPS, scale=EVAL_SCALE)
    n_phases = {t: int(_probe.action_space.spaces[t].n) for t in tls_ids}
    _probe.close()

    systems = {
        "Static Timer":       _make_static_step_fn(tls_ids, n_phases),
        "Independent PPO":    _make_indep_ppo_step_fn(tls_ids),
        "CTDE MAPPO":         _make_ctde_step_fn(actors),
    }
    colors = {
        "Static Timer":   "#e74c3c",
        "Independent PPO": "#f39c12",
        "CTDE MAPPO":     "#2ecc71",
    }

    results = {}
    for name, step_fn in systems.items():
        print(f"\n[CTDE Compare] Running: {name} ...")
        env = make_env(
            "bangalore_corridor", use_gui=False,
            max_steps=MAX_STEPS, scale=EVAL_SCALE
        )
        results[name] = _run_episode_with_trace(env, step_fn)
        env.close()

        r = results[name]
        print(f"  avg_queue={r['avg_queue']:.1f}  avg_wait={r['avg_wait']:.1f}  "
              f"reward={r['total_reward']:.1f}  throughput={r['throughput']}")

    # ── Generate 3-way comparison plot ────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"3-Way Comparison — Bangalore MG Road (scale={EVAL_SCALE}×)",
                 fontsize=14, fontweight="bold")

    step_axis = np.arange(len(results["Static Timer"]["queues"])) * DELTA_TIME

    for name, res in results.items():
        c = colors[name]
        # Trim all traces to same length
        n = min(len(res["queues"]), len(step_axis))

        axes[0, 0].plot(step_axis[:n], res["queues"][:n], label=name, color=c, alpha=0.85)
        axes[0, 1].plot(step_axis[:n], res["waits"][:n],  label=name, color=c, alpha=0.85)
        axes[1, 0].plot(step_axis[:n], np.cumsum(res["rewards"][:n]), label=name, color=c, alpha=0.85)

    # Bar chart for throughput
    names = list(results.keys())
    throughputs = [results[n]["throughput"] for n in names]
    bar_colors  = [colors[n] for n in names]
    axes[1, 1].bar(names, throughputs, color=bar_colors, alpha=0.85, edgecolor="white")
    axes[1, 1].set_ylabel("Total vehicles arrived")
    axes[1, 1].set_title("Throughput")
    for spine in axes[1, 1].spines.values():
        spine.set_visible(False)

    axes[0, 0].set(title="Queue Length", ylabel="Vehicles waiting", xlabel="Simulation time (s)")
    axes[0, 1].set(title="Total Wait Time", ylabel="Cumulative wait (s)", xlabel="Simulation time (s)")
    axes[1, 0].set(title="Cumulative Reward", ylabel="Reward", xlabel="Simulation time (s)")

    for ax in axes.flat[:3]:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
        for spine in ax.spines.values():
            spine.set_visible(False)

    plt.tight_layout()
    plot_path = RESULTS_DIR / "ctde_3way_comparison.png"
    plt.savefig(str(plot_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[CTDE Compare] Plot saved → {plot_path}")

    # ── Compute improvement percentages ───────────────────────────────────
    ref_q  = results["Static Timer"]["avg_queue"]
    ref_w  = results["Static Timer"]["avg_wait"]
    ref_r  = results["Static Timer"]["total_reward"]

    def pct_improve(val, ref, higher_is_better=False):
        if abs(ref) < 1e-8:
            return 0.0
        return ((ref - val) / abs(ref) * 100) if not higher_is_better else ((val - ref) / abs(ref) * 100)

    summary = {}
    for name in ["Independent PPO", "CTDE MAPPO"]:
        r = results[name]
        summary[f"compare/{name.replace(' ', '_')}/queue_reduction_pct"] = pct_improve(r["avg_queue"], ref_q)
        summary[f"compare/{name.replace(' ', '_')}/wait_reduction_pct"]  = pct_improve(r["avg_wait"],  ref_w)
        summary[f"compare/{name.replace(' ', '_')}/reward_improvement_pct"] = pct_improve(
            r["total_reward"], ref_r, higher_is_better=True)
        summary[f"compare/{name.replace(' ', '_')}/throughput"] = r["throughput"]

    # Raw values for all 3
    for name in names:
        r = results[name]
        key = name.replace(" ", "_")
        summary[f"compare/{key}/avg_queue"]    = r["avg_queue"]
        summary[f"compare/{key}/avg_wait"]     = r["avg_wait"]
        summary[f"compare/{key}/total_reward"] = r["total_reward"]

    wandb.log(summary)
    wandb.log({"compare/3way_plot": wandb.Image(str(plot_path))})

    # ── Print results table ───────────────────────────────────────────────
    print("\n" + "="*65)
    print(f"{'System':<20} {'Avg Queue':>10} {'Avg Wait':>10} {'Reward':>12} {'Throughput':>12}")
    print("-"*65)
    for name in names:
        r = results[name]
        print(f"{name:<20} {r['avg_queue']:>10.1f} {r['avg_wait']:>10.1f} "
              f"{r['total_reward']:>12.1f} {r['throughput']:>12}")
    print("="*65)

    indep_q_imp = summary.get("compare/Independent_PPO/queue_reduction_pct", 0)
    ctde_q_imp  = summary.get("compare/CTDE_MAPPO/queue_reduction_pct", 0)
    print(f"\nVs Static Timer:")
    print(f"  Independent PPO: {indep_q_imp:+.1f}% queue reduction")
    print(f"  CTDE MAPPO:      {ctde_q_imp:+.1f}% queue reduction")

    ctde_vs_indep = ((results["Independent PPO"]["avg_queue"] - results["CTDE MAPPO"]["avg_queue"])
                     / max(results["Independent PPO"]["avg_queue"], 1e-8) * 100)
    print(f"  CTDE vs Independent PPO: {ctde_vs_indep:+.1f}% additional queue reduction")

    wandb.finish()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="CTDE MAPPO evaluation — Bangalore corridor")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--evaluate", action="store_true", help="Headless eval of CTDE actors")
    group.add_argument("--demo",     action="store_true", help="Visual GUI demo of CTDE actors")
    group.add_argument("--compare",  action="store_true", help="3-way comparison: Static vs IndePPO vs CTDE")
    parser.add_argument("--port",    type=int, default=8000)
    args = parser.parse_args()

    if args.evaluate:
        run_ctde_eval(args.port)
    elif args.demo:
        run_ctde_demo(args.port)
    elif args.compare:
        run_ctde_compare(args.port)


if __name__ == "__main__":
    main()
