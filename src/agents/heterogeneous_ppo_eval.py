"""
Evaluation & comparison for the Bangalore MG Road heterogeneous MARL setup.

Modes:
  --evaluate   : Run trained parallel PPO agents (headless)
  --static     : Run static-timer baseline (headless)
  --compare    : Run both, generate plot, log results back into the TRAINING W&B run
  --demo       : Visual (sumo-gui) run of trained PPO agents

Usage (via Docker):
  docker compose run --rm -e MODE=blr-compare agent
  docker compose run --rm -e MODE=blr-demo    agent
"""

import argparse
import csv
import json
import pathlib
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import wandb
from stable_baselines3 import PPO

from src.envs import make_env
from src.utils.metrics import start_metrics_server

MODELS_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "models"
RESULTS_DIR = MODELS_DIR.parent / "results"
META_PATH   = MODELS_DIR / "blr_run_metadata.json"

# Curriculum: use eval scale (grad 3 = 0.6) to match end-of-training conditions
EVAL_SCALE  = 0.8
MAX_STEPS   = 1800
DELTA_TIME  = 5
GREEN_DUR   = 30   # static-timer green phase duration (seconds)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _record(info: dict[str, Any], step: int) -> dict:
    return {
        "step": step,
        "queue_length": info.get("queue_length", 0),
        "wait_time_total": info.get("wait_time_total", 0),
        "reward": info.get("reward", 0),
        "throughput": info.get("throughput", 0),
    }


def _save_csv(records: list[dict], path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    print(f"[blr-eval] Saved {path.name}")


def _smooth(values: list[float], window: int = 10) -> np.ndarray:
    if len(values) < window:
        return np.array(values)
    return np.convolve(values, np.ones(window) / window, mode="valid")


def _load_models(tls_ids: list[str]) -> dict[str, PPO]:
    """Load one PPO model per junction cluster from disk."""
    models = {}
    for t in tls_ids:
        path = MODELS_DIR / f"ppo_blr_parallel_{t}.zip"
        if not path.exists():
            raise FileNotFoundError(
                f"No model at {path}.\n"
                f"  Run: make blr-train ARGS='--run-name bangalore-curriculum --timesteps 500000'"
            )
        models[t] = PPO.load(str(path), device="cpu")
    return models


# ---------------------------------------------------------------------------
# Static-timer baseline
# ---------------------------------------------------------------------------

def run_blr_static(metrics_port: int = 8000) -> list[dict]:
    """Fixed green-phase timer for all junctions — Bangalore corridor."""
    start_metrics_server(metrics_port)

    env = make_env("bangalore_corridor", use_gui=False, max_steps=MAX_STEPS,
                   delta_time=DELTA_TIME, scale=EVAL_SCALE)
    tls_ids = env.tls_ids
    obs, info = env.reset()
    records: list[dict] = [_record(info, 0)]

    green_timer = {t: 0 for t in tls_ids}
    terminated = False
    step = 0

    print(f"[blr-static] Running static timer (green={GREEN_DUR}s) on {len(tls_ids)} junctions  "
          f"[scale={EVAL_SCALE}]")

    while not terminated:
        actions = {}
        for t in tls_ids:
            num_phases = len(env.tls_phases[t])
            green_timer[t] += DELTA_TIME
            if green_timer[t] >= GREEN_DUR:
                actions[t] = (env._current_green_phase_idx[t] + 1) % num_phases
                green_timer[t] = 0
            else:
                actions[t] = env._current_green_phase_idx[t]

        obs, reward, terminated, _, info = env.step(actions)
        step += 1
        records.append(_record(info, step))

        if step % 30 == 0:
            print(f"[blr-static] step={step:4d}  queue={info['queue_length']:.0f}  "
                  f"wait={info['wait_time_total']:.0f}  reward={reward:.1f}")

    env.close()
    total_reward = sum(r["reward"] for r in records)
    print(f"[blr-static] Done. Total reward: {total_reward:.1f}")
    return records


# ---------------------------------------------------------------------------
# PPO evaluation
# ---------------------------------------------------------------------------

def run_blr_ppo(metrics_port: int = 8000, use_gui: bool = False) -> list[dict]:
    """Evaluate the trained heterogeneous PPO agents on the Bangalore corridor."""
    start_metrics_server(metrics_port)

    env = make_env("bangalore_corridor", use_gui=use_gui, max_steps=MAX_STEPS,
                   delta_time=DELTA_TIME, scale=EVAL_SCALE)
    tls_ids = env.tls_ids
    models = _load_models(tls_ids)

    obs, info = env.reset()
    records: list[dict] = [_record(info, 0)]

    terminated = False
    step = 0
    mode_tag = "blr-demo" if use_gui else "blr-ppo"

    print(f"[{mode_tag}] Running {len(tls_ids)} independent PPO agents  [scale={EVAL_SCALE}]")

    while not terminated:
        actions = {}
        for t in tls_ids:
            action, _ = models[t].predict(obs[t], deterministic=True)
            actions[t] = int(action)

        obs, reward, terminated, _, info = env.step(actions)
        step += 1
        records.append(_record(info, step))

        if step % 30 == 0:
            print(f"[{mode_tag}] step={step:4d}  queue={info['queue_length']:.0f}  "
                  f"wait={info['wait_time_total']:.0f}  reward={reward:.1f}")

    env.close()
    total_reward = sum(r["reward"] for r in records)
    print(f"[{mode_tag}] Done. Total reward: {total_reward:.1f}")
    return records


# ---------------------------------------------------------------------------
# Comparison plot
# ---------------------------------------------------------------------------

def _generate_comparison_plot(
    static: list[dict],
    ppo: list[dict],
    out_path: pathlib.Path,
    num_agents: int,
) -> None:
    metrics = [
        ("queue_length",    "Queue Length (vehicles)",  "#e74c3c", "#2ecc71"),
        ("wait_time_total", "Total Wait Time (s)",      "#e67e22", "#3498db"),
        ("reward",          "Reward per Step",          "#9b59b6", "#1abc9c"),
        ("throughput",      "Throughput (arrived)",     "#c0392b", "#27ae60"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.patch.set_facecolor("#1a1a2e")
    fig.suptitle(
        f"Static Timer vs Heterogeneous PPO — Bangalore MG Road ({num_agents} Junctions)",
        fontsize=15, fontweight="bold", color="white", y=0.98,
    )

    for ax, (key, title, s_color, p_color) in zip(axes.flat, metrics):
        ax.set_facecolor("#16213e")
        s_vals = [r[key] for r in static]
        p_vals = [r[key] for r in ppo]

        ax.plot(s_vals, color=s_color, alpha=0.15, linewidth=0.5)
        ax.plot(p_vals, color=p_color, alpha=0.15, linewidth=0.5)

        window = max(5, len(s_vals) // 50)
        ax.plot(_smooth(s_vals, window), color=s_color, linewidth=2, label="Static Timer")
        ax.plot(_smooth(p_vals, window), color=p_color, linewidth=2, label=f"PPO (×{num_agents})")

        ax.set_title(title, fontsize=12, color="white", pad=8)
        ax.tick_params(colors="white")
        ax.set_xlabel("Step", color="white")
        for spine in ("bottom", "left"):
            ax.spines[spine].set_color("#444")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc="upper right", facecolor="#0f3460", edgecolor="#444", labelcolor="white")
        ax.grid(True, alpha=0.15, color="white")

    s_total = sum(r["reward"] for r in static)
    p_total = sum(r["reward"] for r in ppo)
    s_q     = np.mean([r["queue_length"] for r in static])
    p_q     = np.mean([r["queue_length"] for r in ppo])
    improvement = (1 - p_total / s_total) * 100 if s_total != 0 else 0

    summary = (
        f"Total Reward  →  Static: {s_total:,.0f}  |  PPO: {p_total:,.0f}  "
        f"({improvement:+.1f}% improvement)\n"
        f"Avg Queue     →  Static: {s_q:.1f}  |  PPO: {p_q:.1f}"
    )
    fig.text(
        0.5, 0.02, summary,
        ha="center", va="center", fontsize=11, color="#ecf0f1",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#0f3460", edgecolor="#3498db", alpha=0.9),
    )

    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    print(f"[blr-compare] Plot saved → {out_path.name}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Full comparison (logs back into the TRAINING W&B run)
# ---------------------------------------------------------------------------

def run_blr_comparison(wandb_run_name: str = "blr-comparison") -> None:
    """Compare static timer vs trained PPO on Bangalore, attach results to training run."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("  BLR COMPARISON: Static Timer vs Heterogeneous PPO")
    print("=" * 64)

    print("\n[1/5] Running static-timer baseline...")
    static_records = run_blr_static(metrics_port=8000)

    print("\n[2/5] Running PPO agents...")
    ppo_records = run_blr_ppo(metrics_port=8001)

    print("\n[3/5] Saving CSVs...")
    _save_csv(static_records, RESULTS_DIR / "blr_static_metrics.csv")
    _save_csv(ppo_records,    RESULTS_DIR / "blr_ppo_metrics.csv")

    print("\n[4/5] Generating comparison plot...")
    plot_path = RESULTS_DIR / "blr_comparison.png"
    # Use count from already-run eval rather than spinning up a new env
    num_agents = len([p for p in MODELS_DIR.glob("ppo_blr_parallel_*.zip")])
    _generate_comparison_plot(static_records, ppo_records, plot_path, num_agents)

    s_total     = sum(r["reward"] for r in static_records)
    p_total     = sum(r["reward"] for r in ppo_records)
    s_q         = np.mean([r["queue_length"] for r in static_records])
    p_q         = np.mean([r["queue_length"] for r in ppo_records])
    s_wait      = np.mean([r["wait_time_total"] for r in static_records])
    p_wait      = np.mean([r["wait_time_total"] for r in ppo_records])
    improvement = (1 - p_total / s_total) * 100 if s_total != 0 else 0

    print("\n[5/5] Logging evaluation to W&B...")

    # ── Try to resume the original training run ──────────────────────────────
    init_kwargs: dict = dict(project="marl-traffic", job_type="evaluation")
    if META_PATH.exists():
        meta = json.loads(META_PATH.read_text())
        print(f"[wandb] Resuming training run '{meta['run_name']}' ({meta['run_id']})")
        init_kwargs.update(id=meta["run_id"], resume="allow", name=meta["run_name"])
    else:
        print(f"[wandb] No metadata found — creating new run '{wandb_run_name}'")
        init_kwargs["name"] = wandb_run_name

    wandb.init(**init_kwargs)

    wandb.log({
        "blr_eval/static_total_reward":  s_total,
        "blr_eval/ppo_total_reward":     p_total,
        "blr_eval/static_avg_queue":     s_q,
        "blr_eval/ppo_avg_queue":        p_q,
        "blr_eval/static_avg_wait":      s_wait,
        "blr_eval/ppo_avg_wait":         p_wait,
        "blr_eval/reward_improvement_pct": improvement,
        "blr_eval/queue_reduction_pct":  (1 - p_q / s_q) * 100 if s_q != 0 else 0,
        "blr_eval/comparison_plot":      wandb.Image(str(plot_path)),
    })

    # Also log per-step curves so they appear on the run page
    for i, (sr, pr) in enumerate(zip(static_records, ppo_records)):
        wandb.log({
            "eval/step":                i,
            "eval/static_queue":        sr["queue_length"],
            "eval/ppo_queue":           pr["queue_length"],
            "eval/static_reward":       sr["reward"],
            "eval/ppo_reward":          pr["reward"],
            "eval/static_throughput":   sr["throughput"],
            "eval/ppo_throughput":      pr["throughput"],
        }, step=None)

    wandb.finish()

    print("\n" + "=" * 64)
    print(f"  Static Timer Total Reward:  {s_total:,.0f}")
    print(f"  PPO Agents Total Reward:    {p_total:,.0f}")
    print(f"  Reward Improvement:         {improvement:+.1f}%")
    print(f"  Queue  Reduction:           {(1-p_q/s_q)*100 if s_q else 0:+.1f}%")
    print("  Results synced to W&B training run ✓")
    print("=" * 64)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Heterogeneous PPO eval — Bangalore MG Road")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--evaluate", action="store_true", help="Run trained PPO agents (headless)")
    group.add_argument("--static",   action="store_true", help="Run static-timer baseline")
    group.add_argument("--compare",  action="store_true", help="Compare static vs PPO → W&B")
    group.add_argument("--demo",     action="store_true", help="Visual demo (sumo-gui)")
    parser.add_argument("--run-name", type=str, default="blr-comparison")
    parser.add_argument("--port",     type=int, default=8000)
    args = parser.parse_args()

    if args.evaluate:
        run_blr_ppo(metrics_port=args.port)
    elif args.static:
        run_blr_static(metrics_port=args.port)
    elif args.compare:
        run_blr_comparison(wandb_run_name=args.run_name)
    elif args.demo:
        run_blr_ppo(metrics_port=args.port, use_gui=True)


if __name__ == "__main__":
    main()
