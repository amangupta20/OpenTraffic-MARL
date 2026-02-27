"""
Offline comparison: Dumb (static-timer) vs Smart (PPO) controller.

Runs both controllers headlessly with libsumo, saves per-step metrics
to CSV, and generates side-by-side matplotlib comparison plots.

Usage:
    python -m src.compare [--max-steps 3600]

Output:
    results/dumb_metrics.csv
    results/ppo_metrics.csv
    results/comparison.png
"""

import argparse
import csv
import pathlib
from typing import Any

import matplotlib
matplotlib.use("Agg")  # headless backend — no display needed
import matplotlib.pyplot as plt
import numpy as np

import wandb
from stable_baselines3 import PPO

from src.env import SumoEnv

RESULTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "results"
MODELS_DIR = pathlib.Path(__file__).resolve().parent.parent / "models"

# ── Dumb controller logic (static 40s green cycle) ──────────────────
DUMB_GREEN_DURATION = 40  # seconds per green phase


def run_dumb(max_steps: int, delta_time: int, switch_penalty: float) -> list[dict]:
    """Run the static-timer controller and collect per-step metrics."""
    env = SumoEnv(
        use_gui=False,
        max_steps=max_steps,
        delta_time=delta_time,
        switch_penalty=switch_penalty,
    )
    obs, info = env.reset()
    records: list[dict] = [_record(info, 0)]

    green_timer = 0
    terminated = False
    step = 0

    while not terminated:
        green_timer += delta_time
        action = 1 if green_timer >= DUMB_GREEN_DURATION else 0
        if action == 1:
            green_timer = 0

        obs, reward, terminated, _, info = env.step(action)
        step += 1
        records.append(_record(info, step))

    env.close()
    print(f"[compare] Dumb controller done — {step} steps, reward={sum(r['reward'] for r in records):.0f}")
    return records


# ── Smart controller logic (PPO) ────────────────────────────────────

def run_ppo(max_steps: int, delta_time: int, switch_penalty: float) -> list[dict]:
    """Run the trained PPO agent and collect per-step metrics."""
    model_path = MODELS_DIR / "ppo_traffic.zip"
    if not model_path.exists():
        raise FileNotFoundError(f"No saved model at {model_path}. Train first.")

    env = SumoEnv(
        use_gui=False,
        max_steps=max_steps,
        delta_time=delta_time,
        switch_penalty=switch_penalty,
    )
    model = PPO.load(str(model_path), env=env, device="cpu")

    obs, info = env.reset()
    records: list[dict] = [_record(info, 0)]

    terminated = False
    step = 0

    while not terminated:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, _, info = env.step(int(action))
        step += 1
        records.append(_record(info, step))

    env.close()
    print(f"[compare] PPO controller done — {step} steps, reward={sum(r['reward'] for r in records):.0f}")
    return records


# ── Helpers ──────────────────────────────────────────────────────────

def _record(info: dict[str, Any], step: int) -> dict:
    return {
        "step": step,
        "queue_length": info.get("queue_length", 0),
        "wait_time_total": info.get("wait_time_total", 0),
        "reward": info.get("reward", 0),
        "throughput": info.get("throughput", 0),
    }


def save_csv(records: list[dict], path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    print(f"[compare] Saved {path}")


def smooth(values: list[float], window: int = 10) -> np.ndarray:
    """Simple moving average for smoother plots."""
    if len(values) < window:
        return np.array(values)
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def generate_plots(
    dumb: list[dict], ppo: list[dict], out_path: pathlib.Path
) -> None:
    """Generate side-by-side comparison plots and save to file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    metrics = [
        ("queue_length", "Queue Length (vehicles)", "#e74c3c", "#2ecc71"),
        ("wait_time_total", "Total Wait Time (s)", "#e67e22", "#3498db"),
        ("reward", "Reward per Step", "#9b59b6", "#1abc9c"),
        ("throughput", "Throughput (arrived)", "#c0392b", "#27ae60"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.patch.set_facecolor("#1a1a2e")
    fig.suptitle(
        "Dumb (Static Timer) vs PPO Agent — Traffic Control",
        fontsize=16, fontweight="bold", color="white", y=0.98,
    )

    for ax, (key, title, dumb_color, ppo_color) in zip(axes.flat, metrics):
        ax.set_facecolor("#16213e")

        dumb_vals = [r[key] for r in dumb]
        ppo_vals = [r[key] for r in ppo]

        # Raw data (faint)
        ax.plot(dumb_vals, color=dumb_color, alpha=0.15, linewidth=0.5)
        ax.plot(ppo_vals, color=ppo_color, alpha=0.15, linewidth=0.5)

        # Smoothed (bold)
        window = max(5, len(dumb_vals) // 50)
        ax.plot(smooth(dumb_vals, window), color=dumb_color, linewidth=2, label="Dumb")
        ax.plot(smooth(ppo_vals, window), color=ppo_color, linewidth=2, label="PPO")

        ax.set_title(title, fontsize=12, color="white", pad=8)
        ax.tick_params(colors="white")
        ax.set_xlabel("Step", color="white")
        ax.spines["bottom"].set_color("#444")
        ax.spines["left"].set_color("#444")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc="upper right", facecolor="#0f3460", edgecolor="#444", labelcolor="white")
        ax.grid(True, alpha=0.15, color="white")

    # ── Summary stats box ──
    dumb_total_reward = sum(r["reward"] for r in dumb)
    ppo_total_reward = sum(r["reward"] for r in ppo)
    dumb_avg_queue = np.mean([r["queue_length"] for r in dumb])
    ppo_avg_queue = np.mean([r["queue_length"] for r in ppo])
    improvement = (1 - ppo_total_reward / dumb_total_reward) * 100 if dumb_total_reward != 0 else 0

    summary = (
        f"Total Reward  →  Dumb: {dumb_total_reward:,.0f}  |  PPO: {ppo_total_reward:,.0f}  "
        f"({improvement:+.1f}% improvement)\n"
        f"Avg Queue     →  Dumb: {dumb_avg_queue:.1f}  |  PPO: {ppo_avg_queue:.1f}"
    )
    fig.text(
        0.5, 0.02, summary,
        ha="center", va="center", fontsize=11, color="#ecf0f1",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#0f3460", edgecolor="#3498db", alpha=0.9),
    )

    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    print(f"[compare] Plot saved to {out_path}")
    plt.close(fig)


# ── CLI ──────────────────────────────────────────────────────────────

def run_comparison(
    max_steps: int = 3600,
    delta_time: int = 5,
    switch_penalty: float = 2.0,
    wandb_run_name: str = "offline-comparison",
) -> None:
    """Run full dumb-vs-PPO comparison, save CSVs/plots, and log to W&B."""

    print("=" * 60)
    print("  OFFLINE COMPARISON: Baseline vs PPO")
    print("=" * 60)

    print("\n[1/5] Running baseline (static-timer) controller...")
    dumb_records = run_dumb(max_steps, delta_time, switch_penalty)

    print("\n[2/5] Running PPO controller...")
    ppo_records = run_ppo(max_steps, delta_time, switch_penalty)

    print("\n[3/5] Saving CSV files...")
    save_csv(dumb_records, RESULTS_DIR / "baseline_metrics.csv")
    save_csv(ppo_records, RESULTS_DIR / "ppo_metrics.csv")

    print("\n[4/5] Generating comparison plots...")
    generate_plots(dumb_records, ppo_records, RESULTS_DIR / "comparison.png")

    print("\n[5/5] Logging evaluation to W&B...")
    run = wandb.init(
        project="marl-traffic",
        job_type="evaluation",
        name=wandb_run_name,
        config={
            "max_steps": max_steps,
            "delta_time": delta_time,
            "switch_penalty": switch_penalty,
        },
    )

    dumb_reward = sum(r["reward"] for r in dumb_records)
    ppo_reward = sum(r["reward"] for r in ppo_records)
    dumb_q = np.mean([r["queue_length"] for r in dumb_records])
    ppo_q = np.mean([r["queue_length"] for r in ppo_records])
    
    # Log summary metrics
    wandb.log({
        "eval/baseline_total_reward": dumb_reward,
        "eval/ppo_total_reward": ppo_reward,
        "eval/baseline_avg_queue": dumb_q,
        "eval/ppo_avg_queue": ppo_q,
        "eval/reward_improvement_pct": (1 - ppo_reward / dumb_reward) * 100 if dumb_reward != 0 else 0,
    })

    # Upload plot as media
    wandb.log({"eval/comparison_plot": wandb.Image(str(RESULTS_DIR / "comparison.png"))})
    wandb.finish()

    print("\n" + "=" * 60)
    print("  DONE! Results in: results/ and W&B")
    print("=" * 60)


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compare Baseline vs PPO controllers")
    parser.add_argument("--max-steps", type=int, default=3600, help="Simulation seconds")
    parser.add_argument("--delta-time", type=int, default=5, help="Seconds between decisions")
    parser.add_argument("--switch-penalty", type=float, default=2.0, help="Phase switch penalty")
    args = parser.parse_args()

    run_comparison(
        max_steps=args.max_steps,
        delta_time=args.delta_time,
        switch_penalty=args.switch_penalty,
    )


if __name__ == "__main__":
    main()
