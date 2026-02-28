"""
Independent PPO controller for multi-junction grids.

Loads a trained single-intersection PPO model and deploys it independently
on each junction in a multi-agent environment (zero-shot cloning).

Usage:
    python -m src.agents.independent_ppo --evaluate
    python -m src.agents.independent_ppo --evaluate --env grid_2x2
"""

import argparse
import pathlib
from typing import Any

import numpy as np
from stable_baselines3 import PPO

from src.envs import make_env, ENV_REGISTRY
from src.envs.grid_2x2 import TLS_IDS
from src.utils.metrics import start_metrics_server, update

MODELS_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "models"


# ---------------------------------------------------------------------------
# Static timer for grid (baseline)
# ---------------------------------------------------------------------------
def run_grid_static(
    env_name: str = "grid_2x2",
    max_steps: int = 3600,
    delta_time: int = 5,
    switch_penalty: float = 2.0,
    green_duration: int = 30,
    metrics_port: int = 8000,
) -> list[dict]:
    """Run static-timer baseline on all junctions."""
    start_metrics_server(metrics_port)

    env = make_env(env_name, use_gui=False, max_steps=max_steps,
                   delta_time=delta_time, switch_penalty=switch_penalty)
    obs, info = env.reset()
    records: list[dict] = [_record(info, 0)]

    green_timer = {t: 0 for t in TLS_IDS}
    terminated = False
    step = 0

    print(f"[grid-static] Running static timer (green={green_duration}s) on {len(TLS_IDS)} junctions")

    while not terminated:
        actions = {}
        for t in TLS_IDS:
            green_timer[t] += delta_time
            if green_timer[t] >= green_duration:
                actions[t] = 1  # switch
                green_timer[t] = 0
            else:
                actions[t] = 0  # keep

        obs, reward, terminated, _, info = env.step(actions)
        step += 1
        records.append(_record(info, step))
        update(info)

        if step % 20 == 0:
            print(
                f"[grid-static] step={step:4d}  queue={info['queue_length']:.0f}  "
                f"wait={info['wait_time_total']:.0f}  reward={reward:.1f}"
            )

    env.close()
    total_reward = sum(r["reward"] for r in records)
    print(f"[grid-static] Done. Total reward: {total_reward:.1f}")
    return records


# ---------------------------------------------------------------------------
# Independent PPO for grid (zero-shot from single-intersection model)
# ---------------------------------------------------------------------------
def run_grid_ppo(
    env_name: str = "grid_2x2",
    max_steps: int = 3600,
    delta_time: int = 5,
    switch_penalty: float = 2.0,
    metrics_port: int = 8000,
    use_gui: bool = False,
) -> list[dict]:
    """Run cloned PPO agents (one per junction) on the grid."""
    model_path = MODELS_DIR / "ppo_traffic.zip"
    if not model_path.exists():
        raise FileNotFoundError(
            f"No saved model at {model_path}. Train first with: make train"
        )

    start_metrics_server(metrics_port)
    model = PPO.load(str(model_path), device="cpu")

    env = make_env(env_name, use_gui=use_gui, max_steps=max_steps,
                   delta_time=delta_time, switch_penalty=switch_penalty)
    obs, info = env.reset()
    records: list[dict] = [_record(info, 0)]

    terminated = False
    step = 0

    mode = "grid-demo" if use_gui else "grid-ppo"
    print(f"[{mode}] Running {len(TLS_IDS)} cloned PPO agents (zero-shot)")

    while not terminated:
        # Each agent independently predicts from its local observation
        actions = {}
        for tls_id in TLS_IDS:
            action, _ = model.predict(obs[tls_id], deterministic=True)
            actions[tls_id] = int(action)

        obs, reward, terminated, _, info = env.step(actions)
        step += 1
        records.append(_record(info, step))
        update(info)

        if step % 20 == 0:
            print(
                f"[{mode}] step={step:4d}  queue={info['queue_length']:.0f}  "
                f"wait={info['wait_time_total']:.0f}  reward={reward:.1f}"
            )

    env.close()
    total_reward = sum(r["reward"] for r in records)
    print(f"[{mode}] Done. Total reward: {total_reward:.1f}")
    return records


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------
def run_grid_comparison(
    env_name: str = "grid_2x2",
    max_steps: int = 3600,
    delta_time: int = 5,
    switch_penalty: float = 2.0,
    wandb_run_name: str = "grid-comparison",
) -> None:
    """Run grid static-timer vs cloned PPO comparison, log to W&B."""
    import csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import wandb

    results_dir = MODELS_DIR.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  GRID COMPARISON: Static Timer vs Cloned PPO (4 agents)")
    print("=" * 60)

    print("\n[1/5] Running static-timer baseline on grid...")
    static_records = run_grid_static(
        env_name=env_name, max_steps=max_steps,
        delta_time=delta_time, switch_penalty=switch_penalty,
    )

    print("\n[2/5] Running cloned PPO agents on grid...")
    ppo_records = run_grid_ppo(
        env_name=env_name, max_steps=max_steps,
        delta_time=delta_time, switch_penalty=switch_penalty,
        metrics_port=8001,
    )

    print("\n[3/5] Saving CSV files...")
    _save_csv(static_records, results_dir / "grid_static_metrics.csv")
    _save_csv(ppo_records, results_dir / "grid_ppo_metrics.csv")

    print("\n[4/5] Generating comparison plot...")
    plot_path = results_dir / "grid_comparison.png"
    _generate_plot(static_records, ppo_records, plot_path)

    static_reward = sum(r["reward"] for r in static_records)
    ppo_reward = sum(r["reward"] for r in ppo_records)
    static_q = np.mean([r["queue_length"] for r in static_records])
    ppo_q = np.mean([r["queue_length"] for r in ppo_records])
    improvement = (1 - ppo_reward / static_reward) * 100 if static_reward != 0 else 0

    print("\n[5/5] Logging evaluation to W&B...")
    run = wandb.init(
        project="marl-traffic",
        job_type="grid-evaluation",
        name=wandb_run_name,
        config={
            "env": env_name,
            "max_steps": max_steps,
            "delta_time": delta_time,
            "switch_penalty": switch_penalty,
            "num_agents": 4,
            "agent_type": "independent_ppo_zero_shot",
            "baseline": "static_timer_30s",
        },
    )

    wandb.log({
        "grid_eval/static_total_reward": static_reward,
        "grid_eval/ppo_total_reward": ppo_reward,
        "grid_eval/static_avg_queue": static_q,
        "grid_eval/ppo_avg_queue": ppo_q,
        "grid_eval/reward_improvement_pct": improvement,
    })

    wandb.log({"grid_eval/comparison_plot": wandb.Image(str(plot_path))})
    wandb.finish()

    print("\n" + "=" * 60)
    print(f"  Static Timer Total Reward:  {static_reward:,.0f}")
    print(f"  Cloned PPO Total Reward:    {ppo_reward:,.0f}")
    print(f"  Improvement:                {improvement:+.1f}%")
    print("  Results logged to W&B ✓")
    print("=" * 60)


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
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    print(f"[grid] Saved {path}")


def _smooth(values: list[float], window: int = 10) -> np.ndarray:
    if len(values) < window:
        return np.array(values)
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def _generate_plot(
    static: list[dict], ppo: list[dict], out_path: pathlib.Path
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = [
        ("queue_length", "Queue Length (vehicles)", "#e74c3c", "#2ecc71"),
        ("wait_time_total", "Total Wait Time (s)", "#e67e22", "#3498db"),
        ("reward", "Reward per Step", "#9b59b6", "#1abc9c"),
        ("throughput", "Throughput (arrived)", "#c0392b", "#27ae60"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.patch.set_facecolor("#1a1a2e")
    fig.suptitle(
        "Static Timer vs Cloned PPO — 2×2 Grid (4 Junctions)",
        fontsize=16, fontweight="bold", color="white", y=0.98,
    )

    for ax, (key, title, s_color, p_color) in zip(axes.flat, metrics):
        ax.set_facecolor("#16213e")
        s_vals = [r[key] for r in static]
        p_vals = [r[key] for r in ppo]

        ax.plot(s_vals, color=s_color, alpha=0.15, linewidth=0.5)
        ax.plot(p_vals, color=p_color, alpha=0.15, linewidth=0.5)

        window = max(5, len(s_vals) // 50)
        ax.plot(_smooth(s_vals, window), color=s_color, linewidth=2, label="Static")
        ax.plot(_smooth(p_vals, window), color=p_color, linewidth=2, label="PPO (×4)")

        ax.set_title(title, fontsize=12, color="white", pad=8)
        ax.tick_params(colors="white")
        ax.set_xlabel("Step", color="white")
        ax.spines["bottom"].set_color("#444")
        ax.spines["left"].set_color("#444")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc="upper right", facecolor="#0f3460", edgecolor="#444", labelcolor="white")
        ax.grid(True, alpha=0.15, color="white")

    # Summary stats
    s_total = sum(r["reward"] for r in static)
    p_total = sum(r["reward"] for r in ppo)
    s_q = np.mean([r["queue_length"] for r in static])
    p_q = np.mean([r["queue_length"] for r in ppo])
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
    print(f"[grid] Plot saved to {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Independent PPO agents for multi-junction grids"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--evaluate", action="store_true", help="Run cloned PPO agents")
    group.add_argument("--demo", action="store_true", help="Visual demo (sumo-gui)")
    group.add_argument("--static", action="store_true", help="Run static-timer baseline")
    group.add_argument("--compare", action="store_true", help="Compare static vs PPO")

    parser.add_argument("--env", type=str, default="grid_2x2",
                        choices=list(ENV_REGISTRY.keys()), help="Environment")
    parser.add_argument("--max-steps", type=int, default=3600, help="Simulation seconds")
    parser.add_argument("--delta-time", type=int, default=5, help="Decision interval")
    parser.add_argument("--switch-penalty", type=float, default=2.0, help="Switch penalty α")
    parser.add_argument("--run-name", type=str, default="grid-comparison",
                        help="W&B run name for comparison")
    parser.add_argument("--port", type=int, default=8000, help="Prometheus port")
    args = parser.parse_args()

    if args.evaluate:
        run_grid_ppo(
            env_name=args.env, max_steps=args.max_steps,
            delta_time=args.delta_time, switch_penalty=args.switch_penalty,
            metrics_port=args.port,
        )
    elif args.demo:
        run_grid_ppo(
            env_name=args.env, max_steps=args.max_steps,
            delta_time=args.delta_time, switch_penalty=args.switch_penalty,
            metrics_port=args.port, use_gui=True,
        )
    elif args.static:
        run_grid_static(
            env_name=args.env, max_steps=args.max_steps,
            delta_time=args.delta_time, switch_penalty=args.switch_penalty,
            metrics_port=args.port,
        )
    elif args.compare:
        run_grid_comparison(
            env_name=args.env, max_steps=args.max_steps,
            delta_time=args.delta_time, switch_penalty=args.switch_penalty,
            wandb_run_name=args.run_name,
        )


if __name__ == "__main__":
    main()
