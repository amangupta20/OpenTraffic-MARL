"""
Smart (PPO) traffic controller.

Modes:
  --train     Train with libsumo (fast, headless). Saves model + TB logs.
  --evaluate  Load saved model, run deterministic eval with libsumo.
  --demo      Load saved model, run with sumo-gui for visual demo.

All modes log metrics to Prometheus.
"""

import argparse
import os
import pathlib
import time

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed

import wandb
from wandb.integration.sb3 import WandbCallback

from src.env import SumoEnv
from src.metrics import start_metrics_server, update
from src.compare import run_comparison

MODELS_DIR = pathlib.Path(__file__).resolve().parent.parent / "models"
TB_LOG_DIR = pathlib.Path(__file__).resolve().parent.parent / "tb_logs"


# ---------------------------------------------------------------------------
# Custom callback to push per-step metrics to Prometheus during training
# ---------------------------------------------------------------------------
class MetricsCallback(BaseCallback):
    """Push env info to Prometheus after each step during training."""

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            update(info)
        return True


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
def train(
    total_timesteps: int = 100_000,
    delta_time: int = 5,
    switch_penalty: float = 2.0,
    metrics_port: int = 8000,
    learning_rate: float = 3e-4,
    device: str = "auto",
    num_envs: int = 4,
    run_name: str | None = None,
    notes: str | None = None,
) -> None:
    """Train PPO agent with libsumo backend (supports parallel envs)."""

    start_metrics_server(metrics_port)
    MODELS_DIR.mkdir(exist_ok=True)
    TB_LOG_DIR.mkdir(exist_ok=True)

    # Initialize weights & biases (W&B)
    run_config = {
        "learning_rate": learning_rate,
        "total_timesteps": total_timesteps,
        "delta_time": delta_time,
        "switch_penalty": switch_penalty,
        "num_envs": num_envs,
        "n_steps": 512,
        "batch_size": 128,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
    }
    
    run = wandb.init(
        project="marl-traffic",
        name=run_name,
        notes=notes,
        config=run_config,
        sync_tensorboard=True,  # Auto-upload TB metrics
        monitor_gym=True,
        save_code=True,         # Log code state
    )

    # Factory for creating env instances
    def make_env(rank: int, seed: int = 0):
        def _init():
            e = SumoEnv(
                use_gui=False,
                delta_time=delta_time,
                switch_penalty=switch_penalty,
                sumo_seed=seed + rank,
            )
            return e
        return _init

    # Create vectorized environment
    # libsumo requires SubprocVecEnv because it uses global state
    env = make_vec_env(
        make_env(0),
        n_envs=num_envs,
        seed=42,
        vec_env_cls=SubprocVecEnv,
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=learning_rate,
        n_steps=512,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        tensorboard_log=str(TB_LOG_DIR),
        device=device,
    )

    print(f"[smart] Using device: {model.device} with {num_envs} parallel environments")

    # Callbacks: Prometheus + W&B
    callbacks = CallbackList([
        MetricsCallback(),
        WandbCallback(
            gradient_save_freq=100,
            model_save_path=f"models/{run.id}",
            verbose=2,
        )
    ])

    print(f"[smart] Training PPO for {total_timesteps} timesteps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        tb_log_name="ppo_traffic",
    )

    save_path = MODELS_DIR / "ppo_traffic"
    model.save(str(save_path))
    print(f"[smart] Model saved to {save_path}.zip")

    # Version the model weights artifact in W&B
    artifact = wandb.Artifact(
        name="ppo_traffic_model",
        type="model",
        description=f"PPO traffic controller weights (run: {run.name})",
    )
    artifact.add_file(str(save_path) + ".zip")
    run.log_artifact(artifact)

    env.close()
    wandb.finish()


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------
def evaluate(
    delta_time: int = 5,
    switch_penalty: float = 2.0,
    use_gui: bool = False,
    metrics_port: int = 8000,
    max_steps: int = 3600,
) -> None:
    """Load saved model and run a deterministic evaluation episode."""

    start_metrics_server(metrics_port)
    model_path = MODELS_DIR / "ppo_traffic.zip"
    if not model_path.exists():
        raise FileNotFoundError(
            f"No saved model at {model_path}. Train first with --train."
        )

    env = SumoEnv(
        use_gui=use_gui,
        max_steps=max_steps,
        delta_time=delta_time,
        switch_penalty=switch_penalty,
    )

    model = PPO.load(str(model_path), env=env)

    obs, info = env.reset()
    update(info)

    total_reward = 0.0
    step = 0
    terminated = False

    mode_label = "demo" if use_gui else "eval"
    print(f"[smart-{mode_label}] Running evaluation...")

    while not terminated:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        total_reward += reward
        step += 1
        update(info)

        if step % 20 == 0:
            ql = info["queue_length"]
            wt = info["wait_time_total"]
            print(
                f"[smart-{mode_label}] step={step:4d}  queue={ql:.0f}  "
                f"wait={wt:.0f}  reward={reward:.1f}"
            )

    env.close()
    print(f"[smart-{mode_label}] Done. Total reward: {total_reward:.1f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Smart PPO traffic controller")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train", action="store_true", help="Train PPO agent")
    group.add_argument(
        "--evaluate", action="store_true", help="Evaluate saved model (headless)"
    )
    group.add_argument(
        "--demo", action="store_true", help="Visual demo with sumo-gui"
    )

    parser.add_argument(
        "--timesteps", type=int, default=100_000, help="Training timesteps"
    )
    parser.add_argument(
        "--switch-penalty", type=float, default=2.0, help="Phase switch penalty α"
    )
    parser.add_argument(
        "--lr", type=float, default=3e-4, help="Learning rate"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Prometheus metrics port"
    )
    parser.add_argument(
        "--max-steps", type=int, default=3600, help="Simulation seconds"
    )
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cpu", "cuda"],
        help="Device for training: auto (detect GPU), cpu, or cuda",
    )
    parser.add_argument(
        "--num-envs", type=int, default=4, help="Number of parallel envs (CPU cores)"
    )
    parser.add_argument(
        "--run-name", type=str, default=None, help="Custom name for the W&B run (e.g., 'baseline')"
    )
    parser.add_argument(
        "--notes", type=str, default=None, help="Detailed notes for the W&B run (e.g., 'fixing starvation')"
    )
    parser.add_argument(
        "--compare-static", action="store_true",
        help="Run static-timer vs PPO comparison after training completes"
    )
    args = parser.parse_args()

    if args.train:
        train(
            total_timesteps=args.timesteps,
            switch_penalty=args.switch_penalty,
            metrics_port=args.port,
            learning_rate=args.lr,
            device=args.device,
            num_envs=args.num_envs,
            run_name=args.run_name,
            notes=args.notes,
        )
        if args.compare_static:
            eval_name = f"{args.run_name or 'ppo'}_vs_static"
            print(f"\n[smart] Running post-training comparison as '{eval_name}'...")
            run_comparison(
                delta_time=5,
                switch_penalty=args.switch_penalty,
                wandb_run_name=eval_name,
            )
    elif args.evaluate:
        evaluate(
            switch_penalty=args.switch_penalty,
            metrics_port=args.port,
            max_steps=args.max_steps,
        )
    elif args.demo:
        evaluate(
            use_gui=True,
            switch_penalty=args.switch_penalty,
            metrics_port=args.port,
            max_steps=args.max_steps,
        )


if __name__ == "__main__":
    main()
