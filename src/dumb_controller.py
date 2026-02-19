"""
Dumb (static-timer) traffic controller baseline.

Cycles through fixed green/yellow phases regardless of traffic conditions.
Logs metrics to Prometheus every step.
"""

import argparse
import time

from src.env import SumoEnv
from src.metrics import start_metrics_server, update


def run_dumb_controller(
    max_steps: int = 3600,
    green_duration: int = 40,
    delta_time: int = 5,
    use_gui: bool = False,
    metrics_port: int = 8000,
) -> None:
    """Run the fixed-cycle controller for one episode."""

    start_metrics_server(metrics_port)

    env = SumoEnv(
        use_gui=use_gui,
        max_steps=max_steps,
        delta_time=delta_time,
    )

    obs, info = env.reset()
    update(info)

    step = 0
    time_in_phase = 0
    total_reward = 0.0

    print(f"[dumb] Starting static-timer controller (green={green_duration}s)")

    terminated = False
    while not terminated:
        # Decide action: switch if we've been green long enough
        if time_in_phase >= green_duration:
            action = 1  # switch
            time_in_phase = 0
        else:
            action = 0  # keep
            time_in_phase += delta_time

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step += 1

        # Update Prometheus
        update(info)

        if step % 20 == 0:
            ql = info["queue_length"]
            wt = info["wait_time_total"]
            print(
                f"[dumb] step={step:4d}  queue={ql:.0f}  "
                f"wait={wt:.0f}  reward={reward:.1f}"
            )

    env.close()
    print(f"[dumb] Done. Total reward: {total_reward:.1f}")


def main():
    parser = argparse.ArgumentParser(description="Dumb static-timer controller")
    parser.add_argument(
        "--max-steps", type=int, default=3600, help="Simulation seconds"
    )
    parser.add_argument(
        "--green-duration", type=int, default=40, help="Green phase length (s)"
    )
    parser.add_argument("--gui", action="store_true", help="Use sumo-gui")
    parser.add_argument(
        "--port", type=int, default=8000, help="Prometheus metrics port"
    )
    args = parser.parse_args()

    run_dumb_controller(
        max_steps=args.max_steps,
        green_duration=args.green_duration,
        use_gui=args.gui,
        metrics_port=args.port,
    )


if __name__ == "__main__":
    main()
