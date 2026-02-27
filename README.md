# MARL Traffic Control

A reproducible Multi-Agent Reinforcement Learning framework for urban traffic signal control.

## Project Structure

```
marl/
├── src/
│   ├── envs/                              # Gymnasium environments
│   │   ├── __init__.py                    # ENV_REGISTRY — lookup by name
│   │   └── single_intersection.py         # 4-way single intersection
│   ├── agents/                            # RL agents
│   │   └── ppo.py                         # PPO train / eval / demo
│   ├── baselines/                         # Non-learning controllers
│   │   └── static_timer.py                # Fixed 40s green cycle
│   ├── evaluation/                        # Comparison & analysis
│   │   └── compare.py                     # Static vs PPO comparison
│   └── utils/                             # Shared utilities
│       └── metrics.py                     # Prometheus gauges
├── sumo_net/
│   └── single_intersection/               # SUMO network files
├── docker-compose.yml
├── Makefile
├── TECHNICAL.md                           # Full technical spec
└── README.md
```

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  docker-compose                      │
│                                                      │
│  ┌──────────────┐  scrape   ┌────────────┐          │
│  │ traffic-agent │◄─────────│ prometheus │          │
│  │  (SUMO+PPO)  │  :8000   │   :9090    │          │
│  └──────────────┘           └─────┬──────┘          │
│                                   │                  │
│  ┌──────────────┐           ┌─────▼──────┐          │
│  │ traffic-demo │           │  grafana   │          │
│  │ (noVNC+GUI)  │           │   :3000    │          │
│  │    :6080     │           └────────────┘          │
│  └──────────────┘                                    │
│                             ┌────────────┐          │
│                             │ tensorboard│          │
│                             │   :6006    │          │
│                             └────────────┘          │
└─────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# Build and start all services
docker compose up --build

# Or use Make
make build
make up
```

## Services

| Service | URL | Description |
|---------|-----|-------------|
| Grafana | http://localhost:3000 | Metrics dashboard (admin/admin) |
| Prometheus | http://localhost:9010 | Metrics store |
| TensorBoard | http://localhost:6006 | Training curves |
| noVNC Demo | http://localhost:6080 | Live SUMO simulation |
| Metrics | http://localhost:8000 | Prometheus scrape endpoint |

## Local Development

```bash
# One-time setup: create venv with system-site-packages (needed for libsumo)
make venv

# Run static-timer baseline
make local-dumb

# Train PPO agent (default: single_intersection env)
make local-train

# Evaluate saved model
make local-eval

# Visual demo with sumo-gui
make local-demo

# Run offline comparison (static-timer vs PPO) — saves plots to results/
make local-compare

# Launch TensorBoard
make local-tb

# Select a different environment (when available)
make local-train ARGS="--env single_intersection"
```

## Experiment Tracking (Weights & Biases)

```bash
# One-time: login with your free W&B account
make wandb-login

# Train with a descriptive run name and notes (both optional)
make local-train ARGS="--run-name baseline --notes 'Fixed-cycle 40s green baseline'"
make local-train ARGS="--run-name starving-lane-fix --notes 'Added wait-time penalty to eliminate starvation'"

# Control how long to train (default: 100,000 steps)
make local-train ARGS="--run-name long-run --timesteps 1000000"

# Combine all flags
make local-train ARGS="--run-name baseline-1M --timesteps 1000000 --notes 'Full 1M step baseline run'"

# Auto-run static-timer comparison after training (creates a separate W&B eval run)
make local-train ARGS="--run-name baseline --timesteps 100000 --compare-static"
# → Training run: 'baseline'
# → Eval run:     'baseline_vs_static' (with comparison plot uploaded)
```

Every training run automatically captures:
- All hyperparameters (lr, gamma, n_steps, etc.)
- Training curves synced from TensorBoard
- Git commit hash + exact CLI command used
- Model weights versioned as a W&B artifact (`ppo_traffic_model:v0`, `v1`, ...)

Comparison evaluations (`make local-compare`) upload the static-timer vs PPO plot and summary stats to W&B automatically.

## Metrics Exposed

- `traffic_queue_length` — live count of waiting cars
- `traffic_wait_time_total` — cumulative wait time
- `agent_reward_total` — current step reward
- `agent_switch_penalty` — switch penalty applied
- `traffic_throughput` — vehicles that completed their trip per step
