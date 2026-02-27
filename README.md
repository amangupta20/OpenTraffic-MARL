# MARL Traffic Control

A reproducible Multi-Agent Reinforcement Learning framework for urban traffic signal control.
**Docker-first** — all experiments run in containers for complete reproducibility.

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
├── Dockerfile
├── Makefile
├── TECHNICAL.md                           # Full technical spec
└── README.md
```

## Quick Start

```bash
# Build the container (once)
make build

# Train PPO agent (100K steps by default)
make train ARGS="--run-name baseline --timesteps 100000"

# Evaluate trained model
make eval

# Run static-timer baseline
make dumb

# Compare static-timer vs PPO (generates plots + logs to W&B)
make compare

# Visual demo (sumo-gui at http://localhost:6080)
make demo
```

## Experiment Tracking (Weights & Biases)

```bash
# Set your W&B API key (get it from https://wandb.ai/authorize)
export WANDB_API_KEY=your_key_here

# Train with a descriptive run name and notes
make train ARGS="--run-name baseline --notes 'Original queue-based reward' --timesteps 100000"

# Auto-run comparison after training
make train ARGS="--run-name baseline --timesteps 100000 --compare-static"

# Select a different environment (when available)
make train ARGS="--env single_intersection --run-name grid-test"
```

Every training run automatically captures:
- All hyperparameters (lr, gamma, n_steps, etc.)
- Training curves synced from TensorBoard
- Git commit hash + exact CLI command used
- Model weights versioned as a W&B artifact (`ppo_traffic_model:v0`, `v1`, ...)

## Monitoring

```bash
# Start Prometheus + Grafana + TensorBoard
make dashboard

# TensorBoard only
make tb
```

| Service | URL | Description |
|---------|-----|-------------|
| Grafana | http://localhost:3000 | Metrics dashboard (admin/admin) |
| Prometheus | http://localhost:9010 | Metrics store |
| TensorBoard | http://localhost:6006 | Training curves |
| noVNC Demo | http://localhost:6080 | Live SUMO simulation |

## Lifecycle

```bash
# Stop all services
make down

# Clean persistent data (models, logs, results)
make clean
```

## Metrics Exposed

- `traffic_queue_length` — live count of waiting cars
- `traffic_wait_time_total` — cumulative wait time
- `agent_reward_total` — current step reward
- `agent_switch_penalty` — switch penalty applied
- `traffic_throughput` — vehicles that completed their trip per step
