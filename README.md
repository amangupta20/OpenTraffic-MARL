# MARL Traffic Control

A reproducible Multi-Agent Reinforcement Learning framework for urban traffic signal control.
**Docker-first** — all experiments run in containers for complete reproducibility.

## Project Structure

```
marl/
├── src/
│   ├── envs/                              # Gymnasium environments
│   │   ├── __init__.py                    # ENV_REGISTRY
│   │   ├── single_intersection.py         # Stage 1: single 4-way intersection
│   │   └── grid_2x2.py                   # Stage 2: 2×2 grid (4 junctions)
│   ├── agents/                            # RL agents
│   │   ├── ppo.py                         # Single-agent PPO train / eval
│   │   └── independent_ppo.py             # Multi-agent: cloned PPO per junction
│   ├── baselines/                         # Non-learning controllers
│   │   └── static_timer.py                # Fixed-cycle baseline
│   ├── evaluation/                        # Comparison & analysis
│   │   └── compare.py                     # Static vs PPO comparison
│   └── utils/
│       └── metrics.py                     # Prometheus gauges
├── sumo_net/
│   ├── single_intersection/               # Stage 1 network
│   └── grid_2x2/                          # Stage 2 network (4 TLS junctions)
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── TECHNICAL.md
└── README.md
```

## Quick Start

```bash
# Build the container (once)
make build

# Train PPO agent on single intersection (100K steps)
make train ARGS="--run-name baseline --timesteps 100000"

# Evaluate trained model
make eval

# Run static-timer baseline
make dumb

# Compare static-timer vs PPO (generates plots)
make compare

# Visual demo (sumo-gui at http://localhost:6080)
make demo
```

## Stage 2: 2×2 Grid — Independent Multi-Agent

Deploy the trained single-intersection model across 4 junctions (zero-shot cloning):

```bash
# Evaluate 4 cloned PPO agents on the 2×2 grid
make grid-eval

# Run static-timer baseline on the grid (30s green cycles)
make grid-static

# Compare static vs cloned PPO (generates plot + logs to W&B)
make grid-compare ARGS="--run-name zero-shot-baseline"

# Visual demo via noVNC (sumo-gui at http://localhost:6080)
make grid-demo
```

## Experiment Tracking (Weights & Biases)

```bash
# Set your W&B API key (get it from https://wandb.ai/authorize)
export WANDB_API_KEY=your_key_here

# Train with descriptive run name and notes
make train ARGS="--run-name baseline --notes 'Original queue-based reward' --timesteps 100000"

# Auto-run comparison after training
make train ARGS="--run-name baseline --timesteps 100000 --compare-static"
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
