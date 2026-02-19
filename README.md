# MARL Traffic Control

A reproducible Multi-Agent Reinforcement Learning framework for urban traffic signal control.

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

## Modes

```bash
# Run dumb (static timer) baseline
MODE=dumb docker compose up traffic-agent

# Train PPO agent (fast, uses libsumo)
MODE=train docker compose up traffic-agent

# Evaluate trained agent
MODE=evaluate docker compose up traffic-agent

# Visual demo (streams sumo-gui via noVNC)
docker compose up traffic-demo
```

## Local Development (Arch Linux)

```bash
# One-time setup: create venv with system-site-packages (needed for libsumo)
make venv

# Run dumb baseline
make local-dumb

# Train PPO agent
make local-train

# Evaluate saved model
make local-eval

# Visual demo with sumo-gui
make local-demo
```

## Metrics Exposed

- `traffic_queue_length` — live count of waiting cars
- `traffic_wait_time_total` — cumulative wait time
- `agent_reward_total` — current step reward
- `agent_switch_penalty` — switch penalty applied
