.PHONY: build train eval dumb compare grid-eval grid-static grid-compare grid-demo demo tb dashboard down logs clean wandb-login blr-train blr-eval blr-static blr-compare blr-demo c8-ppo-train

# ═══════════════════════════════════════════════════════════════════
# Docker-first workflow (reproducible, host-independent)
# ═══════════════════════════════════════════════════════════════════

# Build all container images (including demo)
build:
	docker compose --progress=plain --profile demo build

# Train PPO agent
#   make train ARGS="--run-name baseline --timesteps 100000 --compare-static"
train:
	docker compose run --rm -e MODE=train agent $(ARGS)

# Evaluate saved model (headless)
eval:
	docker compose run --rm -e MODE=evaluate agent $(ARGS)

# Run static-timer baseline
dumb:
	docker compose run --rm -e MODE=dumb agent $(ARGS)

# Run offline comparison (static-timer vs PPO)
compare:
	docker compose run --rm -e MODE=compare agent $(ARGS)

# ═══════════════════════════════════════════════════════════════════
# Stage 2: 2×2 Grid — Independent Multi-Agent
# ═══════════════════════════════════════════════════════════════════

# Evaluate cloned PPO agents on 2×2 grid
grid-eval:
	docker compose run --rm -e MODE=grid-eval agent $(ARGS)

# Run static-timer baseline on 2×2 grid
grid-static:
	docker compose run --rm -e MODE=grid-static agent $(ARGS)

# Compare static vs cloned PPO on 2×2 grid
grid-compare:
	docker compose run --rm -e MODE=grid-compare agent $(ARGS)

# Visual demo of 4 cloned PPO agents on 2×2 grid (noVNC at http://localhost:6080)
grid-demo:
	docker compose run --rm -p 6080:6080 -e MODE=grid-demo agent $(ARGS)

# ═══════════════════════════════════════════════════════════════════
# Stage 3: Bangalore MG Road — Heterogeneous Multi-Agent (Curriculum)
# ═══════════════════════════════════════════════════════════════════

# Preprocess: remove internal-compound trips, reduce density (run once before training)
#   make blr-filter-trips DENSITY=0.45
blr-filter-trips:
	docker compose run --rm --entrypoint python3 agent scripts/filter_trips.py --density-factor $(or $(DENSITY),0.45)

# Train 5 heterogeneous PPO agents simultaneously using Curriculum Learning
blr-train:
	docker compose run --rm -e MODE=blr-train agent $(ARGS)

# Evaluate trained PPO agents on Bangalore corridor (headless)
blr-eval:
	docker compose run --rm -e MODE=blr-eval agent $(ARGS)

# Run static-timer baseline on Bangalore corridor
blr-static:
	docker compose run --rm -e MODE=blr-static agent $(ARGS)

# Compare static timer vs trained PPO — logs results to training W&B run
blr-compare:
	docker compose run --rm -e MODE=blr-compare agent $(ARGS)

# Visual demo (sumo-gui streamed via noVNC at http://localhost:6080)
blr-demo:
	docker compose run --rm -p 6080:6080 -e MODE=blr-demo agent $(ARGS)

# Visual demo of static timer baseline (sumo-gui streamed via noVNC)
blr-dumb-demo:
	docker compose run --rm -p 6080:6080 -e MODE=blr-dumb-demo agent $(ARGS)

# ──────────────────────────────────────────────────────────────
# Phase 3: CTDE MAPPO (Centralized Training, Decentralized Execution)
# ──────────────────────────────────────────────────────────────

# Train independent PPO agents on Cologne8
c8-ppo-train:
	docker compose run --rm -e MODE=c8-ppo-train agent $(ARGS)

# Train CTDE MAPPO agents (curriculum 0.6→0.8→1.0)
# Usage: make ctde-train ARGS="--timesteps 500000 --run-name ctde-run-1"
ctde-train:
	docker compose run --rm -e MODE=ctde-train agent $(ARGS)

# Decentralized eval — actors only, no critic needed
ctde-eval:
	docker compose run --rm -e MODE=ctde-eval agent $(ARGS)

# 3-way comparison: Static Timer vs Independent PPO vs CTDE MAPPO
ctde-compare:
	docker compose run --rm -e MODE=ctde-compare agent $(ARGS)

# Visual CTDE demo (sumo-gui streamed via noVNC at http://localhost:6080)
ctde-demo:
	docker compose run --rm -p 6080:6080 -e MODE=ctde-demo agent $(ARGS)

# Visual demo (sumo-gui streamed via noVNC at http://localhost:6080)
demo:
	docker compose --profile demo up demo

# Login to Weights & Biases (interactive — saves key to host)
wandb-login:
	docker compose run --rm -e MODE=wandb-login -it agent

# Start monitoring stack (Prometheus + Grafana + TensorBoard)
dashboard:
	docker compose --profile monitoring up -d prometheus grafana tensorboard
	@echo ""
	@echo "  Grafana:     http://localhost:3000  (admin/admin)"
	@echo "  Prometheus:  http://localhost:9010"
	@echo "  TensorBoard: http://localhost:6006"
	@echo ""

# TensorBoard only
tb:
	docker compose --profile monitoring up tensorboard

# Stop all services
down:
	docker compose --profile demo --profile monitoring down

# Tail logs
logs:
	docker compose logs -f

# Clean persistent data
clean:
	rm -rf models/ tb_logs/ results/
