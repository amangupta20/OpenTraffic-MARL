.PHONY: build train eval dumb compare grid-eval grid-static grid-compare grid-demo demo tb dashboard down logs clean wandb-login

# ═══════════════════════════════════════════════════════════════════
# Docker-first workflow (reproducible, host-independent)
# ═══════════════════════════════════════════════════════════════════

# Build the container image
build:
	docker compose build

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
	docker compose --profile demo run --rm --service-ports -e MODE=grid-demo demo $(ARGS)

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
