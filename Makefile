.PHONY: build up down train evaluate demo logs clean

# Build Docker images
build:
	docker compose build

# Start core stack (agent + prometheus + grafana)
up:
	docker compose up -d

# Start with training profile (includes TensorBoard)
train:
	MODE=train docker compose --profile train up -d

# Start visual demo
demo:
	docker compose --profile demo up -d

# Evaluate saved model
evaluate:
	MODE=evaluate docker compose up -d traffic-agent

# Stop all services
down:
	docker compose --profile demo --profile train down

# Tail logs
logs:
	docker compose logs -f

# Clean models and TB logs
clean:
	rm -rf models/*.zip tb_logs/*

# --- Local dev commands (no Docker) ---
local-dumb:
	python -m src.dumb_controller

local-train:
	python -m src.smart_controller --train

local-eval:
	python -m src.smart_controller --evaluate

local-demo:
	python -m src.smart_controller --demo
