.PHONY: build up down train evaluate demo logs clean local-dumb local-train local-eval local-demo local-compare local-tb

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

# --- Local dev setup ---
SUMO_ENV = SUMO_HOME=/usr/share/sumo PYTHONPATH=.:/usr/share/sumo/tools CUDA_VISIBLE_DEVICES=
PY = $(SUMO_ENV) .venv/bin/python

venv:
	uv venv --system-site-packages --python /usr/bin/python3 .venv
	uv pip install gymnasium 'stable-baselines3[extra]' prometheus-client tensorboard numpy

# --- Local dev commands (no Docker) ---
local-dumb:
	$(PY) -m src.dumb_controller

local-train:
	$(PY) -m src.smart_controller --train

local-eval:
	$(PY) -m src.smart_controller --evaluate

local-demo:
	$(PY) -m src.smart_controller --demo

local-compare:
	$(PY) -m src.compare

local-tb:
	$(PY) -m tensorboard.main --logdir=tb_logs --host=0.0.0.0 --port=6006
