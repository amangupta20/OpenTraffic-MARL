#!/bin/bash
# Entrypoint for the traffic container.
# MODE env var selects the behavior.
set -e

MODE="${MODE:-dumb}"

case "$MODE" in
  dumb)
    echo "[entrypoint] Running static-timer controller"
    exec python3 -m src.baselines.static_timer --port 8000 "$@"
    ;;
  train)
    echo "[entrypoint] Training PPO agent"
    exec python3 -m src.agents.ppo --train --port 8000 "$@"
    ;;
  evaluate)
    echo "[entrypoint] Evaluating saved PPO model"
    exec python3 -m src.agents.ppo --evaluate --port 8000 "$@"
    ;;
  compare)
    echo "[entrypoint] Running offline comparison (static vs PPO)"
    exec python3 -m src.evaluation.compare "$@"
    ;;
  demo)
    echo "[entrypoint] Starting visual demo (sumo-gui via noVNC)"

    # Start Xvfb virtual display
    Xvfb :99 -screen 0 1280x720x24 &
    export DISPLAY=:99
    sleep 1

    # Start x11vnc
    x11vnc -display :99 -forever -nopw -quiet &
    sleep 1

    # Start noVNC (websockify)
    websockify --web=/usr/share/novnc 6080 localhost:5900 &
    sleep 1

    echo "[entrypoint] noVNC available at http://localhost:6080"
    exec python3 -m src.agents.ppo --demo --port 8000 "$@"
    ;;
  wandb-login)
    echo "[entrypoint] Logging into Weights & Biases"
    exec python3 -m wandb login "$@"
    ;;
  grid-demo)
    echo "[entrypoint] Starting visual demo of 4 cloned PPO agents on 2×2 grid"

    Xvfb :99 -screen 0 1280x720x24 &
    export DISPLAY=:99
    sleep 1

    x11vnc -display :99 -forever -nopw -quiet &
    sleep 1

    websockify --web=/usr/share/novnc 6080 localhost:5900 &
    sleep 1

    echo "[entrypoint] noVNC available at http://localhost:6080"
    exec python3 -m src.agents.independent_ppo --demo --port 8000 "$@"
    ;;
  grid-eval)
    echo "[entrypoint] Evaluating cloned PPO agents on 2×2 grid"
    exec python3 -m src.agents.independent_ppo --evaluate "$@"
    ;;
  grid-static)
    echo "[entrypoint] Running static-timer baseline on 2×2 grid"
    exec python3 -m src.agents.independent_ppo --static "$@"
    ;;
  grid-compare)
    echo "[entrypoint] Comparing static vs cloned PPO on 2×2 grid"
    exec python3 -m src.agents.independent_ppo --compare "$@"
    ;;
  *)
    echo "[entrypoint] Unknown MODE=$MODE"
    echo "  Available: dumb|train|evaluate|compare|demo|wandb-login"
    echo "             grid-eval|grid-static|grid-compare"
    exit 1
    ;;
esac
