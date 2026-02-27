#!/bin/bash
# Entrypoint for the traffic-agent container.
# MODE env var selects the behavior.
set -e

MODE="${MODE:-dumb}"

case "$MODE" in
  dumb)
    echo "[entrypoint] Running static-timer controller"
    exec python -m src.baselines.static_timer --port 8000 "$@"
    ;;
  train)
    echo "[entrypoint] Training PPO agent"
    exec python -m src.agents.ppo --train --port 8000 "$@"
    ;;
  evaluate)
    echo "[entrypoint] Evaluating saved PPO model"
    exec python -m src.agents.ppo --evaluate --port 8000 "$@"
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
    /usr/share/novnc/utils/novnc_proxy --vnc localhost:5900 --listen 6080 &
    sleep 1

    echo "[entrypoint] noVNC available at http://localhost:6080"
    exec python -m src.agents.ppo --demo --port 8000 "$@"
    ;;
  *)
    echo "[entrypoint] Unknown MODE=$MODE (use: dumb|train|evaluate|demo)"
    exit 1
    ;;
esac
