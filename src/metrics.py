"""
Prometheus metrics instrumentation for traffic simulation.

Exposes 4 custom Gauges scraped by Prometheus at /metrics on port 8000.
"""

from prometheus_client import Gauge, start_http_server

# ---------------------------------------------------------------------------
# Custom Gauges
# ---------------------------------------------------------------------------
traffic_queue_length = Gauge(
    "traffic_queue_length",
    "Live count of waiting (queued) vehicles in the network",
)

traffic_wait_time_total = Gauge(
    "traffic_wait_time_total",
    "Total cumulative waiting time of all vehicles (seconds)",
)

agent_reward_total = Gauge(
    "agent_reward_total",
    "Composite reward for the current simulation step",
)

agent_switch_penalty = Gauge(
    "agent_switch_penalty",
    "Switch penalty applied at current step (0 or -alpha)",
)


def start_metrics_server(port: int = 8000) -> None:
    """Start the Prometheus HTTP endpoint in a background thread."""
    start_http_server(port)
    print(f"[metrics] Prometheus endpoint serving on :{port}/metrics")


def update(info: dict) -> None:
    """Update all gauges from a step-info dict produced by the env."""
    traffic_queue_length.set(info.get("queue_length", 0))
    traffic_wait_time_total.set(info.get("wait_time_total", 0))
    agent_reward_total.set(info.get("reward", 0))
    agent_switch_penalty.set(info.get("switch_penalty", 0))
