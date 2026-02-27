# Environment registry for all SUMO environments.
# New stages add entries here; agents use make_env() to instantiate.

from src.envs.single_intersection import SumoSingleIntersectionEnv

ENV_REGISTRY = {
    "single_intersection": SumoSingleIntersectionEnv,
}


def make_env(name: str, **kwargs):
    """Create an environment by name from the registry."""
    if name not in ENV_REGISTRY:
        available = ", ".join(ENV_REGISTRY.keys())
        raise ValueError(f"Unknown env '{name}'. Available: {available}")
    return ENV_REGISTRY[name](**kwargs)
