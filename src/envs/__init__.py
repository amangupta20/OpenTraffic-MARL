# Environment registry for all SUMO environments.
# New stages add entries here; agents use make_env() to instantiate.

from src.envs.single_intersection import SumoSingleIntersectionEnv
from src.envs.grid_2x2 import SumoGrid2x2Env
from src.envs.bangalore_corridor import SumoBangaloreCorridor

ENV_REGISTRY = {
    "single_intersection": SumoSingleIntersectionEnv,
    "grid_2x2": SumoGrid2x2Env,
    "bangalore_corridor": SumoBangaloreCorridor,
}


def make_env(name: str, **kwargs):
    """Create an environment by name from the registry."""
    if name not in ENV_REGISTRY:
        available = ", ".join(ENV_REGISTRY.keys())
        raise ValueError(f"Unknown env '{name}'. Available: {available}")
    return ENV_REGISTRY[name](**kwargs)
