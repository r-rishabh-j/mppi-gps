"""Environment factory."""

from src.envs.base import BaseEnv


_REGISTRY = {
    "acrobot": ("src.envs.acrobot", "Acrobot"),
    "half_cheetah": ("src.envs.half_cheetah", "HalfCheetah"),
    "hopper": ("src.envs.hopper", "Hopper"),
    "point_mass": ("src.envs.point_mass", "PointMass"),
}


def make_env(name: str, **kwargs) -> BaseEnv:
    """Create an environment by name."""
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown env {name!r}. Available: {list(_REGISTRY.keys())}"
        )
    mod_path, cls_name = _REGISTRY[name]
    import importlib
    mod = importlib.import_module(mod_path)
    return getattr(mod, cls_name)(**kwargs)
