"""Environment factory with CPU/GPU backend selection."""

from src.envs.base import BaseEnv


_REGISTRY = {
    "cpu": {
        "acrobot": ("src.envs.acrobot", "Acrobot"),
        "half_cheetah": ("src.envs.half_cheetah", "HalfCheetah"),
        "hopper": ("src.envs.hopper", "Hopper"),
        "point_mass": ("src.envs.point_mass", "PointMass"),
    },
    "gpu": {
        "acrobot": ("src.envs.mjx_acrobot", "MJXAcrobot"),
        "half_cheetah": ("src.envs.mjx_half_cheetah", "MJXHalfCheetah"),
        "hopper": ("src.envs.mjx_hopper", "MJXHopper"),
        "point_mass": ("src.envs.mjx_point_mass", "MJXPointMass"),
    },
}


def make_env(name: str, backend: str = "cpu", **kwargs) -> BaseEnv:
    """Create an environment by name and backend.

    Args:
        name:    Environment name (e.g. "acrobot", "hopper").
        backend: "cpu" (MuJoCo C API + threads) or "gpu" (MJX + JAX).
        **kwargs: Forwarded to the environment constructor.

    Returns:
        An environment instance implementing BaseEnv.
    """
    if backend not in _REGISTRY:
        raise ValueError(f"Unknown backend {backend!r}. Choose 'cpu' or 'gpu'.")
    if name not in _REGISTRY[backend]:
        raise ValueError(
            f"Unknown env {name!r} for backend {backend!r}. "
            f"Available: {list(_REGISTRY[backend].keys())}"
        )
    mod_path, cls_name = _REGISTRY[backend][name]
    import importlib
    mod = importlib.import_module(mod_path)
    return getattr(mod, cls_name)(**kwargs)
