"""Environment factory."""

from src.envs.base import BaseEnv


_REGISTRY = {
    "acrobot": ("src.envs.acrobot", "Acrobot"),
    "adroit_pen": ("src.envs.adroit_pen", "AdroitPen"),
    "adroit_relocate": ("src.envs.adroit_relocate", "AdroitRelocate"),
    "hopper": ("src.envs.hopper", "Hopper"),
    "point_mass": ("src.envs.point_mass", "PointMass"),
    "ur5_push": ("src.envs.ur5_push", "UR5Push"),
}

# Warp/mujoco_warp GPU rollout variants (CUDA only).
_WARP_REGISTRY = {
    "adroit_relocate": ("src.envs.adroit_relocate_warp", "AdroitRelocateWarp"),
    "hopper": ("src.envs.hopper_warp", "HopperWarp"),
}


def make_env(name: str, use_warp: bool = False, **kwargs) -> BaseEnv:
    """Create an environment by name.

    ``use_warp=True`` selects the Warp GPU variant; caller MUST pass
    ``nworld=K`` matching MPPI's batch size (fixed for env lifetime).
    """
    registry = _WARP_REGISTRY if use_warp else _REGISTRY
    if name not in registry:
        if use_warp:
            raise ValueError(
                f"No Warp variant for env {name!r}. "
                f"Available: {list(_WARP_REGISTRY.keys())}."
            )
        raise ValueError(
            f"Unknown env {name!r}. Available: {list(_REGISTRY.keys())}"
        )
    mod_path, cls_name = registry[name]
    import importlib
    mod = importlib.import_module(mod_path)
    return getattr(mod, cls_name)(**kwargs)
