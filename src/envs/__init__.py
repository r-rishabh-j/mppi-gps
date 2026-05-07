"""Environment factory."""

from src.envs.base import BaseEnv


_REGISTRY = {
    "acrobot": ("src.envs.acrobot", "Acrobot"),
    "adroit_pen": ("src.envs.adroit_pen", "AdroitPen"),
    "adroit_relocate": ("src.envs.adroit_relocate", "AdroitRelocate"),
    "half_cheetah": ("src.envs.half_cheetah", "HalfCheetah"),
    "hopper": ("src.envs.hopper", "Hopper"),
    "point_mass": ("src.envs.point_mass", "PointMass"),
}

# Warp/mujoco_warp GPU rollout variants. Selected via
# ``make_env(name, use_warp=True, nworld=K)`` where K matches MPPI's
# batch size. Each entry must subclass the corresponding CPU env so
# `cost`, `obs`, sensors, and bounds are inherited unchanged.
_WARP_REGISTRY = {
    "adroit_relocate": ("src.envs.adroit_relocate_warp", "AdroitRelocateWarp"),
}


def make_env(name: str, use_warp: bool = False, **kwargs) -> BaseEnv:
    """Create an environment by name.

    ``use_warp=True`` opts into a Warp/``mujoco_warp`` GPU rollout variant
    when one exists for ``name``. Caller MUST pass ``nworld=K`` (MPPI batch
    size) in **kwargs**; ``nworld`` is fixed for the env's lifetime.

    See ``src/envs/warp_rollout.py`` for constraints (na==0, CUDA-only
    graph replay, fixed K).
    """
    registry = _WARP_REGISTRY if use_warp else _REGISTRY
    if name not in registry:
        if use_warp:
            raise ValueError(
                f"No Warp variant for env {name!r}. "
                f"Available Warp envs: {list(_WARP_REGISTRY.keys())}. "
                "Run without --warp to use the CPU rollout path."
            )
        raise ValueError(
            f"Unknown env {name!r}. Available: {list(_REGISTRY.keys())}"
        )
    mod_path, cls_name = registry[name]
    import importlib
    mod = importlib.import_module(mod_path)
    return getattr(mod, cls_name)(**kwargs)
