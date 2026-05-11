"""Environment factory.

A note on the MJX path that used to live here
---------------------------------------------
We briefly added a ``use_mjx=True`` route + an ``_MJX_REGISTRY`` to run
batched rollouts via ``mujoco-mjx`` (JAX/XLA) so Apple Silicon machines
could fall back to something other than the CPU thread-pool. We removed
it after benchmarking on M5 Pro because the Metal backend is not viable
today for MPPI-scale rollouts:

  * ``jax-metal 0.1.1`` (Apple, last updated mid-2024) requires
    ``jax/jaxlib==0.4.34``. Newer JAX emits StableHLO bytecode the
    Metal plugin can't parse — even ``jit(sin(x)+cos(x))`` fails.
  * After downgrading, ``mjx.put_model`` works on Metal only with
    ``impl='jax'`` AND ``opt.jacobian=SPARSE`` AND ``opt.solver=CG``
    (dense+Newton uses ``cho_factor`` which jax-metal lacks).
  * Even with all of the above, ``jax.lax.scan`` containing
    ``mjx.step`` hangs indefinitely on Metal compile (>5 min, no CPU
    activity). Fully unrolled H also hangs. Single ``jit(mjx.step)``
    works at ~38 ms/step single world — slower than the existing
    8-thread ``mujoco.rollout`` path it would replace.
  * The CPU-JAX fallback (with the same downgrade) compiles K=32/H=16
    in ~3.5 min and runs at ~5 s/rollout — orders of magnitude slower
    than the native CPU thread-pool path.

If jax-metal ever grows scan support, or if mujoco-mjx ports to MLX,
this is the place to re-add the route. Until then, Apple Silicon users
should stick with the default CPU rollout (or run the Warp variant on
a CUDA box for actual GPU acceleration).
"""

from src.envs.base import BaseEnv


_REGISTRY = {
    "acrobot": ("src.envs.acrobot", "Acrobot"),
    "adroit_pen": ("src.envs.adroit_pen", "AdroitPen"),
    "adroit_relocate": ("src.envs.adroit_relocate", "AdroitRelocate"),
    "hopper": ("src.envs.hopper", "Hopper"),
    "point_mass": ("src.envs.point_mass", "PointMass"),
    "ur5_push": ("src.envs.ur5_push", "UR5Push"),
}

# Warp/mujoco_warp GPU rollout variants. Selected via
# ``make_env(name, use_warp=True, nworld=K)`` where K matches MPPI's
# batch size. Each entry must subclass the corresponding CPU env so
# `cost`, `obs`, sensors, and bounds are inherited unchanged.
_WARP_REGISTRY = {
    "adroit_relocate": ("src.envs.adroit_relocate_warp", "AdroitRelocateWarp"),
    "hopper": ("src.envs.hopper_warp", "HopperWarp"),
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
