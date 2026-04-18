"""Pluggable batch-rollout backends (cpu threads, mujoco_warp on CUDA)."""

from src.envs.rollout_backends.base import RolloutBackend
from src.envs.rollout_backends.cpu import CPURolloutBackend


def make_backend(name: str, model, frame_skip: int, **kwargs) -> RolloutBackend:
    if name == "cpu":
        return CPURolloutBackend(model, frame_skip=frame_skip, **kwargs)
    if name == "warp":
        from src.envs.rollout_backends.warp_backend import WarpRolloutBackend
        return WarpRolloutBackend(model, frame_skip=frame_skip, **kwargs)
    raise ValueError(f"unknown rollout backend {name!r} (want 'cpu' or 'warp')")


__all__ = ["RolloutBackend", "CPURolloutBackend", "make_backend"]
