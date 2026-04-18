"""Rollout backend protocol.

A backend runs K action sequences of length H through the physics and returns
per-step states, per-trajectory costs, and per-step sensordata — all as numpy
arrays at the boundary. CPU backend uses mujoco.rollout + a thread pool;
Warp backend runs mujoco_warp on CUDA and calls torch cost functions before
copying back to host.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Protocol
import numpy as np

if TYPE_CHECKING:
    from src.envs.mujoco_env import MuJoCoEnv


class RolloutBackend(Protocol):
    def rollout(
        self,
        env: "MuJoCoEnv",
        initial_state: np.ndarray,          # (nstate,)
        action_sequences: np.ndarray,       # (K, H, nu)
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (states (K,H,nstate), costs (K,), sensordata (K,H,nsensor))."""
        ...

    def close(self) -> None: ...
