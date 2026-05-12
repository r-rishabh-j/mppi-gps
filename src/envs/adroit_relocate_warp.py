"""Warp-backed AdroitRelocate. Same cost/obs as ``AdroitRelocate``;
``batch_rollout`` runs on GPU (see ``WarpRolloutMixin``).

``nworld`` is fixed for the env's lifetime; pass ``nworld=K`` to match
MPPI batch size. Mocap target is broadcast from the CPU ``self.data``
on every rollout (set by ``env.reset()``).
"""
from __future__ import annotations

import numpy as np

from src.envs.adroit_relocate import AdroitRelocate
from src.envs.warp_rollout import WarpRolloutMixin


class AdroitRelocateWarp(WarpRolloutMixin, AdroitRelocate):
    def __init__(
        self,
        nworld: int,
        frame_skip: int = 5,
        njmax: int = 256,
        nconmax: int = 96,
        **kwargs,
    ) -> None:
        """Per-world njmax/nconmax sized for Adroit grasping. Bump on
        "nefc overflow" with ``int(N * 1.3)``.
        """
        super().__init__(frame_skip=frame_skip, **kwargs)
        self._init_warp(nworld=nworld, njmax=njmax, nconmax=nconmax)

    def batch_rollout(
        self,
        initial_state: np.ndarray,
        action_sequences: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._batch_rollout_warp(initial_state, action_sequences)
