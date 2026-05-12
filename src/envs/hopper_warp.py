"""Warp-backed Hopper. Same cost/obs as ``Hopper``; only ``batch_rollout``
runs on GPU (see ``WarpRolloutMixin``).

``nworld`` is fixed for the env's lifetime; pass ``nworld=K`` to match
MPPI batch size (or ``N*K`` under WarpMPPIGPS).
"""
from __future__ import annotations

import numpy as np

from src.envs.hopper import Hopper
from src.envs.warp_rollout import WarpRolloutMixin


class HopperWarp(WarpRolloutMixin, Hopper):
    def __init__(
        self,
        nworld: int,
        frame_skip: int = 4,
        njmax: int = 128,
        nconmax: int = 48,
        **kwargs,
    ) -> None:
        """Per-world njmax/nconmax sized for Hopper's contact load. Bump
        on "nefc overflow" with ``int(N * 1.3)``.
        """
        super().__init__(frame_skip=frame_skip, **kwargs)
        self._init_warp(nworld=nworld, njmax=njmax, nconmax=nconmax)

    def batch_rollout(
        self,
        initial_state: np.ndarray,
        action_sequences: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Filter to the public sensor slice (matches CPU path).
        states, costs, sensordata = self._batch_rollout_warp(
            initial_state, action_sequences,
        )
        public_sensordata = self._public_sensordata(sensordata)
        return states, costs, public_sensordata
