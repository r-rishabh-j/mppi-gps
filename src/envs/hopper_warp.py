"""Warp-backed Hopper. Same cost / observation / sensor layout as the CPU
``Hopper`` env; only ``batch_rollout`` is replaced with a graph-captured
``mujoco_warp`` rollout.

Construction:
    HopperWarp(nworld=K)        # K must equal MPPIConfig.K (single-condition)
    # or, under WarpMPPIGPS:
    HopperWarp(nworld=N*K)      # N=num_conditions, K per-condition samples

`nworld` is fixed for the env's lifetime — re-instantiate to change. See
``WarpRolloutMixin`` for the full constraint list.

Hopper specifics handled here:
- ``nmocap == 0`` — no per-rollout mocap broadcast; mixin's mocap branch
  is a silent no-op.
- ``frame_skip == 4`` (Hopper default) is handled by the mixin's inner
  ``frame_skip × mjw.step`` loop, matching the CPU path's downsampling.
- ``Hopper.batch_rollout`` overrides the base CPU ``MuJoCoEnv.batch_rollout``
  to filter the returned sensor array to ``_public_sensordata(...)`` so
  external callers don't see the v2 reward-only sensors. The warp variant
  preserves that contract: the mixin's full sensordata is filtered through
  the same helper before returning. v2 cost is still computed against the
  full (private) sensor array inside the mixin's ``_batch_rollout_warp``,
  so cost values are unchanged.

What this env does NOT change:
- ``running_cost``, ``terminal_cost``, ``state_to_obs``, ``noise_scale``,
  ``action_bounds``, healthy-state termination — inherited verbatim from
  ``Hopper``.
- ``reset(state=None)`` — randomized start pose perturbation runs on the
  CPU model. The GPU side is re-seeded from CPU state on every
  ``batch_rollout`` call so per-rollout reproducibility is unaffected.
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
        """``njmax`` / ``nconmax`` are **per-world** constraint / contact
        buffer budgets for the GPU rollout (see WarpRolloutMixin._init_warp).
        Defaults are sized for Hopper's contact load (foot-ground +
        occasional self-contact); bump them if you hit "nefc overflow -
        please increase njmax to N" with ``int(N * 1.3)`` to leave headroom.
        """
        super().__init__(frame_skip=frame_skip, **kwargs)
        self._init_warp(nworld=nworld, njmax=njmax, nconmax=nconmax)

    def batch_rollout(
        self,
        initial_state: np.ndarray,
        action_sequences: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Mixin returns the full sensordata array (private, used for v2
        # cost inside ``running_cost``). Filter through the same
        # ``_public_sensordata`` helper the CPU path uses so external
        # callers see the agreed-upon (here: empty) public sensor slice.
        states, costs, sensordata = self._batch_rollout_warp(
            initial_state, action_sequences,
        )
        public_sensordata = self._public_sensordata(sensordata)
        return states, costs, public_sensordata
