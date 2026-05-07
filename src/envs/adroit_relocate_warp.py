"""Warp-backed AdroitRelocate. Same cost / observation / sensor layout as
the CPU env; only `batch_rollout` changes — it dispatches to GPU via
`mujoco_warp` (see `src/envs/warp_rollout.py`).

Construction:
    AdroitRelocateWarp(nworld=K)        # K must equal MPPIConfig.K

`nworld` is fixed for the env's lifetime. Re-instantiate to change MPPI's
batch size. See `WarpRolloutMixin` for full constraint list.

Adroit specifics handled here:
- ``nmocap == 1`` (the relocate target is a mocap body). The mixin
  detects this via ``self._has_mocap`` and broadcasts ``data.mocap_pos``
  / ``data.mocap_quat`` to all `nworld` GPU worlds on each rollout. The
  CPU ``self.data`` (set by ``env.reset()``) is the source of truth for
  the per-episode target; rollouts inherit whatever target was last
  written there.
- ``frame_skip == 5`` is handled by the mixin's inner loop (5 mjw.step
  calls per outer control step, matching the CPU path's downsampling).
- All Adroit framepos / framequat / touch sensors run inside ``mjw.step``
  exactly as on CPU, populating ``wd.sensordata`` (nsensordata = 89).
  ``state_to_obs`` gets the same sensordata it does on the CPU path,
  so the DAPG observation construction is unchanged.

What this env does NOT change:
- ``running_cost``, ``terminal_cost``, ``state_to_obs``, ``noise_scale``,
  ``action_bounds``, all sensor-slice plumbing — inherited verbatim from
  ``AdroitRelocate``.
- ``reset(state=None)`` and the per-episode target randomization — these
  run on the CPU model since they're called once per episode and the
  GPU side gets re-seeded from CPU state on every ``batch_rollout`` call
  anyway.
"""
from __future__ import annotations

import numpy as np

from src.envs.adroit_relocate import AdroitRelocate
from src.envs.warp_rollout import WarpRolloutMixin


class AdroitRelocateWarp(WarpRolloutMixin, AdroitRelocate):
    def __init__(self, nworld: int, frame_skip: int = 5, **kwargs) -> None:
        super().__init__(frame_skip=frame_skip, **kwargs)
        self._init_warp(nworld=nworld)

    def batch_rollout(
        self,
        initial_state: np.ndarray,
        action_sequences: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._batch_rollout_warp(initial_state, action_sequences)
