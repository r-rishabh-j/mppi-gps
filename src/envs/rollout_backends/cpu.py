"""CPU batch rollout via mujoco.rollout + thread pool.

Lifted from the original MuJoCoEnv.batch_rollout; behavior is unchanged.
"""

from __future__ import annotations
import os
from typing import TYPE_CHECKING
import numpy as np
import mujoco
from mujoco import rollout

if TYPE_CHECKING:
    from src.envs.mujoco_env import MuJoCoEnv


class CPURolloutBackend:
    def __init__(self, model: mujoco.MjModel, frame_skip: int = 1, nthread: int | None = None):
        self._model = model
        self._frame_skip = frame_skip
        self._nthread = nthread or os.cpu_count()
        self._data_pool = [mujoco.MjData(model) for _ in range(self._nthread)]
        self._rollout_ctx = rollout.Rollout(nthread=self._nthread)

    def rollout(
        self,
        env: "MuJoCoEnv",
        initial_state: np.ndarray,
        action_sequences: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        K, H, _ = action_sequences.shape

        actions_expanded = np.repeat(action_sequences, self._frame_skip, axis=1)
        states_full, sensordata_full = self._rollout_ctx.rollout(
            self._model,
            self._data_pool,
            initial_state,
            actions_expanded,
        )

        states = states_full[:, ::self._frame_skip, :]
        sensordata = sensordata_full[:, ::self._frame_skip, :]

        c = env.running_cost(states, action_sequences, sensordata)       # (K, H)
        tc = env.terminal_cost(states[:, -1, :], sensordata[:, -1, :])   # (K,)
        costs = c.sum(axis=1) + tc
        return states, costs, sensordata

    def close(self) -> None:
        self._rollout_ctx.__exit__(None, None, None)
