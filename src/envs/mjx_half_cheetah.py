"""MJX-backed HalfCheetah environment."""

import numpy as np
import jax.numpy as jnp
from pathlib import Path

from src.envs.mjx_env import MJXEnv
from src.envs.costs_jax import half_cheetah_running_cost, half_cheetah_terminal_cost

_XML = str(Path(__file__).resolve().parents[2] / "assets" / "half_cheetah.xml")


class MJXHalfCheetah(MJXEnv):
    def __init__(self, ctrl_cost_weight: float = 0.0, frame_skip: int = 5, **kwargs):
        nq, nv = 9, 9  # known from the model
        super().__init__(
            model_path=_XML,
            frame_skip=frame_skip,
            running_cost_fn=half_cheetah_running_cost,
            terminal_cost_fn=half_cheetah_terminal_cost,
            cost_kwargs=dict(
                running=dict(nq=nq, nv=nv, w_vel=1.0, w_pitch=0.5, w_controls=0.001),
                terminal=dict(),
            ),
            **kwargs,
        )
        self._nq = self.model.nq
        self._nv = self.model.nv

    def _get_obs(self):
        return np.concatenate([self.data.qpos[1:], self.data.qvel])

    def state_to_obs(self, states):
        qpos = self.state_qpos(states)[..., 1:]  # skip root x
        qvel = self.state_qvel(states)
        if isinstance(states, jnp.ndarray):
            return jnp.concatenate([qpos, qvel], axis=-1)
        return np.concatenate([qpos, qvel], axis=-1)

    @property
    def obs_dim(self):
        return (self._nq - 1) + self._nv
