"""MJX-backed PointMass environment."""

import mujoco
import numpy as np
import jax.numpy as jnp
from pathlib import Path

from src.envs.mjx_env import MJXEnv
from src.envs.costs_jax import point_mass_running_cost, point_mass_terminal_cost

_XML = str(Path(__file__).resolve().parents[2] / "assets" / "point_mass.xml")
_GOAL = np.array([0.0, 0.0])


class MJXPointMass(MJXEnv):
    def __init__(self, frame_skip: int = 1, ctrl_cost_weight: float = 0.01, **kwargs):
        super().__init__(
            model_path=_XML,
            frame_skip=frame_skip,
            running_cost_fn=point_mass_running_cost,
            terminal_cost_fn=point_mass_terminal_cost,
            cost_kwargs=dict(
                running=dict(goal=_GOAL, nq=2, nv=2),
                terminal=dict(goal=_GOAL, nq=2, nv=2),
            ),
            **kwargs,
        )
        self._nq = self.model.nq
        self._nv = self.model.nv

    def reset(self, state=None):
        if state is not None:
            return super().reset(state=state)
        obs = super().reset()
        self.data.qpos[:] = np.random.uniform(-0.25, 0.25, size=self._nq)
        self.data.qvel[:] = np.random.normal(0.0, 0.1, size=self._nv)
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel])

    def state_to_obs(self, states):
        qpos = self.state_qpos(states)
        qvel = self.state_qvel(states)
        # Works on both numpy and jax arrays
        if isinstance(states, jnp.ndarray):
            return jnp.concatenate([qpos, qvel], axis=-1)
        return np.concatenate([qpos, qvel], axis=-1)

    @property
    def obs_dim(self):
        return self._nq + self._nv
