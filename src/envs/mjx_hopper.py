"""MJX-backed Hopper environment."""

import mujoco
import numpy as np
import jax.numpy as jnp
from pathlib import Path

from src.envs.mjx_env import MJXEnv
from src.envs.costs_jax import hopper_running_cost, hopper_terminal_cost

_XML = str(Path(__file__).resolve().parents[2] / "assets" / "hopper.xml")

_Z_MIN = 0.7
_ANGLE_MAX = 0.2


class MJXHopper(MJXEnv):
    def __init__(
        self,
        frame_skip: int = 4,
        ctrl_cost_weight: float = 0.001,
        forward_reward_weight: float = 1.0,
        healthy_reward: float = 1.0,
        **kwargs,
    ):
        nq, nv = 6, 6  # known from the model
        super().__init__(
            model_path=_XML,
            frame_skip=frame_skip,
            running_cost_fn=hopper_running_cost,
            terminal_cost_fn=hopper_terminal_cost,
            cost_kwargs=dict(
                running=dict(
                    nq=nq, nv=nv,
                    fwd_w=forward_reward_weight,
                    healthy_reward=healthy_reward,
                    ctrl_w=ctrl_cost_weight,
                    z_min=_Z_MIN,
                    angle_max=_ANGLE_MAX,
                ),
                terminal=dict(),
            ),
            **kwargs,
        )
        self._nq = self.model.nq
        self._nv = self.model.nv
        self._ctrl_w = ctrl_cost_weight
        self._fwd_w = forward_reward_weight
        self._healthy_reward = healthy_reward

    def reset(self, state=None):
        if state is not None:
            return super().reset(state=state)
        obs = super().reset()
        self.data.qpos[:] += np.random.uniform(-0.005, 0.005, size=self._nq)
        self.data.qvel[:] += np.random.uniform(-0.005, 0.005, size=self._nv)
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([
            self.data.qpos[1:],
            np.clip(self.data.qvel, -10.0, 10.0),
        ])

    def state_to_obs(self, states):
        qpos = self.state_qpos(states)[..., 1:]  # skip root x
        qvel = self.state_qvel(states)
        if isinstance(states, jnp.ndarray):
            qvel = jnp.clip(qvel, -10.0, 10.0)
            return jnp.concatenate([qpos, qvel], axis=-1)
        qvel = np.clip(qvel, -10.0, 10.0)
        return np.concatenate([qpos, qvel], axis=-1)

    def step(self, action):
        obs, cost, _, info = super().step(action)
        z = self.data.qpos[1]
        angle = self.data.qpos[2]
        state_ok = np.all(np.isfinite(obs)) and np.all(np.abs(obs) < 100.0)
        done = (z <= _Z_MIN) or (abs(angle) >= _ANGLE_MAX) or (not state_ok)
        return obs, cost, done, info

    @property
    def obs_dim(self):
        return (self._nq - 1) + self._nv
