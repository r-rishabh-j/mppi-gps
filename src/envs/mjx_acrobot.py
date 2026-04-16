"""MJX-backed Acrobot environment.

Uses acrobot_mjx.xml (energy flag removed for MJX compatibility).
Tip position comes from ``site_xpos`` instead of ``sensordata``.
"""

import mujoco
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path
from typing import Any

from src.envs.mjx_env import MJXEnv
from src.envs.costs_jax import acrobot_running_cost, acrobot_terminal_cost

_XML = str(Path(__file__).resolve().parents[2] / "assets" / "acrobot_mjx.xml")
_TARGET = np.array([0.0, 0.0, 4.0])


class MJXAcrobot(MJXEnv):
    def __init__(self, frame_skip: int = 1, **kwargs):
        super().__init__(
            model_path=_XML,
            frame_skip=frame_skip,
            running_cost_fn=acrobot_running_cost,
            terminal_cost_fn=acrobot_terminal_cost,
            cost_kwargs=dict(
                running=dict(target=_TARGET, margin=4.0, nq=2, nv=2),
                terminal=dict(target=_TARGET, w_terminal=0.0, nq=2, nv=2),
            ),
            **kwargs,
        )
        self._nq = self.model.nq
        self._nv = self.model.nv
        # Find the tip site index for site_xpos extraction
        self._tip_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "tip"
        )

    def _extract_sensor_or_site(self, mx_d: Any) -> jax.Array:
        """Return tip site position (3,) instead of sensordata.

        The CPU acrobot uses a ``framepos`` sensor for the tip.  MJX
        computes ``site_xpos`` after each step, which gives the same
        3-D position.
        """
        return mx_d.site_xpos[self._tip_site_id]  # (3,)

    def reset(self, state=None):
        if state is not None:
            return super().reset(state=state)
        obs = super().reset()
        self.data.qpos[:] = np.random.uniform(-np.pi, np.pi, size=self._nq)
        self.data.qvel[:] = np.random.normal(0.0, 0.05, size=self._nv) * 0.0
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel])

    def state_to_obs(self, states):
        qpos = self.state_qpos(states)
        qvel = self.state_qvel(states)
        if isinstance(states, jnp.ndarray):
            return jnp.concatenate([qpos, qvel], axis=-1)
        return np.concatenate([qpos, qvel], axis=-1)

    @property
    def obs_dim(self):
        return self._nq + self._nv
