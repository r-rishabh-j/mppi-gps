"""2D point-mass goal-reaching task for MPPI sanity checks."""

import mujoco
import numpy as np
from pathlib import Path
from jaxtyping import Array, Float
from numpy import ndarray

from src.envs.mujoco_env import MuJoCoEnv

_XML = str(Path(__file__).resolve().parents[2] / "assets" / "point_mass.xml")
_GOAL = np.array([0.0, 0.0])


class PointMass(MuJoCoEnv):
    def __init__(self, frame_skip: int = 1, ctrl_cost_weight: float = 0.01, **kwargs) -> None:
        super().__init__(model_path=_XML, frame_skip=frame_skip, **kwargs)
        self._ctrl_w = ctrl_cost_weight
        self._nq = self.model.nq
        self._nv = self.model.nv

    def reset(
        self,
        state: Float[ndarray, "state_dim"] | None = None,
    ) -> Float[ndarray, "4"]:
        if state is not None:
            return super().reset(state=state)

        obs = super().reset()
        self.data.qpos[:] = np.random.uniform(-0.25, 0.25, size=self._nq)
        self.data.qvel[:] = np.random.normal(0.0, 0.1, size=self._nv)
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs()

    def running_cost(
        self,
        states: Float[Array, "K H nstate"],
        actions: Float[Array, "K H nu"],
        sensordata: Float[Array, "K H nsensor"] | None = None,
    ) -> Float[Array, "K H"]:
        qpos = self.state_qpos(states)
        qvel = self.state_qvel(states)
        pos_err = qpos - _GOAL[None, None, :]
        pos_cost = np.sum(pos_err ** 2, axis=-1)
        vel_cost = 5.0 * np.sum(qvel * pos_err, axis=-1)
        return pos_cost + vel_cost

    def terminal_cost(
        self,
        states: Float[Array, "K nstate"],
        sensordata: Float[Array, "K nsensor"] | None = None,
    ) -> Float[Array, "K"]:
        qpos = self.state_qpos(states)
        qvel = self.state_qvel(states)
        pos_err = qpos - _GOAL[None, :]
        return 0.0 * np.sum(pos_err ** 2, axis=-1) + 0.5 * np.sum(qvel ** 2, axis=-1)

    def _get_obs(self) -> Float[ndarray, "4"]:
        return np.concatenate([self.data.qpos, self.data.qvel])

    def state_to_obs(
        self,
        states: np.ndarray,
        sensordata: np.ndarray | None = None,
    ) -> np.ndarray:
        qpos = self.state_qpos(states)
        qvel = self.state_qvel(states)
        return np.concatenate([qpos, qvel], axis=-1)

    @property
    def obs_dim(self) -> int:
        return self._nq + self._nv
