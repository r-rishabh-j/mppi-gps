"""acrobot swing up env"""

import mujoco
import numpy as np
from pathlib import Path
from jaxtyping import Array, Float
from numpy import ndarray

from src.envs.mujoco_env import MuJoCoEnv

_XML = str(Path(__file__).resolve().parents[2] / "assets" / "acrobot.xml")

_TARGET = np.array([0.0, 0.0, 4.0])

class Acrobot(MuJoCoEnv):
    def __init__(self, frame_skip: int = 1, **kwargs) -> None:
        super().__init__(model_path=_XML, frame_skip=frame_skip, **kwargs)
        self._nq = self.model.nq #2
        self._nv = self.model.nv #2

        self._w_terminal = 0.0


    def reset(
            self,
            state: Float[ndarray, "state_dim"] | None = None,
    ) -> Float[ndarray, "4"]:
        if state is not None:
            return super().reset(state = state)

        obs = super().reset()
        self.data.qpos[:] += np.random.uniform(-np.pi, np.pi, size=self._nq)
        self.data.qvel[:] = np.random.normal(0.0, 0.05, size=self._nv) * 0.0 
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs()

    def running_cost(
            self,
            states: Float[Array, "K H nstate"],
            actions: Float[Array, "K H nu"],
            sensordata: Float[Array, "K H nsensor"] | None = None,
    ) -> Float[Array, "K H"]:
        tip_pos = sensordata[:, :, :3]
        target_radius = 0.20
        dist = np.linalg.norm(tip_pos - _TARGET, axis=-1)
        return (dist > target_radius).astype(dtype=np.float32) + (dist**2 * 0.05)

    def terminal_cost(
            self,
            states: Float[Array, "K nstate"],
            sensordata: Float[Array, "K nsensor"] | None = None,
    ) -> Float[Array, "K"]:
        tip_pos = sensordata[:, :3]
        dist = np.linalg.norm(tip_pos - _TARGET, axis=-1)
        qvel = self.state_qvel(states)
        vel_cost = np.sum(qvel**2, axis=-1)
        return self._w_terminal * (dist + 5.0 * vel_cost)

    def _get_obs(self) -> Float[ndarray, "4"]:
        return np.concatenate([self.data.qpos, self.data.qvel])
