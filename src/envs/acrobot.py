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
        self.data.qpos[:] = np.random.uniform(-np.pi, np.pi, size=self._nq)
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
        target_radius = 0.20 # hardset based on xml
        dist = np.linalg.norm(tip_pos - _TARGET, axis=-1)
        margin = 4.0 # where the decay hits a specific value
        # adding a gaussian tolerance 
        # _value_at_margin = 0.1 
        # scale = np.sqrt(-2.0 * np.log(_value_at_margin)) # how steep do you want the decay to be
        # d_beyond = np.maximum(dist - target_radius, 0.0)
        # reward = np.where(
        #     dist <= target_radius, 
        #     1.0, 
        #     np.exp(-0.5 * (d_beyond * scale / margin) ** 2), 
        # )
        # gaussian_cost = 1.0 - reward
        # tip_z = sensordata[:, :, 2] 

        return (dist / margin) + 2 *(4 - np.linalg.norm(sensordata, axis=2))

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
