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
    def __init__(self, frame_skip: int = 2, **kwargs) -> None:
        super().__init__(model_path=_XML, frame_skip=frame_skip, **kwargs)
        self._nq = self.model.nq #2
        self._nv = self.model.nv #2

        self._w_terminal = 5.0
        # --- previous reward-shaping params (commented for reference) ---
        # self._energy_target = 4.0
        # self._energy_weight = 0.08
        # self._near_top_radius = 0.5
        # self._vel_weight = 0.002
        # self._vel_weight_near_top = 0.05
        # self._ctrl_weight = 0.001


    def reset(
            self,
            state: Float[ndarray, "state_dim"] | None = None,
    ) -> Float[ndarray, "6"]:
        if state is not None:
            return super().reset(state = state)

        obs = super().reset()
        self.data.qpos[:] = np.random.uniform(-np.pi, np.pi, size=self._nq)
        # self.data.qvel[:] = np.random.normal(0.0, 0.05, size=self._nv) * 0.0
        self.data.qvel[:] = np.random.normal(0.0, 3.0, size=self._nv)
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs()

    def running_cost(
            self,
            states: Float[Array, "K H nstate"],
            actions: Float[Array, "K H nu"],
            sensordata: Float[Array, "K H nsensor"] | None = None,
    ) -> Float[Array, "K H"]:
        tip_pos = sensordata[:, :, :3]
        dist = np.linalg.norm(tip_pos - _TARGET, axis=-1)       # (K, H)
        margin = 4.0
        return dist / margin + 2 *(4 - np.linalg.norm(sensordata, axis=2))

        # --- previous shaped reward (commented for reference) -------------
        # Combined linear far-field pull + Gaussian-tolerance near-top
        # attractor + one-sided pseudo-energy deficit + near-top velocity
        # damping. Worked on easy starts but trapped on low-energy starts
        # at tip_z≈2 where modest KE satisfies the energy target.
        #
        # tip_z = tip_pos[..., 2]
        # qvel = self.state_qvel(states)
        # vel_sq = np.sum(qvel ** 2, axis=-1)
        #
        # dist_cost = dist / 4.0
        #
        # target_radius = 0.20
        # value_at_margin = 0.1
        # scale = np.sqrt(-2.0 * np.log(value_at_margin))
        # d_beyond = np.maximum(dist - target_radius, 0.0)
        # reward = np.where(
        #     dist <= target_radius,
        #     1.0,
        #     np.exp(-0.5 * (d_beyond * scale / margin) ** 2),
        # )
        # tip_cost = dist_cost + (1.0 - reward)
        #
        # E = 0.5 * vel_sq + tip_z
        # energy_deficit = np.maximum(self._energy_target - E, 0.0)
        # energy_cost = self._energy_weight * energy_deficit ** 2
        #
        # near_top = (dist < self._near_top_radius).astype(vel_sq.dtype)
        # vel_cost = (self._vel_weight * vel_sq
        #             + self._vel_weight_near_top * near_top * vel_sq)
        #
        # ctrl_cost = self._ctrl_weight * np.sum(actions ** 2, axis=-1)
        #
        # return tip_cost + energy_cost + vel_cost + ctrl_cost

    def terminal_cost(
            self,
            states: Float[Array, "K nstate"],
            sensordata: Float[Array, "K nsensor"] | None = None,
    ) -> Float[Array, "K"]:
        tip_pos = sensordata[:, :3]
        dist = np.linalg.norm(tip_pos - _TARGET, axis=-1)
        qvel = self.state_qvel(states)
        vel_cost = np.sum(qvel ** 2, axis=-1)
        return self._w_terminal * (dist + 5.0 * vel_cost)

    def _get_obs(self) -> Float[ndarray, "6"]:
        # sin/cos encoding removes the ±π wrap-around discontinuity the MLP
        # would otherwise have to memorise around the upright state.
        q = self.data.qpos
        v = self.data.qvel
        return np.concatenate([np.sin(q), np.cos(q), v])

    def state_to_obs(self, states: np.ndarray) -> np.ndarray:
        qpos = self.state_qpos(states)
        qvel = self.state_qvel(states)
        return np.concatenate([np.sin(qpos), np.cos(qpos), qvel], axis=-1)

    @property
    def obs_dim(self) -> int:
        return 2 * self._nq + self._nv
