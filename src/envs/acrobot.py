
"""acrobot swing up env"""

import mujoco
import numpy as np
from pathlib import Path
from jaxtyping import Array, Float
from numpy import ndarray

from src.envs.mujoco_env import MuJoCoEnv

_XML = str(Path(__file__).resolve().parents[2] / "assets" / "acrobot.xml")

class Acrobot(MuJoCoEnv):
    def __init__(self, frame_skip: int = 1, **kwargs) -> None:
        super().__init__(model_path=_XML, frame_skip=frame_skip, **kwargs)
        self._nq = self.model.nq #2
        self._nv = self.model.nv #2

        # this is upright, zero velocity and then the angles for the two joints
        self._x_goal: Float[ndarray, "4"] = np.array([0.0, 0.0, 0.0, 0.0])

        # FK constants (from acrobot.xml)
        self._shoulder_z = 2.0  # shoulder mount height
        self._l1 = 1.0  # upper arm length
        self._l2 = 1.0  # lower arm length
        self._target_z = 4.0  # tip z when upright

        # cost weights — sparse running cost, strong terminal
        self._w_ctrl = 0.01
        self._w_height = 0.5
        self._Q_terminal: Float[ndarray, "4"] = np.array([10.0, 10.0, 1.0, 1.0])
        self._w_terminal = 200.0

    def reset(
            self,
            state: Float[ndarray, "state_dim"] | None = None,
    ) -> Float[ndarray, "4"]:
        # should be able to establish a fresh simulator state or restore a full state that's been saved
        if state is not None:
            return super().reset(state = state)

        obs = super().reset()
        self.data.qpos[:] += np.random.normal(0.0, 0.05, size=self._nq)
        self.data.qvel[:] = np.random.normal(0.0, 0.05, size=self._nv)
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs()

    def _tip_dist(self, qpos: Float[Array, "... nq"]) -> Float[Array, "..."]:
        """Euclidean distance from tip to target via forward kinematics.
        Target is at (0, 0, 4). Planar system so y=0 always."""
        q0 = qpos[..., 0]
        q1 = qpos[..., 1]
        tip_x = self._l1 * np.sin(q0) + self._l2 * np.sin(q0 + q1)
        tip_z = self._shoulder_z + self._l1 * np.cos(q0) + self._l2 * np.cos(q0 + q1)
        return np.sqrt(tip_x**2 + (tip_z - self._target_z)**2)

    def running_cost(
            self,
          states: Float[Array, "K H nstate"],
          actions: Float[Array, "K H nu"],
      ) -> Float[Array, "K H"]:
        qpos: Float[Array, "K H nq"] = self.state_qpos(states)
        dist = self._tip_dist(qpos)

        # quadratic Cartesian distance cost (good gradient near target for stabilization)
        cost = (self._w_height * dist**2
                + self._w_ctrl * np.sum(np.square(actions), axis=-1))
        return cost  # (K, H)

    def terminal_cost(
        self,
        states: Float[Array, "K nstate"],
    ) -> Float[Array, "K"]:
        qpos: Float[Array, "K nq"] = self.state_qpos(states)
        qvel: Float[Array, "K nv"] = self.state_qvel(states)

        dist = self._tip_dist(qpos)
        vel_cost = np.sum(qvel**2, axis=-1)

        # terminal: quadratic distance + velocity penalty, scaled up
        return self._w_terminal * (dist**2 + 0.1 * vel_cost)

    def _get_obs(self) -> Float[ndarray, "4"]:
        return np.concatenate([self.data.qpos, self.data.qvel])

def _angle_diff(
        a: Float[ndarray, "..."],
        b: Float[ndarray, "..."],
) -> Float[ndarray, "..."]:
    diff = b - a
    return ((diff + np.pi) % (2.0 * np.pi)) - np.pi
