
"""acrobot swing up env"""

import numpy as np 
from pathlib import Path 
from jaxtyping import Array, Float

from src.envs.mujoco_env import MuJoCoEnv

_XML = str(Path(__file__).resolve().parents[2] / "assets" / "half_cheetah.xml")

class Acrobot(MuJoCoEnv):
    def __init__(self, frame_skip: int = 5, **kwargs):
        super().__init__(model_path=_XML, frame_skip=frame_skip, **kwargs)
        self._nq = self.model.nq # 2 
        self._nv = self.model.nv #2 

        # this is upright, zero velocity and then the angles for the two joints 
        self._x_goal = np.array([np.pi, 0.0, 0.0, 0.0])

        # cost weights 
        self._Q = np.array([10.0, 1.0, 0.1, 0.1])  # [shoulder, elbow, dshoul, delbow])
        self._R = 0.01 
        self._P_scale = 1000.0 # terminal cost multiplier 

    
    def running_cost(self,
                     states: np.ndarray, 
                     actions: np.ndarray) -> float:
        # states: (K, H, nstate), actions: (K, H, nu)
        qpos = states[:, :, :self._nq]
        qvel = states[:, :, self._nq:self._nq + self._nv]

        # angle error (wrap the shoulder between negative pi and pi)
        angle_err_shoulder = _angle_diff(qpos[:, :, 0], self._x_goal[0])
        angle_err_elbow = qpos[:, :, 1] - self._x_goal[1]
        vel_err_shoulder = qvel[:, :, 0] - self._x_goal[2]
        vel_err_elbow = qvel[:, :, 1] - self._x_goal[3]

        # weighted quadratic cost
        cost = (self._Q[0] * angle_err_shoulder**2
            + self._Q[1] * angle_err_elbow**2
            + self._Q[2] * vel_err_shoulder**2
            + self._Q[3] * vel_err_elbow**2
            + self._R * np.sum(np.square(actions), axis=-1))
        return cost  # (K, H)
    
    def terminal_cost(self, states: np.ndarray) -> float:
    # states: (K, nstate)
        qpos = states[:, :self._nq]
        qvel = states[:, self._nq:self._nq + self._nv]

        angle_err_shoulder = _angle_diff(qpos[:, 0], self._x_goal[0])
        angle_err_elbow = qpos[:, 1] - self._x_goal[1]
        vel_err_shoulder = qvel[:, 0] - self._x_goal[2]
        vel_err_elbow = qvel[:, 1] - self._x_goal[3]

        cost = self._P_scale * (
            self._Q[0] * angle_err_shoulder**2
        + self._Q[1] * angle_err_elbow**2
        + self._Q[2] * vel_err_shoulder**2
        + self._Q[3] * vel_err_elbow**2)
        return cost  # (K,)
    
    def _get_obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel])
    
def _angle_diff(a, b) -> float:
    # signed angular distance wrapped by [-pi, pi]
    diff = a - b 
    return ((diff + np.pi) % 2*np.pi) - np.pi 
 


        