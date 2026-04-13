"""half cheetah env"""

import numpy as np 
from pathlib import Path 
from jaxtyping import Array, Float

from src.envs.mujoco_env import MuJoCoEnv

_XML = str(
    Path(__file__).resolve().parents[2] / "assets" / "half_cheetah.xml"
)

class HalfCheetah(MuJoCoEnv):

    def __init__(self, ctrl_cost_weight: float = 0.0, frame_skip: int = 5,**kwargs):
        super().__init__(model_path=_XML, frame_skip=frame_skip, **kwargs)
        self._ctrl_w = ctrl_cost_weight 
        self._nq = self.model.nq # 9 
        self._nv = self.model.nv # 9 

    def running_cost(
            self,
            states: np.ndarray,
            actions: np.ndarray, # (K, H, action_dim)
            sensordata: np.ndarray | None = None,
    ) -> np.ndarray:
        # states are (K, H, nstate)
        # forward velocity is qvel[0] (root x-velocity)

        w_vel: float = 1.0 
        w_pitch: float = 0.5 
        w_controls: float = 0.001  

        qpos: Float[Array, "K T nq"] = states[:, :, :self._nq]
        qvel: Float[Array, "K T nv"] = states[:, :, self._nq:self._nq + self._nv]

        vx: Float[Array, "K T"] = qvel[:, :, 0]
        torso_pitch: Float[Array, "K T"] = qpos[:, :, 2]

        ctrls_cost = np.sum(np.square(actions), axis=-1)
        stage_cost = -w_vel * vx + w_pitch * (torso_pitch ** 2) + w_controls * ctrls_cost

        return stage_cost

    
    def terminal_cost(self, states: np.ndarray, sensordata: np.ndarray | None = None) -> np.ndarray:
          # no terminal cost to start
          return np.zeros(states.shape[0])

    def _get_obs(self) -> np.ndarray:
        # standard half-cheetah obs: qpos[1:] + qvel (exclude root x)
        return np.concatenate([
            self.data.qpos[1:],
            self.data.qvel,
        ])

    def state_to_obs(self, states: np.ndarray) -> np.ndarray:
        qpos = self.state_qpos(states)[..., 1:]  # skip root x
        qvel = self.state_qvel(states)
        return np.concatenate([qpos, qvel], axis=-1)

    @property
    def obs_dim(self) -> int:
        return (self._nq - 1) + self._nv
    
