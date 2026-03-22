"""half cheetah env"""

import numpy as np 
from pathlib import Path 

from src.envs.mujoco_env import MuJoCoEnv

_XML = str(
    Path(__file__).resolve().parents[2] / "assets" / "half_cheetah.xml"
)

class HalfCheetah(MuJoCoEnv):

    def __init__(self, ctrl_cost_weight: float = 0.1, frame_skip: int = 5,**kwargs):
        super().__init__(model_path=_XML, frame_skip=frame_skip, **kwargs)
        self._ctrl_w = ctrl_cost_weight 
        self._nq = self.model.nq # 9 
        self._nv = self.model.nv # 9 

    def running_cost(
            self, 
            states: np.ndarray, 
            actions: np.ndarray, # (K, H, action_dim)
    ) -> np.ndarray:
        # states are (K, H, nstate)
        # forward velocity is qvel[0] (root x-velocity)

        xvel = states[..., self._nq] # (K, H)
        ctrl = np.sum(actions ** 2, axis = -1) # (K, H)
        return -xvel + self._ctrl_w * ctrl # (K, H)
    
    def terminal_cost(self, states: np.ndarray) -> np.ndarray:
          # no terminal cost to start
          return np.zeros(states.shape[0])

    def _get_obs(self) -> np.ndarray:
        # standard half-cheetah obs: qpos[1:] + qvel (exclude root x)
        return np.concatenate([
            self.data.qpos[1:],
            self.data.qvel,
        ])
    
