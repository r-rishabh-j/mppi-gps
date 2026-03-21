"""mujoco env with batched rollouts using mujoco.rollout."""

import os
import numpy as np 
import mujoco 
from mujoco import rollout 

from src.envs.base import BaseEnv

class MuJoCoEnv(BaseEnv):
    def __init__(self, model_path: str, nthread: int | None = None):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        # state size for full physics state (qpos, qvel, actions, etc)
        self._nstate = mujoco.mj_stateSize(
            self.model, mujoco.mjtState.mjSTATE_FULLPHYSICS
        )

        # thread pool for the batched rollouts 
        self._nthread = nthread or os.cpu_count()
        self._data_pool = [
            mujoco.MjData(self.model) for _ in range(self._nthread)
        ]

        self._rollout_ctx = rollout.Rollout(nthread = self._nthread)

    def reset(self, state: np.ndarray | None = None) -> np.ndarray:
        mujoco.mj_resetData(self.model, self.data)
        if state is not None:
            self.set_state(state)
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs()
    
    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        self.data.ctrl[:] = action 
        mujoco.mj_step(self.model, self.data)
        obs = self._get_obs()
        state = self.get_state()
        c = self.running_cost(
            state.reshape(1, 1, -1), action.reshape(1, 1, -1) # running cost has batched inputs (K, H, nstate) but step is just one state
        ).item()
        return obs, c, False, {}
    
    def get_state(self) -> np.ndarray:
        state = np.empty(self._nstate)
        mujoco.mj_getState(
            self.model, self.data, state, 
            mujoco.mjtState.mjSTATE_FULLPHYSICS, 
        )

    def batch_rollout(
            self, 
            initial_state: np.ndarray, 
            action_sequences: np.ndarray, 
    ) -> tuple[np.ndarray, np.ndarray]:
        K, H, _ = action_sequences.shape 
        states, _ = self._rollout_ctx.rollout(
            self.model, 
            self._data_pool, 
            initial_state, 
            action_sequences, 
        )
        c = self.running_cost(states, action_sequences) # (K, H)
        tc = self.terminal_cost(states[:, -1, :]) # (K, )
        costs = c.sum(axis = 1) + tc 
        return states, costs 
    

    def _get_obs(self) -> np.ndarray:
        """ can be overrided in the sub class if you need to change the obs space"""
        return self.get_state()
    
    @property 
    def state_dim(self) -> int:
        return self._nstate
    
    @property 
    def action_dim(self) -> int:
        return self.model.nu 
    
    @property
    def action_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            self.model.actuator_ctrlrange[:, 0].copy(),
            self.model.actuator_ctrlrange[:, 1].copy(),
        )

    def close(self):
        self._rollout_ctx.__exit__(None, None, None)


