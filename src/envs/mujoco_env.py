"""mujoco env with batched rollouts using mujoco.rollout."""

import os
import numpy as np 
import mujoco 
from jaxtyping import Array, Float
from mujoco import rollout 

from src.envs.base import BaseEnv

class MuJoCoEnv(BaseEnv):
    def __init__(self, model_path: str, nthread: int | None = None, frame_skip: int = 1):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        # adding frame skip 
        self._frame_skip = frame_skip
        self._dt = self.model.opt.timestep * frame_skip

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
        for _ in range(self._frame_skip):
            mujoco.mj_step(self.model, self.data)
            obs = self._get_obs()
            state = self.get_state()
            sensor = self.data.sensordata.copy().reshape(1, 1, -1)
            c = self.running_cost(
                state.reshape(1, 1, -1), action.reshape(1, 1, -1), sensor
            ).item()
        return obs, c, False, {}
    
    def get_state(self) -> np.ndarray:
        state = np.empty(self._nstate)
        mujoco.mj_getState(
            self.model, self.data, state, 
            mujoco.mjtState.mjSTATE_FULLPHYSICS, 
        )
        return state

    def set_state(self, state: np.ndarray) -> None:
        mujoco.mj_setState(
            self.model, self.data, state,
            mujoco.mjtState.mjSTATE_FULLPHYSICS,
        )

    def state_qpos(
        self,
        states: Float[Array, "... nstate"],
    ) -> Float[Array, "... nq"]:
        return states[..., 1 : 1 + self.model.nq]

    def state_qvel(
        self,
        states: Float[Array, "... nstate"],
    ) -> Float[Array, "... nv"]:
        start = 1 + self.model.nq
        end = start + self.model.nv
        return states[..., start:end]

    def batch_rollout(
            self, 
            initial_state: np.ndarray, 
            action_sequences: np.ndarray, 
    ) -> tuple[np.ndarray, np.ndarray]:
        K, H, _ = action_sequences.shape 
        
        # repeat each action for frame_skip physics steps 
        actions_expanded = np.repeat(action_sequences, self._frame_skip, axis = 1) 
        states_full, sensordata_full = self._rollout_ctx.rollout(
            self.model,
            self._data_pool,
            initial_state,
            actions_expanded,
        )

        # downsample states: take every frame_skip-th frame
        states = states_full[:, ::self._frame_skip, :]
        sensordata = sensordata_full[:, ::self._frame_skip, :]

        c = self.running_cost(states, action_sequences, sensordata) # (K, H)
        tc = self.terminal_cost(states[:, -1, :], sensordata[:, -1, :]) # (K, )
        costs = c.sum(axis = 1) + tc
        return states, costs, sensordata
    

    def _get_obs(self) -> np.ndarray:
        """ can be overrided in the sub class if you need to change the obs space"""
        return self.get_state()

    def state_to_obs(self, states: np.ndarray) -> np.ndarray:
        """Default: return full state. Override in subclasses that have reduced obs."""
        return states

    @property
    def obs_dim(self) -> int:
        return self._nstate

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

