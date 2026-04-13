"""wrap a gymnasium env for MPPI usage"""

import os 
import numpy as np 
import mujoco 
from mujoco import rollout 
import gymnasium as gym 

from src.envs.base import BaseEnv 

class GymEnv(BaseEnv):
    """Wraps a Gymnasium MuJoCo env, adding batch_rollout and state access."""

    def __init__(self, env_id: str, nthread: int | None = None, render_mode = "human"):
        self.gym_env = gym.make(env_id, render_mode = render_mode)
        unwrapped = self.gym_env.unwrapped

        # access the underlying MuJoCo model/data from Gymnasium
        self.model = unwrapped.model
        self.data = unwrapped.data
        self._frame_skip = unwrapped.frame_skip
        self._dt = self.model.opt.timestep * self._frame_skip

        self._nstate = mujoco.mj_stateSize(
            self.model, mujoco.mjtState.mjSTATE_FULLPHYSICS
        )

        # thread pool for batched rollouts
        self._nthread = nthread or os.cpu_count() or 1
        self._data_pool = [
            mujoco.MjData(self.model) for _ in range(self._nthread)
        ]
        self._rollout_ctx = rollout.Rollout(nthread=self._nthread)

    def reset(self, state=None):
        if state is not None:
            obs, _ = self.gym_env.reset()
            self.set_state(state)
            return self._get_obs()
        obs, _ = self.gym_env.reset()
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self.gym_env.step(action)
        # MPPI minimizes cost, Gymnasium maximizes reward
        cost = -reward
        return obs, cost, terminated or truncated, info

    def get_state(self):
        state = np.empty(self._nstate)
        mujoco.mj_getState(
            self.model, self.data, state,
            mujoco.mjtState.mjSTATE_FULLPHYSICS,
        )
        return state

    def set_state(self, state):
        mujoco.mj_setState(
            self.model, self.data, state,
            mujoco.mjtState.mjSTATE_FULLPHYSICS,
        )
        mujoco.mj_forward(self.model, self.data)

    def running_cost(self, states, actions):
        # extract forward velocity and control cost matching Gymnasium's reward
        xvel = states[..., self.model.nq]          # qvel[0]
        ctrl = np.sum(actions ** 2, axis=-1)
        # Gymnasium HalfCheetah: reward = forward_reward - ctrl_cost
        return -xvel + 0.1 * ctrl

    def terminal_cost(self, states):
        return np.zeros(states.shape[0])

    def batch_rollout(self, initial_state, action_sequences):
        K, H, _ = action_sequences.shape
        actions_expanded = np.repeat(action_sequences, self._frame_skip, axis=1)
        states_full, _ = self._rollout_ctx.rollout(
            self.model, self._data_pool, initial_state, actions_expanded,
        )
        states = states_full[:, self._frame_skip - 1::self._frame_skip, :]
        c = self.running_cost(states, action_sequences)
        tc = self.terminal_cost(states[:, -1, :])
        costs = c.sum(axis=1) + tc
        return states, costs

    def _get_obs(self):
        return self.gym_env.unwrapped._get_obs()

    def state_to_obs(self, states):
        return states

    @property
    def obs_dim(self):
        return self.gym_env.observation_space.shape[0]

    @property
    def state_dim(self):
        return self._nstate

    @property
    def action_dim(self):
        return self.model.nu

    @property
    def action_bounds(self):
        return (
            self.gym_env.action_space.low.copy(),
            self.gym_env.action_space.high.copy(),
        )

    def close(self):
        self._rollout_ctx.__exit__(None, None, None)
        self.gym_env.close()