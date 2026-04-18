"""mujoco env with pluggable batched-rollout backend (cpu threads or mujoco_warp)."""

import numpy as np
import mujoco
from jaxtyping import Array, Float

from src.envs.base import BaseEnv
from src.envs.rollout_backends import RolloutBackend, make_backend


class MuJoCoEnv(BaseEnv):
    def __init__(
        self,
        model_path: str,
        nthread: int | None = None,
        frame_skip: int = 1,
        backend: str = "cpu",
        backend_kwargs: dict | None = None,
    ):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        self._frame_skip = frame_skip
        self._dt = self.model.opt.timestep * frame_skip

        self._nstate = mujoco.mj_stateSize(
            self.model, mujoco.mjtState.mjSTATE_FULLPHYSICS
        )

        bkwargs = dict(backend_kwargs or {})
        if backend == "cpu" and nthread is not None:
            bkwargs.setdefault("nthread", nthread)
        self._backend_name = backend
        self._backend: RolloutBackend = make_backend(
            backend, self.model, frame_skip=frame_skip, **bkwargs
        )

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
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._backend.rollout(self, initial_state, action_sequences)

    def running_cost_torch(self, states, actions, sensordata=None):
        """Torch/GPU cost. Subclasses targeting the warp backend must override.

        Signature mirrors running_cost but operates on torch tensors on the
        backend's device; returns a (K, H) torch tensor.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.running_cost_torch not implemented; "
            "required when rollout backend is 'warp'."
        )

    def terminal_cost_torch(self, states, sensordata=None):
        raise NotImplementedError(
            f"{type(self).__name__}.terminal_cost_torch not implemented; "
            "required when rollout backend is 'warp'."
        )

    def _get_obs(self) -> np.ndarray:
        """Can be overridden in the subclass if you need to change the obs space."""
        return self.get_state()

    def state_to_obs(self, states: np.ndarray) -> np.ndarray:
        """Default: return full state. Override in subclasses with reduced obs."""
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
        self._backend.close()
