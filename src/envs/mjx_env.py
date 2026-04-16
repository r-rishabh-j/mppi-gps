"""MJX-backed environment base class for GPU-accelerated batch rollouts.

Uses MuJoCo's JAX backend (MJX) to run K parallel trajectory rollouts
on GPU via ``jax.vmap`` + ``jax.lax.scan``.  The CPU ``MjModel``/``MjData``
is kept for single-step interactive use (reset, step, get/set state, render).

Cost functions are passed in as pure JAX callables from ``costs_jax.py``.
"""

from __future__ import annotations

import mujoco
import numpy as np
import jax
import jax.numpy as jnp
from mujoco import mjx
from typing import Any, Callable

from src.envs.base import BaseEnv


def _get_mjx_device():
    """Return the best JAX device that MJX supports.

    MJX works on CUDA GPUs and CPU.  Metal (Apple Silicon) is not
    supported by MJX, so we fall back to CPU on Metal — JAX JIT
    compilation still provides benefits over plain numpy.

    When falling back to CPU, we also set JAX's default device so that
    all ``jnp.array()``, ``jax.random.PRNGKey()``, etc. create arrays
    on CPU rather than the Metal device (which triggers unsupported ops).
    """
    for d in jax.devices():
        if d.platform == "gpu":  # CUDA
            return d
    # No CUDA GPU found — force JAX default to CPU so all array
    # creation avoids the Metal backend.
    cpu = jax.devices("cpu")[0]
    jax.config.update("jax_default_device", cpu)
    return cpu


class MJXEnv(BaseEnv):
    """GPU-accelerated environment using MJX for batch rollouts."""

    def __init__(
        self,
        model_path: str,
        frame_skip: int = 1,
        running_cost_fn: Callable | None = None,
        terminal_cost_fn: Callable | None = None,
        cost_kwargs: dict | None = None,
        nthread: int | None = None,
    ):
        # CPU model — used for reset/step/render/get_state/set_state
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        self._frame_skip = frame_skip
        self._dt = self.model.opt.timestep * frame_skip

        self._nstate = mujoco.mj_stateSize(
            self.model, mujoco.mjtState.mjSTATE_FULLPHYSICS
        )

        # MJX model — used for GPU/JIT batch rollouts.
        # MJX supports CUDA GPUs natively.  On Metal (Apple Silicon) or
        # other unsupported devices, fall back to CPU — we still get JAX
        # JIT compilation benefits over plain numpy.
        self._mjx_device = _get_mjx_device()
        self.mx_model = mjx.put_model(self.model, device=self._mjx_device)

        # Pre-build a template MjxData (used by _make_mjx_data inside JIT)
        self._mx_data_template = mjx.put_data(
            self.model, self.data, device=self._mjx_device
        )

        # Cost functions (pure JAX, from costs_jax.py).
        # cost_kwargs has "running" and "terminal" sub-dicts so each cost
        # function only receives the parameters it accepts.
        self._running_cost_fn = running_cost_fn
        self._terminal_cost_fn = terminal_cost_fn
        cost_kwargs = cost_kwargs or {}
        self._running_cost_kwargs = cost_kwargs.get("running", {})
        self._terminal_cost_kwargs = cost_kwargs.get("terminal", {})

        # Build and JIT the batch rollout.  We define it as a closure over
        # self so that mx_model and cost functions are captured at trace time.
        self._jit_batch_rollout = jax.jit(self._batch_rollout_inner)

    # ------------------------------------------------------------------
    # MJX batch rollout (GPU path)
    # ------------------------------------------------------------------

    def _make_mjx_data(self, state: jax.Array) -> Any:
        """Create an MjxData initialised from a flat state vector.

        The state layout matches ``mjSTATE_FULLPHYSICS``:
        ``[time, qpos(nq), qvel(nv), act(na)]``.
        """
        nq = self.model.nq
        nv = self.model.nv
        na = self.model.na

        t = state[0]
        qpos = state[1 : 1 + nq]
        qvel = state[1 + nq : 1 + nq + nv]
        act = state[1 + nq + nv : 1 + nq + nv + na] if na > 0 else jnp.zeros(0)

        # Use the pre-built template (avoids put_data inside JIT)
        mx_d = self._mx_data_template
        mx_d = mx_d.replace(time=t, qpos=qpos, qvel=qvel, act=act)
        return mx_d

    def _extract_state(self, mx_d: Any) -> jax.Array:
        """Reconstruct the flat state vector from MjxData.

        Matches the ``mjSTATE_FULLPHYSICS`` layout used by the CPU path.
        """
        parts = [jnp.array([mx_d.time]), mx_d.qpos, mx_d.qvel]
        if self.model.na > 0:
            parts.append(mx_d.act)
        return jnp.concatenate(parts)

    def _extract_sensor_or_site(self, mx_d: Any) -> jax.Array:
        """Extract sensor-equivalent data from MjxData.

        For environments that use sensordata (like acrobot's framepos),
        subclasses can override this.  Default returns sensordata.
        """
        return mx_d.sensordata

    def _batch_rollout_inner(
        self,
        initial_state: jax.Array,
        action_sequences: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Core JIT-compiled batch rollout.

        Args:
            initial_state: (nstate,) flat physics state.
            action_sequences: (K, H, nu) action sequences.

        Returns:
            states: (K, H, nstate) — state at each planning step.
            costs: (K,) — total cost per trajectory.
            sensordata: (K, H, nsensor_equiv) — sensor/site data.
        """
        K, H, nu = action_sequences.shape

        # Expand actions for frame_skip physics substeps
        actions_expanded = jnp.repeat(action_sequences, self._frame_skip, axis=1)

        # Single-trajectory rollout via lax.scan
        def rollout_one(mx_d_init: Any, actions: jax.Array):
            """Roll out one trajectory of H*frame_skip steps.

            actions: (H*frame_skip, nu)
            Returns: states (H*frame_skip, nstate), sensor (H*frame_skip, nsensor)
            """
            def scan_fn(mx_d, action):
                mx_d = mx_d.replace(ctrl=action)
                mx_d = mjx.step(self.mx_model, mx_d)
                state = self._extract_state(mx_d)
                sensor = self._extract_sensor_or_site(mx_d)
                return mx_d, (state, sensor)

            _, (states, sensordata) = jax.lax.scan(scan_fn, mx_d_init, actions)
            return states, sensordata

        # Initialise MjxData from the flat state — done once, shared by vmap
        mx_d_init = self._make_mjx_data(initial_state)

        # vmap over K trajectories
        batched_rollout = jax.vmap(rollout_one, in_axes=(None, 0))
        states_full, sensor_full = batched_rollout(mx_d_init, actions_expanded)
        # states_full: (K, H*frame_skip, nstate)
        # sensor_full: (K, H*frame_skip, nsensor)

        # Downsample to planning horizon
        states = states_full[:, self._frame_skip - 1 :: self._frame_skip, :]
        sensordata = sensor_full[:, self._frame_skip - 1 :: self._frame_skip, :]

        # Compute costs on-device
        c = self._running_cost_fn(
            states, action_sequences, sensordata, **self._running_cost_kwargs
        )  # (K, H)
        tc = self._terminal_cost_fn(
            states[:, -1, :], sensordata[:, -1, :], **self._terminal_cost_kwargs
        )  # (K,)
        costs = c.sum(axis=1) + tc

        return states, costs, sensordata

    def _to_jax(self, arr: np.ndarray) -> jax.Array:
        """Convert numpy to JAX array on the same device as the MJX model."""
        return jax.device_put(np.asarray(arr, dtype=np.float32), self._mjx_device)

    def batch_rollout(
        self,
        initial_state: np.ndarray,
        action_sequences: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Public API matching BaseEnv.  Accepts numpy, returns numpy."""
        states, costs, sensordata = self._jit_batch_rollout(
            self._to_jax(initial_state),
            self._to_jax(action_sequences),
        )
        return (
            np.asarray(states),
            np.asarray(costs),
            np.asarray(sensordata),
        )

    # ------------------------------------------------------------------
    # GPU-native batch rollout (returns JAX arrays, no conversion)
    # ------------------------------------------------------------------

    def batch_rollout_jax(
        self,
        initial_state: jax.Array,
        action_sequences: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Like batch_rollout but stays on-device (JAX arrays in/out).

        Used by MPPIJAX to avoid unnecessary device transfers.
        """
        return self._jit_batch_rollout(initial_state, action_sequences)

    # ------------------------------------------------------------------
    # CPU single-step interface (unchanged from MuJoCoEnv)
    # ------------------------------------------------------------------

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

    def state_qpos(self, states):
        return states[..., 1 : 1 + self.model.nq]

    def state_qvel(self, states):
        start = 1 + self.model.nq
        return states[..., start : start + self.model.nv]

    def _get_obs(self) -> np.ndarray:
        return self.get_state()

    # ------------------------------------------------------------------
    # Running / terminal cost (numpy, for single-step use)
    # ------------------------------------------------------------------

    def running_cost(self, states, actions, sensordata=None):
        """Numpy fallback for single-step cost (used by step())."""
        c = self._running_cost_fn(
            jnp.asarray(states), jnp.asarray(actions),
            jnp.asarray(sensordata) if sensordata is not None else None,
            **self._running_cost_kwargs,
        )
        return np.asarray(c)

    def terminal_cost(self, states, sensordata=None):
        tc = self._terminal_cost_fn(
            jnp.asarray(states),
            jnp.asarray(sensordata) if sensordata is not None else None,
            **self._terminal_cost_kwargs,
        )
        return np.asarray(tc)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def state_dim(self) -> int:
        return self._nstate

    @property
    def action_dim(self) -> int:
        return self.model.nu

    @property
    def obs_dim(self) -> int:
        return self._nstate  # Override in subclasses

    @property
    def action_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            self.model.actuator_ctrlrange[:, 0].copy(),
            self.model.actuator_ctrlrange[:, 1].copy(),
        )

    def state_to_obs(self, states):
        return states  # Override in subclasses

    def close(self):
        pass
