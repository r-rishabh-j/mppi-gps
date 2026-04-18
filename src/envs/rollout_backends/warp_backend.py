"""GPU batch rollout via mujoco_warp + torch cost functions.

Physics runs on CUDA using mujoco_warp's batched `nworld` data. The entire
H-step rollout is captured once as a CUDA graph (per (K, H) shape) and
replayed on each MPPI iteration — this is the only way to amortise
kernel-launch overhead, which otherwise dominates (~100-1000× slower than
the CPU threadpool on small-K Hopper).

Buffers (actions, ctrl, qpos/qvel/sensor outputs) are stable torch tensors
on CUDA; the warp↔torch bridge is zero-copy. Only initial state is written
outside the captured graph; per-iter the host writes the action buffer and
launches the graph.

Only time/qpos/qvel slots of the MuJoCo FULLPHYSICS state are populated in
the returned states array — that is all the current cost / state_to_obs
paths consume.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Any
import numpy as np
import mujoco

if TYPE_CHECKING:
    from src.envs.mujoco_env import MuJoCoEnv


def _lazy_imports():
    try:
        import warp as wp
        import mujoco_warp as mjwarp
        import torch
    except ImportError as e:
        raise ImportError(
            "Warp rollout backend requires `warp-lang`, `mujoco-warp`, and `torch`. "
            "Install with `uv sync --extra gpu`."
        ) from e
    return wp, mjwarp, torch


class WarpRolloutBackend:
    def __init__(
        self,
        model: mujoco.MjModel,
        frame_skip: int = 1,
        device: str = "cuda",
    ):
        wp, mjwarp, torch = _lazy_imports()
        wp.init()
        self._wp = wp
        self._mjwarp = mjwarp
        self._torch = torch
        self._device = torch.device(device)
        self._wp_device = str(device)  # e.g. "cuda:0"

        self._mj_model = model
        self._frame_skip = frame_skip
        self._mjx_model = mjwarp.put_model(model)

        self._nq = model.nq
        self._nv = model.nv
        self._nu = model.nu
        self._nsensor = model.nsensordata
        self._nstate = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)

        # Shape-dependent state rebuilt on (K, H) changes.
        self._shape_cache: tuple[int, int] | None = None
        self._mjx_data: Any = None
        self._ctrl_view: Any = None
        self._qpos_view: Any = None
        self._qvel_view: Any = None
        self._time_view: Any = None
        self._sensor_view: Any = None
        self._actions_buf: Any = None
        self._qpos_buf: Any = None
        self._qvel_buf: Any = None
        self._sensor_buf: Any = None
        self._rollout_graph: Any = None

    def _ensure_shape(self, K: int, H: int):
        if self._shape_cache == (K, H):
            return
        wp = self._wp
        mjwarp = self._mjwarp
        torch = self._torch

        # make_data wants the raw mujoco MjModel (not the warp Model).
        # Pass *conmax/*jmax explicitly to dodge a numpy-2 incompatibility in
        # mujoco_warp._default_nconmax for models with a single geom_type.
        # Scale with K so the batched contact broadphase has capacity per world.
        ngeom = max(self._mj_model.ngeom, 1)
        per_world = max(ngeom * ngeom, 8)
        cmax = K * per_world
        self._mjx_data = mjwarp.make_data(
            self._mj_model,
            nworld=K,
            nconmax=cmax, nccdmax=cmax, njmax=cmax,
            naconmax=cmax, naccdmax=cmax,
        )
        self._ctrl_view = wp.to_torch(self._mjx_data.ctrl)
        self._qpos_view = wp.to_torch(self._mjx_data.qpos)
        self._qvel_view = wp.to_torch(self._mjx_data.qvel)
        self._sensor_view = wp.to_torch(self._mjx_data.sensordata)
        self._time_view = wp.to_torch(self._mjx_data.time)

        dtype = self._qpos_view.dtype
        dev = self._device
        self._actions_buf = torch.zeros((K, H, self._nu), dtype=dtype, device=dev)
        self._qpos_buf = torch.zeros((K, H, self._nq), dtype=dtype, device=dev)
        self._qvel_buf = torch.zeros((K, H, self._nv), dtype=dtype, device=dev)
        self._sensor_buf = torch.zeros(
            (K, H, max(self._nsensor, 1)), dtype=dtype, device=dev
        )

        # Forward once to compile kernels outside graph capture.
        mjwarp.forward(self._mjx_model, self._mjx_data)
        mjwarp.step(self._mjx_model, self._mjx_data)

        # Route torch ops onto warp's capture stream — the default torch
        # stream (the CUDA "legacy" stream) is not graph-capture-safe.
        wp_stream = wp.get_stream(self._wp_device)
        self._torch_stream = torch.cuda.ExternalStream(wp_stream.cuda_stream)
        torch.cuda.synchronize()

        with torch.cuda.stream(self._torch_stream):
            with wp.ScopedCapture(device=self._wp_device) as capture:
                for t in range(H):
                    for _ in range(self._frame_skip):
                        self._ctrl_view.copy_(self._actions_buf[:, t, :])
                        mjwarp.step(self._mjx_model, self._mjx_data)
                    self._qpos_buf[:, t, :].copy_(self._qpos_view)
                    self._qvel_buf[:, t, :].copy_(self._qvel_view)
                    if self._nsensor > 0:
                        self._sensor_buf[:, t, :].copy_(self._sensor_view)
            self._rollout_graph = capture.graph

        self._shape_cache = (K, H)

    def _set_initial_state(self, initial_state: np.ndarray, K: int):
        torch = self._torch
        t0 = float(initial_state[0])
        qpos0 = initial_state[1 : 1 + self._nq]
        qvel0 = initial_state[1 + self._nq : 1 + self._nq + self._nv]

        qpos_t = torch.as_tensor(qpos0, dtype=self._qpos_view.dtype, device=self._device)
        qvel_t = torch.as_tensor(qvel0, dtype=self._qvel_view.dtype, device=self._device)
        self._qpos_view.copy_(qpos_t.unsqueeze(0).expand(K, self._nq).contiguous())
        self._qvel_view.copy_(qvel_t.unsqueeze(0).expand(K, self._nv).contiguous())
        self._time_view.fill_(t0)
        self._ctrl_view.zero_()
        self._mjwarp.forward(self._mjx_model, self._mjx_data)

    def rollout(
        self,
        env: "MuJoCoEnv",
        initial_state: np.ndarray,
        action_sequences: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        wp = self._wp
        torch = self._torch

        K, H, nu = action_sequences.shape
        assert nu == self._nu, f"action dim {nu} != model nu {self._nu}"
        self._ensure_shape(K, H)
        self._set_initial_state(initial_state, K)

        # Upload actions into the stable graph-bound buffer.
        actions_src = torch.as_tensor(
            action_sequences, dtype=self._actions_buf.dtype, device=self._device
        )
        self._actions_buf.copy_(actions_src)

        # Launch the captured rollout graph (one python call = H steps).
        wp.capture_launch(self._rollout_graph)

        # Pack states in FULLPHYSICS layout (time + qpos + qvel, rest zero).
        states_buf = torch.zeros(
            (K, H, self._nstate), dtype=self._qpos_buf.dtype, device=self._device
        )
        states_buf[..., 1 : 1 + self._nq] = self._qpos_buf
        states_buf[..., 1 + self._nq : 1 + self._nq + self._nv] = self._qvel_buf

        sensor_view = (
            self._sensor_buf[..., : self._nsensor] if self._nsensor > 0
            else self._sensor_buf[..., :0]
        )

        running = env.running_cost_torch(states_buf, self._actions_buf, sensor_view)        # (K, H)
        terminal = env.terminal_cost_torch(
            states_buf[:, -1, :],
            sensor_view[:, -1, :] if self._nsensor > 0 else sensor_view[:, -1, :],
        )  # (K,)
        costs = running.sum(dim=1) + terminal

        return (
            states_buf.detach().cpu().numpy().astype(np.float64),
            costs.detach().cpu().numpy().astype(np.float64),
            sensor_view.detach().cpu().numpy().astype(np.float64),
        )

    def close(self) -> None:
        self._rollout_graph = None
        self._mjx_data = None
        self._shape_cache = None
