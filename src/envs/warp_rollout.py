"""GPU ``batch_rollout`` mixin via NVIDIA Warp + mujoco_warp (CUDA only).

Mix into a ``MuJoCoEnv`` subclass; call ``_init_warp(nworld=K)`` after
the parent ``__init__``, then dispatch ``batch_rollout`` to
``_batch_rollout_warp``. Returns the same ``(states, costs, sensordata)``
shape as the CPU path.

Constraints (asserted in ``_init_warp``):
- ``model.na == 0`` (no muscle / state actuators).
- ``nworld`` fixed at construction; to change K, reinstantiate.
- ``H`` may change (buffers + CUDA graph re-captured); K may not.
- CUDA only — graph capture/replay is the speedup. macOS / non-NVIDIA
  hosts: use the default CPU rollout.

State layout: ``(K, H, 1 + nq + nv) = [time=0, qpos, qvel]``. Matches
``mj_getState(FULLPHYSICS)`` for ``na==0``; ``time`` is always 0 (warp
loop doesn't track sim time; nothing here reads ``state[..., 0]``).

Mocap is a configuration (set by reset, static during a rollout) and is
NOT carried by FULLPHYSICS, so we re-broadcast ``data.mocap_pos/quat``
from the CPU model to all worlds on every rollout.
"""
from __future__ import annotations

import numpy as np


class WarpRolloutMixin:
    """Warp-backed ``_batch_rollout_warp`` for a MuJoCoEnv subclass.

    Inherits ``model``, ``data``, ``_frame_skip``, ``running_cost``,
    ``terminal_cost`` from the parent.
    """

    def _init_warp(
        self,
        nworld: int,
        njmax: int = 256,
        nconmax: int = 96,
    ) -> None:
        """Initialize the Warp side.

        ``njmax`` / ``nconmax`` are per-world buffer budgets for
        ``mjw.make_data`` (not inherited from XML). Defaults sized for
        Adroit grasping. Bump on "nefc overflow" — memory is cheap
        (``O(njmax · nworld · 8)`` bytes).
        """
        try:
            import warp as wp
            import mujoco_warp as mjw
        except ImportError as e:
            raise ImportError(
                "Warp rollout requires `warp-lang` and `mujoco-warp`. "
                "Install with: `uv pip install warp-lang mujoco-warp` "
                "(NVIDIA GPU + CUDA required for graph-replay speedup)."
            ) from e

        assert self.model.na == 0, (
            f"warp path requires na==0 (no actuator state); got na={self.model.na}. "
            "Position actuators are fine; muscles are not."
        )

        # Deepcopy isolates the warp model from the CPU executor's. If
        # mjw rejects a model feature on your version, uncomment the
        # patch line below (warp side only).
        import copy
        warp_model = copy.deepcopy(self.model)
        # self._patch_model_for_warp(warp_model)

        self._wp = wp
        self._mjw = mjw
        self._wm = mjw.put_model(warp_model)

        # Older mjw versions don't accept the budget kwargs.
        try:
            self._wd = mjw.make_data(
                warp_model, nworld=nworld,
                njmax=njmax, nconmax=nconmax,
            )
        except TypeError:
            import warnings
            warnings.warn(
                "mjw.make_data does not accept njmax/nconmax kwargs in this "
                "mujoco_warp version — falling back to its default per-world "
                "buffers. If you hit 'nefc overflow', upgrade mujoco_warp.",
                stacklevel=3,
            )
            self._wd = mjw.make_data(warp_model, nworld=nworld)

        self._warp_nworld = nworld
        self._warp_H: int | None = None
        self._qpos_buf = None
        self._qvel_buf = None
        self._sensor_buf = None
        self._actions_wp = None
        self._rollout_graph = None
        self._has_mocap = self.model.nmocap > 0
        self._use_warp = True

    def _patch_model_for_warp(self, model=None) -> None:
        """Opt-in patches for older mjw versions that reject model features.

        Pass the warp deepcopy from ``_init_warp`` to keep CPU side
        unchanged. Patches: zero ``noslip_iterations`` (sliding friction
        slightly less accurate), zero ``geom_margin`` / ``pair_margin``
        (contact detection at actual penetration). Each patch emits a
        warning.
        """
        import warnings
        m = model if model is not None else self.model

        if m.opt.noslip_iterations > 0:
            warnings.warn(
                f"Disabling noslip solver (was noslip_iterations="
                f"{m.opt.noslip_iterations}) — mujoco_warp does not implement "
                "it on this version. Friction on sliding contacts will be "
                "slightly less accurate.",
                stacklevel=4,
            )
            m.opt.noslip_iterations = 0

        n_nonzero_geom = int((m.geom_margin > 0).sum())
        if n_nonzero_geom > 0:
            max_margin = float(m.geom_margin.max())
            warnings.warn(
                f"Zeroing geom_margin on {n_nonzero_geom}/{m.ngeom} geoms "
                f"(max was {max_margin:.4g} m) — mujoco_warp rejects non-zero "
                "geom margins under MULTICCD on this version.",
                stacklevel=4,
            )
            m.geom_margin[:] = 0.0

        if m.npair > 0 and (m.pair_margin > 0).any():
            warnings.warn(
                f"Zeroing pair_margin on {int((m.pair_margin > 0).sum())}/"
                f"{m.npair} explicit contact pairs (same MULTICCD reason).",
                stacklevel=4,
            )
            m.pair_margin[:] = 0.0

    def _ensure_warp_buffers(self, K: int, H: int) -> None:
        """(Re)allocate (H, K, *) buffers; invalidate graph if H changed."""
        if K != self._warp_nworld:
            raise RuntimeError(
                f"nworld fixed at construction ({self._warp_nworld}); got K={K}. "
                "Re-instantiate the env with nworld=K."
            )
        if self._warp_H == H:
            return
        wp = self._wp
        m = self.model
        # (H, K, *): buf[h] is a (K, *) subview wp.copy can target.
        self._qpos_buf   = wp.zeros((H, K, m.nq),          dtype=wp.float32)
        self._qvel_buf   = wp.zeros((H, K, m.nv),          dtype=wp.float32)
        self._sensor_buf = wp.zeros((H, K, m.nsensordata), dtype=wp.float32)
        self._actions_wp = wp.zeros((H, K, m.nu),          dtype=wp.float32)
        self._rollout_graph = None
        self._warp_H = H

    def _run_rollout(self, H: int) -> None:
        """Capture-safe inner loop: copy ctrl, frame_skip × mjw.step, snapshot."""
        wp = self._wp
        for h in range(H):
            wp.copy(self._wd.ctrl, self._actions_wp[h])
            for _ in range(self._frame_skip):
                self._mjw.step(self._wm, self._wd)
            wp.copy(self._qpos_buf[h],   self._wd.qpos)
            wp.copy(self._qvel_buf[h],   self._wd.qvel)
            wp.copy(self._sensor_buf[h], self._wd.sensordata)

    def _seed_warp_state(
        self,
        initial_state: np.ndarray,
        per_world_mocap: np.ndarray | None = None,
    ) -> None:
        """Seed GPU buffers with the initial state.

        ``initial_state`` shape:
          * ``(nstate,)``           — broadcast to all worlds.
          * ``(nworld, nstate)``    — per-world (used by BatchedMPPI).

        ``per_world_mocap`` (optional, ``(K, nmocap, 7)`` as ``[pos, quat]``):
        used when conditions have different mocap targets. Default
        broadcasts ``self.data.mocap_*`` to all worlds.
        """
        import mujoco

        K = self._warp_nworld
        nq, nv = self.model.nq, self.model.nv

        is_per_world = (
            initial_state.ndim == 2 and initial_state.shape[0] == K
        )
        if is_per_world:
            # Round-trip each row through self.data to reuse mj_setState's
            # FULLPHYSICS → (qpos, qvel) decoding (K calls, no rollouts).
            qpos0 = np.empty((K, nq), dtype=np.float32)
            qvel0 = np.empty((K, nv), dtype=np.float32)
            for k in range(K):
                mujoco.mj_setState(
                    self.model, self.data, initial_state[k],
                    mujoco.mjtState.mjSTATE_FULLPHYSICS,
                )
                qpos0[k] = self.data.qpos
                qvel0[k] = self.data.qvel
        else:
            mujoco.mj_setState(
                self.model, self.data, initial_state,
                mujoco.mjtState.mjSTATE_FULLPHYSICS,
            )
            qpos0 = np.broadcast_to(
                self.data.qpos.astype(np.float32), (K, nq)
            ).copy()
            qvel0 = np.broadcast_to(
                self.data.qvel.astype(np.float32), (K, nv)
            ).copy()

        self._wd.qpos.assign(qpos0)
        self._wd.qvel.assign(qvel0)

        if self._has_mocap:
            nm = self.model.nmocap
            if per_world_mocap is not None:
                pwm = np.asarray(per_world_mocap, dtype=np.float32)
                if pwm.shape != (K, nm, 7):
                    raise ValueError(
                        f"per_world_mocap shape {pwm.shape} does not match "
                        f"({K}, {nm}, 7). Pack as [pos, quat] per body."
                    )
                self._wd.mocap_pos.assign(pwm[..., :3].copy())
                self._wd.mocap_quat.assign(pwm[..., 3:].copy())
            else:
                mp = np.broadcast_to(
                    self.data.mocap_pos.astype(np.float32), (K, nm, 3)
                ).copy()
                mq = np.broadcast_to(
                    self.data.mocap_quat.astype(np.float32), (K, nm, 4)
                ).copy()
                self._wd.mocap_pos.assign(mp)
                self._wd.mocap_quat.assign(mq)

    def _batch_rollout_warp(
        self,
        initial_state: np.ndarray,
        action_sequences: np.ndarray,
        per_world_mocap: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """B parallel rollouts of horizon H. ``action_sequences`` is (B, H, nu)."""
        wp = self._wp
        B, H, nu = action_sequences.shape
        self._ensure_warp_buffers(B, H)
        self._seed_warp_state(initial_state, per_world_mocap=per_world_mocap)

        # (B, H, nu) → (H, B, nu) so actions_wp[h] is a contiguous (B, nu).
        self._actions_wp.assign(
            np.ascontiguousarray(
                action_sequences.transpose(1, 0, 2).astype(np.float32)
            )
        )

        if self._rollout_graph is None:
            with wp.ScopedCapture() as capture:
                self._run_rollout(H)
            self._rollout_graph = capture.graph
        wp.capture_launch(self._rollout_graph)

        qpos       = self._qpos_buf.numpy().transpose(1, 0, 2)
        qvel       = self._qvel_buf.numpy().transpose(1, 0, 2)
        sensordata = self._sensor_buf.numpy().transpose(1, 0, 2)

        # FULLPHYSICS-layout states; time=0 (warp doesn't track it).
        time_col = np.zeros((B, H, 1), dtype=np.float32)
        states = np.concatenate([time_col, qpos, qvel], axis=-1)
        c  = self.running_cost(states, action_sequences, sensordata)   # (B, H)
        tc = self.terminal_cost(states[:, -1, :], sensordata[:, -1, :])  # (B,)
        costs = c.sum(axis=1) + tc
        return states, costs, sensordata
