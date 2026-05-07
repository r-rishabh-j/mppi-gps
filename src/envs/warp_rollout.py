"""GPU-backed `batch_rollout` for MuJoCo envs via NVIDIA Warp + mujoco_warp.

Drop-in mixin: any class inheriting `WarpRolloutMixin` alongside a
`MuJoCoEnv`-shaped class can call `_init_warp(nworld=K)` after the parent
`__init__`, then `_batch_rollout_warp(state, U)` returns the same
`(states, costs, sensordata)` tuple as the CPU path.

Adapted from upstream `jaiselsingh1/mppi-gps/src/envs/mujoco_env.py`,
generalized to support models with mocap bodies (Adroit's target).

----------------------------------------------------------------------
Constraints (asserted in `_init_warp`)
----------------------------------------------------------------------
- ``model.na == 0`` — actuators with state (e.g. muscles) unsupported.
  All envs in this repo (acrobot, hopper, adroit_pen, adroit_relocate)
  satisfy this; muscles would need a separate path.
- ``nworld`` is fixed at construction. To change MPPI's K, re-instantiate
  the env. (mujoco_warp's per-world buffers are pre-allocated.)
- ``(K, H)`` pairing is captured into a CUDA graph on the first call.
  Changing H invalidates the graph; the mixin handles this by reallocating
  buffers and re-capturing on the next call. K is fixed (see above).
- Hardware: warp + mujoco_warp can theoretically run on CPU, but graph
  capture / replay (the speedup) is **CUDA-only**. On macOS or any
  non-NVIDIA host this path will either fall back to CPU (slow, no graph)
  or fail at `wp.ScopedCapture` — use the default CPU rollout instead.

----------------------------------------------------------------------
State layout returned
----------------------------------------------------------------------
``states.shape == (K, H, 1 + nq + nv)``, layout ``[time=0, qpos, qvel]``.
Identical to ``mj_getState(FULLPHYSICS)`` for envs with ``na==0`` (where
time + qpos + qvel exhausts the FULLPHYSICS state). All `state_qpos` /
`state_qvel` indexing in the codebase keeps working unchanged. The only
divergence is ``time`` — the warp path doesn't track sim time, so it
reports 0. None of our cost or obs functions read state[..., 0], so this
is invisible.

----------------------------------------------------------------------
Mocap handling (Adroit-relevant)
----------------------------------------------------------------------
Mocap pose is **static during a rollout** — set by `env.reset()` and not
updated during stepping (it's a configuration, not a state). Per rollout,
we broadcast `data.mocap_pos` / `data.mocap_quat` from the CPU model to
all `nworld` GPU worlds so collision geometry attached to mocap bodies
(Adroit's target site, containment walls anchored to mocap) sees the
right configuration. `mj_getState(FULLPHYSICS)` does NOT carry mocap, so
this has to be re-broadcast on every `_batch_rollout_warp` call.
"""
from __future__ import annotations

import numpy as np


class WarpRolloutMixin:
    """Adds a Warp-backed `_batch_rollout_warp` to a MuJoCoEnv subclass.

    Subclass usage:
        class MyEnvWarp(WarpRolloutMixin, MyEnv):
            def __init__(self, nworld, **kw):
                super().__init__(**kw)
                self._init_warp(nworld=nworld)
            def batch_rollout(self, state, actions):
                return self._batch_rollout_warp(state, actions)

    Inherits `self.model`, `self.data`, `self._frame_skip`,
    `self.running_cost`, `self.terminal_cost` from the MuJoCoEnv parent.
    """

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _init_warp(
        self,
        nworld: int,
        njmax: int = 256,
        nconmax: int = 96,
    ) -> None:
        """Initialize the Warp rollout side.

        ``njmax`` / ``nconmax`` are **per-world** constraint / contact buffer
        budgets passed into ``mjw.make_data``. They are NOT inherited from
        the XML's ``<size njmax=… nconmax=…>`` — mujoco_warp maintains its
        own per-world buffers separate from the CPU model's. Defaults are
        tuned for Adroit grasping (which routinely sees 60+ constraint
        rows when fingers + palm + table are all in contact). Bump if you
        see "nefc overflow - please increase njmax to N" — pick a value
        20-30% above the largest N reported, plus headroom for divergent
        rollouts.

        Memory cost: ``O(njmax * nworld * 8 bytes)`` per per-world buffer.
        At ``njmax=256, nworld=1024`` that's a couple of MB total — safe
        to crank well above the reported overflow.
        """
        # Lazy import — warp / mujoco_warp aren't required for the CPU path.
        # ImportError surfaces clearly to the user with an install hint.
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

        self._patch_model_for_warp()

        self._wp = wp
        self._mjw = mjw
        self._wm = mjw.put_model(self.model)

        # Try to pass njmax/nconmax explicitly; fall back to model-derived
        # defaults if this mjw version doesn't accept the kwargs (very
        # old/very new API drift). The TypeError path keeps the env
        # constructor working with the default budget so the user gets a
        # clear runtime error from mjw.step instead of a confusing kwarg
        # mismatch at construction.
        try:
            self._wd = mjw.make_data(
                self.model, nworld=nworld,
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
            self._wd = mjw.make_data(self.model, nworld=nworld)

        self._warp_nworld = nworld
        self._warp_H: int | None = None
        self._qpos_buf = None
        self._qvel_buf = None
        self._sensor_buf = None
        self._actions_wp = None
        self._rollout_graph = None
        self._has_mocap = self.model.nmocap > 0
        self._use_warp = True

    # ------------------------------------------------------------------
    # Compatibility patches
    # ------------------------------------------------------------------

    def _patch_model_for_warp(self) -> None:
        """Adjust ``self.model`` in-place to satisfy mujoco_warp's
        ``put_model`` constraints. Each patch emits a single warning so the
        change is visible to the user.

        These changes apply to ``self.model`` — the CPU ``env.step`` path
        also reads from this model, so CPU and warp rollouts stay
        physics-consistent (otherwise the action *executed* via
        ``env.step`` would have different physics from the rollout that
        scored it).

        Patches applied:
        - ``noslip_iterations``: zeroed. mujoco_warp doesn't implement the
          noslip post-pass solver. Adroit's XML defaults to 20 iterations.
          Effect: slightly less accurate sliding-friction handling. For
          grasp-style contacts (sticky, low slip) this is usually a no-op.
        - ``geom_margin`` / ``pair_margin``: zeroed where non-zero.
          mujoco_warp's ``put_model`` rejects geom pairs with non-zero
          margin under MULTICCD-style processing. Adroit's XML sets
          ``margin="0.0005"`` on every geom. Effect: contact detection
          fires only at actual penetration instead of within a 0.5 mm
          shell. Negligible for a grasping task.
        """
        import warnings
        m = self.model

        if m.opt.noslip_iterations > 0:
            warnings.warn(
                f"Disabling noslip solver (was noslip_iterations="
                f"{m.opt.noslip_iterations}) — mujoco_warp does not implement "
                "it. Friction on sliding contacts will be slightly less "
                "accurate; for Adroit grasp rollouts this is usually a no-op "
                "since contacts are sticky.",
                stacklevel=4,
            )
            m.opt.noslip_iterations = 0

        n_nonzero_geom = int((m.geom_margin > 0).sum())
        if n_nonzero_geom > 0:
            max_margin = float(m.geom_margin.max())
            warnings.warn(
                f"Zeroing geom_margin on {n_nonzero_geom}/{m.ngeom} geoms "
                f"(max was {max_margin:.4g} m) — mujoco_warp rejects non-zero "
                "geom margins under MULTICCD. Contact detection now fires at "
                "actual penetration; sub-millimetre margins are negligible "
                "for grasp tasks.",
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

    # ------------------------------------------------------------------
    # Buffer / graph management
    # ------------------------------------------------------------------

    def _ensure_warp_buffers(self, K: int, H: int) -> None:
        """(Re)allocate (H, K, *) buffers; invalidate captured graph if H changed."""
        if K != self._warp_nworld:
            raise RuntimeError(
                f"nworld fixed at construction ({self._warp_nworld}); got K={K}. "
                "Re-instantiate the env with nworld=K."
            )
        if self._warp_H == H:
            return
        wp = self._wp
        m = self.model
        # (H, K, *) so buf[h] is a (K, *) leading-axis subview wp.copy can target.
        self._qpos_buf   = wp.zeros((H, K, m.nq),          dtype=wp.float32)
        self._qvel_buf   = wp.zeros((H, K, m.nv),          dtype=wp.float32)
        self._sensor_buf = wp.zeros((H, K, m.nsensordata), dtype=wp.float32)
        self._actions_wp = wp.zeros((H, K, m.nu),          dtype=wp.float32)
        self._rollout_graph = None    # buffers changed → captured graph stale
        self._warp_H = H

    def _run_rollout(self, H: int) -> None:
        """Capture-safe inner loop. Each outer step copies the per-step ctrl
        into `wd.ctrl`, runs `frame_skip` `mjw.step`s, and snapshots qpos /
        qvel / sensordata for that boundary."""
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
        """Seed the GPU world buffers with the per-world initial state.

        Two input shapes are accepted (overload-by-shape):
          * ``initial_state.shape == (nstate,)`` — single FULLPHYSICS state
            broadcast to all ``nworld`` GPU worlds. Mocap pose is read
            from the CPU ``self.data`` (also broadcast). This is the
            single-condition path used by the standalone MPPI controller.
          * ``initial_state.shape == (nworld, nstate)`` — per-world initial
            state (one row per GPU world). Used by ``BatchedMPPI`` where
            ``nworld = N × K`` and rows are tiled so worlds 0..K-1 share
            condition 0's state, K..2K-1 share condition 1's, etc. Mocap
            broadcast falls back to the CPU ``self.data`` unless
            ``per_world_mocap`` is supplied — see below.

        ``per_world_mocap`` (optional, only used in the per-world path):
          * ``None`` (default) — broadcast ``self.data.mocap_pos`` /
            ``.mocap_quat`` to all worlds. Adequate when all conditions
            share the same mocap (e.g. all conditions point at the same
            relocate target). Most current usage hits this path.
          * ``(nworld, nmocap, 3+4)`` packed as ``(pos, quat)`` — per-world
            mocap. Use when conditions have *different* mocap targets
            (e.g. randomized target per condition). Caller must build it.

        Why FULLPHYSICS doesn't carry mocap: it's a configuration knob
        (set by ``env.reset()`` before the rollout) rather than dynamic
        state, so MuJoCo excludes it from the snapshot. We have to carry
        it separately on each rollout call.
        """
        import mujoco

        K = self._warp_nworld
        nq, nv = self.model.nq, self.model.nv

        # Single-state vs per-world via shape detection.
        is_per_world = (
            initial_state.ndim == 2 and initial_state.shape[0] == K
        )
        if is_per_world:
            # Round-trip each row through self.data to extract qpos/qvel
            # without re-implementing mj_setState. Cheap (only K calls,
            # no rollouts), and keeps a single source of truth for the
            # FULLPHYSICS → (qpos, qvel) layout.
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
                # Caller-supplied per-world mocap. Expect (K, nm, 7) packed
                # as [pos(3), quat(4)] per body. Split for the wd assignments.
                pwm = np.asarray(per_world_mocap, dtype=np.float32)
                if pwm.shape != (K, nm, 7):
                    raise ValueError(
                        f"per_world_mocap shape {pwm.shape} does not match "
                        f"({K}, {nm}, 7). Pack as [pos, quat] per body."
                    )
                self._wd.mocap_pos.assign(pwm[..., :3].copy())
                self._wd.mocap_quat.assign(pwm[..., 3:].copy())
            else:
                # Broadcast self.data mocap to all worlds (single-target
                # default; works for both per-world and single-state paths
                # when conditions share a mocap target).
                mp = np.broadcast_to(
                    self.data.mocap_pos.astype(np.float32), (K, nm, 3)
                ).copy()
                mq = np.broadcast_to(
                    self.data.mocap_quat.astype(np.float32), (K, nm, 4)
                ).copy()
                self._wd.mocap_pos.assign(mp)
                self._wd.mocap_quat.assign(mq)

    # ------------------------------------------------------------------
    # Rollout entry point
    # ------------------------------------------------------------------

    def _batch_rollout_warp(
        self,
        initial_state: np.ndarray,
        action_sequences: np.ndarray,
        per_world_mocap: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run B = nworld parallel rollouts of horizon H.

        ``initial_state`` is either ``(nstate,)`` (broadcast to all worlds)
        or ``(B, nstate)`` (per-world); see ``_seed_warp_state``.
        ``action_sequences`` is always ``(B, H, nu)`` — caller flattens any
        outer batching (e.g. (N, K) → B = N*K) before this call.
        """
        wp = self._wp
        B, H, nu = action_sequences.shape
        self._ensure_warp_buffers(B, H)
        self._seed_warp_state(initial_state, per_world_mocap=per_world_mocap)

        # (B, H, nu) → (H, B, nu): the inner loop indexes by h first so
        # actions_wp[h] is a contiguous (B, nu) slice for wp.copy.
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

        # Reconstruct FULLPHYSICS-layout states. time=0 since the GPU loop
        # doesn't track sim time; nothing in this codebase reads state[..., 0].
        time_col = np.zeros((B, H, 1), dtype=np.float32)
        states = np.concatenate([time_col, qpos, qvel], axis=-1)
        c  = self.running_cost(states, action_sequences, sensordata)   # (B, H)
        tc = self.terminal_cost(states[:, -1, :], sensordata[:, -1, :])  # (B,)
        costs = c.sum(axis=1) + tc
        return states, costs, sensordata
