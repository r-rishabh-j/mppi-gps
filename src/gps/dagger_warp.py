"""Warp-batched DAgger.

Same outer protocol as ``DAggerTrainer`` (rollout → relabel → finetune),
but the inner rollout loop runs ``N`` independent environments in
parallel and queries ``BatchedMPPI`` once per timestep instead of N
times. The amortized cost is one CUDA-graph launch per timestep across
``N × K`` worlds, vs. N sequential CPU MPPI calls each rolling K worlds.

The buffer / training / finetune / warmup driver / step() top-level loop
are inherited from ``DAggerTrainer`` unchanged — they operate on flat
``(obs, action)`` arrays which look identical regardless of how
``_collect`` produced them. Only the rollout collection is overridden.

----------------------------------------------------------------------
Design notes
----------------------------------------------------------------------

* ``N = mppi.N`` (== ``num_conditions`` in BatchedMPPI). The warp env
  must be constructed with ``nworld = N × cfg.K`` so MPPI's batched
  rollouts and the executor's parallel CPU envs both fit.

* If the user requests ``n_rollouts`` that's not a multiple of N, the
  final batch still runs all N envs in parallel but only the first
  ``n_rollouts - chunks_done * N`` of them contribute rows to the buffer.
  Cheap waste (N-rollouts always cost the same to roll out) but keeps
  the row count exactly matching the user's request.

* Per-condition β-mixing: each of the N envs decides independently
  whether to execute MPPI or policy on each step. Mirrors the CPU
  trainer's per-rollout coin flip — under the same ``beta`` the average
  fraction of steps executing MPPI is the same.

* ``done`` handling: the helper ``_step_one_data`` does not return a
  done flag. For Adroit relocate (never terminates) this is invisible.
  For Hopper-style terminating envs, condition-level termination would
  need per-condition reset logic — explicitly out of scope for v1; the
  trainer keeps stepping all N envs through ``ep_len`` regardless.
"""
from __future__ import annotations

import mujoco
import numpy as np
import torch
from tqdm.auto import tqdm

from src.envs.base import BaseEnv
from src.gps.dagger import DAggerTrainer
from src.gps.mppi_gps_warp import _set_state_one_data, _step_one_data
from src.mppi.batched_mppi import BatchedMPPI
from src.utils.config import DAggerConfig


class WarpDAggerTrainer(DAggerTrainer):
    """DAgger trainer driven by ``BatchedMPPI``.

    Constructor signature mirrors ``DAggerTrainer`` exactly so the
    run-script swap is one-line. The two type changes (warp env,
    BatchedMPPI) are validated at construction with clear errors.
    """

    def __init__(
        self,
        env: BaseEnv,
        mppi: BatchedMPPI,
        policy,
        cfg: DAggerConfig,
        rng: np.random.Generator | None = None,
    ):
        if not getattr(env, "_use_warp", False):
            raise ValueError(
                "WarpDAggerTrainer requires a Warp-backed env "
                "(env._use_warp must be True). Use the CPU DAggerTrainer "
                "for non-Warp envs."
            )
        if not isinstance(mppi, BatchedMPPI):
            raise ValueError(
                "WarpDAggerTrainer requires a BatchedMPPI controller. "
                "Use the CPU DAggerTrainer with the standard MPPI class."
            )
        super().__init__(env, mppi, policy, cfg, rng=rng)

        self.N = mppi.N

        # N independent CPU MjData twins (one per parallel rollout). Each
        # twin holds an independent simulation state advanced via
        # `_step_one_data`. Cheap (a few MB per Adroit data); plenty of
        # room for N≤32. Same pattern as WarpMPPIGPS.
        self._batch_data: list[mujoco.MjData] = [
            mujoco.MjData(env.model) for _ in range(self.N)
        ]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_states_obs(self) -> tuple[np.ndarray, np.ndarray]:
        """Snapshot per-condition (state, obs) from the N executor MjDatas.

        Mirrors WarpMPPIGPS._get_states_obs. Returns:
            states: (N, nstate) FULLPHYSICS layout.
            obs:    (N, obs_dim) computed via env.state_to_obs (uses
                    sensordata where available so DAPG-obs envs hit the
                    fast path).
        """
        states = np.stack([
            self._mj_getstate(self._batch_data[n]) for n in range(self.N)
        ])
        sensors = np.stack([
            self._batch_data[n].sensordata.copy() for n in range(self.N)
        ])
        # state_to_obs expects (..., nstate); pass (1, N, ...) so envs
        # that index along the last axis work, then squeeze leading 1.
        obs = np.asarray(
            self.env.state_to_obs(states[None], sensors[None])
        )[0]
        return states, obs

    def _mj_getstate(self, data: mujoco.MjData) -> np.ndarray:
        state = np.empty(self.env._nstate)
        mujoco.mj_getState(
            self.env.model, data, state, mujoco.mjtState.mjSTATE_FULLPHYSICS,
        )
        return state

    # ------------------------------------------------------------------
    # Rollout collection (the only overridden piece of the trainer)
    # ------------------------------------------------------------------

    def _collect(
        self,
        n_rollouts: int,
        beta: float,
        seed_base: int,
        episode_len: int | None = None,
        progress_label: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Batched DAgger rollout collection.

        Equivalent rows to ``DAggerTrainer._collect(n_rollouts, beta, ...)``
        but produced by running ``ceil(n_rollouts / N)`` chunks of N
        parallel envs. Each chunk:

          1. Reset N envs to fresh randomized initial states (one
             ``env.reset()`` per slot, distinct seeds).
          2. For ``ep_len`` timesteps:
               a. Gather (states, obs) from the N MjData twins.
               b. ``BatchedMPPI.plan_step(states)`` → expert actions[N, nu].
               c. Per-condition β-mix to choose execution action.
               d. Per-condition ``env.step`` on the MjData twin.
               e. Record (obs[n], expert_action[n]) for each contributing n.

        Returns the same ``(obs, expert_actions)`` flat arrays as the CPU
        trainer.
        """
        ep_len = episode_len if episode_len is not None else self.cfg.episode_len
        obs_rows: list[np.ndarray] = []
        act_rows: list[np.ndarray] = []

        # Process in chunks of N parallel rollouts. The final chunk may
        # be partial — we still run all N envs (warp's nworld is fixed)
        # but only the first `chunk_size` slots contribute rows.
        n_chunks = (n_rollouts + self.N - 1) // self.N
        total_steps = n_rollouts * ep_len

        with tqdm(
            total=total_steps,
            desc=progress_label,
            leave=False,
            dynamic_ncols=True,
            disable=progress_label is None,
        ) as pbar:
            rollout_start = 0
            for chunk_idx in range(n_chunks):
                chunk_size = min(self.N, n_rollouts - rollout_start)

                # ---- Reset N envs to distinct randomized init states ----
                # Per-slot seed so each parallel rollout's randomization
                # is reproducible AND distinct. Mirrors the CPU loop's
                # `np.random.seed(seed_base + ep)` pattern.
                for n in range(self.N):
                    np.random.seed(seed_base + rollout_start + n)
                    self.env.reset()
                    state_n = self.env.get_state().copy()
                    _set_state_one_data(
                        self.env, self._batch_data[n], state_n,
                    )
                self.mppi.reset()

                # ---- Episode loop: all N envs advance per timestep ----
                for t in range(ep_len):
                    states, obs = self._get_states_obs()  # (N, *), (N, obs_dim)

                    # Batched MPPI relabel for ALL N at once.
                    # No prior_fn — DAgger's MPPI is the pure expert.
                    expert_actions, _ = self.mppi.plan_step(states, prior=None)

                    # Per-condition β-mix to choose execution action.
                    # `act_np` is vectorized: passing (N, obs_dim) gives
                    # (N, act_dim) policy actions in one forward pass.
                    if beta == 0.0:
                        # Pure-policy execution; skip the per-step coin
                        # flip and avoid the expert path. Common case
                        # in late DAgger iters and warmup-disabled runs.
                        exec_actions = self.policy.act_np(obs)
                    elif beta == 1.0:
                        # Pure-expert execution (warmup case). No policy
                        # forward at all — skip the GPU round-trip.
                        exec_actions = expert_actions.copy()
                    else:
                        policy_actions = self.policy.act_np(obs)
                        coin = self.rng.random(self.N)
                        # Boolean mask along leading axis: shape (N,) → (N, 1)
                        # broadcast against (N, act_dim).
                        use_mppi = (coin < beta)[:, None]
                        exec_actions = np.where(
                            use_mppi, expert_actions, policy_actions,
                        )

                    # Record (obs, expert_action) for the *contributing*
                    # rollouts only. Slots [chunk_size:N] are simulated
                    # (they're inflight in the warp world batch) but
                    # don't get recorded — keeps row count exactly equal
                    # to the user's `n_rollouts` request.
                    for n in range(chunk_size):
                        obs_rows.append(obs[n].astype(np.float32))
                        act_rows.append(expert_actions[n].astype(np.float32))

                    # Advance all N executor envs (CPU MjData twins).
                    # Sequential loop — N is small (typically ≤32) and
                    # each step is cheap CPU mj_step calls. The
                    # `_step_one_data` helper is shared with WarpMPPIGPS.
                    for n in range(self.N):
                        _step_one_data(
                            self.env, self._batch_data[n], exec_actions[n],
                        )

                    pbar.update(chunk_size)

                rollout_start += self.N

        obs_arr = np.stack(obs_rows, axis=0)
        act_arr = np.stack(act_rows, axis=0)
        if act_arr.ndim == 1:
            act_arr = act_arr[:, None]
        return obs_arr, act_arr
