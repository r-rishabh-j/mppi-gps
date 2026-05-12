"""Warp-batched DAgger: ``BatchedMPPI`` over N parallel envs per timestep.

Only ``_collect`` is overridden; buffer / training / finetune / warmup /
step inherit from ``DAggerTrainer``. ``N = mppi.N`` and the warp env
must have ``nworld = N × cfg.K``. No condition-level done handling
(designed for non-terminating envs like Adroit relocate).
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
    """DAgger driven by ``BatchedMPPI`` (warp env + BatchedMPPI required)."""

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

        # N independent CPU MjData twins, advanced via _step_one_data.
        self._batch_data: list[mujoco.MjData] = [
            mujoco.MjData(env.model) for _ in range(self.N)
        ]

    def _get_states_obs(self) -> tuple[np.ndarray, np.ndarray]:
        """Snapshot ``(states[N, nstate], obs[N, obs_dim])`` from the twins."""
        states = np.stack([
            self._mj_getstate(self._batch_data[n]) for n in range(self.N)
        ])
        sensors = np.stack([
            self._batch_data[n].sensordata.copy() for n in range(self.N)
        ])
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

    def _collect(
        self,
        n_rollouts: int,
        beta: float,
        seed_base: int,
        episode_len: int | None = None,
        progress_label: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Batched rollout in ``ceil(n_rollouts / N)`` chunks of N envs.

        Returns the same ``(obs, expert_actions)`` shape as the CPU path.
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

                    # Per-condition β-mix.
                    if beta == 0.0:
                        exec_actions = self.policy.act_np(obs)
                    elif beta == 1.0:
                        exec_actions = expert_actions.copy()
                    else:
                        policy_actions = self.policy.act_np(obs)
                        coin = self.rng.random(self.N)
                        use_mppi = (coin < beta)[:, None]
                        exec_actions = np.where(
                            use_mppi, expert_actions, policy_actions,
                        )

                    # Record only the first chunk_size rows (final chunk
                    # may be partial); all N envs still step.
                    for n in range(chunk_size):
                        obs_rows.append(obs[n].astype(np.float32))
                        act_rows.append(expert_actions[n].astype(np.float32))

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
