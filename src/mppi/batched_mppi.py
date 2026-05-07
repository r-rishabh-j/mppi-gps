"""Batched MPPI: N independent control problems, K samples each.

The CPU `MPPI` controller in `src/mppi/mppi.py` plans for **one** state at
a time and feeds K samples into `env.batch_rollout`. With `mujoco_warp`
we can bigger: cram `B = N × K` worlds into one graph-captured rollout
and plan for N conditions in parallel, with K MPPI samples per condition.

This is the lever for GPS speedup on Adroit: the C-step's per-condition
loop (sequential N=5 episodes × 500 steps × MPPI replan) collapses into
a single batched advance per timestep — every plan_step rolls out 5×K
worlds at once on the GPU, every env-step advances 5 CPU envs in
parallel.

----------------------------------------------------------------------
Architecture
----------------------------------------------------------------------

* ``self.U[N, H, nu]``        — one nominal trajectory per condition.
* ``noise[N, K, H, nu]``      — per-condition independent samples.
* ``actions[N, K, H, nu]``    — `U[:, None, :, :] + noise`.
* Flatten to ``(N*K, H, nu)`` and pass to ``env._batch_rollout_warp``,
  along with per-world initial states ``(N*K, nstate)`` built by tiling
  each row of ``states[N, nstate]`` K times.
* Reshape outputs back to ``(N, K, H, *)``; per-condition softmin over
  the K axis gives ``weights[N, K]``; ``U[i] += einsum('k,kha->ha',
  w[i], eps[i])``.

Caches `_last_states / _last_actions / _last_weights` with a leading N
axis so GPS's KL-adaptive estimator can aggregate per-condition.

----------------------------------------------------------------------
Simplifications vs. the CPU `MPPI` (kept tight to ship a working v1)
----------------------------------------------------------------------

* ``open_loop_steps`` is forced to 1 — every plan_step is a full replan.
  Open-loop chunks would need a per-condition cursor; defer until
  measured-need.
* ``adaptive_lam`` is disabled. The CPU path adapts λ to keep n_eff in a
  band; that adaptation is per-condition and serializes the inner loop.
  Use a hand-tuned ``lam`` instead.
* ``dry_run`` (DAgger relabel) is not supported. The trainer can run a
  separate plain-prior plan_step if it needs unbiased labels — costs an
  extra graph launch but is straightforward.

These are noted in `WARNING` log lines if the corresponding cfg values
are set so the user isn't silently downgraded.
"""
from __future__ import annotations

import warnings
from typing import Callable

import numpy as np

from src.envs.base import BaseEnv
from src.utils.config import MPPIConfig
from src.utils.math import effective_sample_size


class BatchedMPPI:
    """N-condition batched MPPI for use with the Warp rollout path.

    Constructor takes:
      * ``env``: a Warp-backed env (must expose ``_batch_rollout_warp``
        and have ``_warp_nworld == num_conditions * cfg.K``).
      * ``cfg``: standard ``MPPIConfig``. ``cfg.K`` is per-condition.
      * ``num_conditions``: N — independent control problems.

    Caller is responsible for advancing the N CPU env states (typically
    via ``mujoco.mj_step`` on N MjData twins). This class only handles
    sampling, rollout dispatch, weighting, and U updates.
    """

    def __init__(
        self,
        env: BaseEnv,
        cfg: MPPIConfig,
        num_conditions: int,
    ):
        if not getattr(env, "_use_warp", False):
            raise ValueError(
                "BatchedMPPI requires a Warp-backed env "
                "(env._use_warp must be True). Use the CPU `MPPI` controller "
                "for non-Warp envs."
            )
        if num_conditions <= 0:
            raise ValueError(f"num_conditions must be positive, got {num_conditions}")

        expected_nworld = num_conditions * cfg.K
        if env._warp_nworld != expected_nworld:
            raise ValueError(
                f"Env was constructed with nworld={env._warp_nworld}, "
                f"but BatchedMPPI needs num_conditions * K = "
                f"{num_conditions} * {cfg.K} = {expected_nworld}. "
                "Re-instantiate the env with the correct nworld."
            )

        self.env = env
        self.cfg = cfg
        self.N = int(num_conditions)
        self.K = int(cfg.K)
        self.H = int(cfg.H)
        self.lam = float(cfg.lam)
        self.nu = env.action_dim
        self.act_low, self.act_high = env.action_bounds

        # Loud opt-out warnings for cfg knobs we don't honour.
        if getattr(cfg, "open_loop_steps", 1) > 1:
            warnings.warn(
                f"BatchedMPPI ignores open_loop_steps={cfg.open_loop_steps}; "
                "every plan_step does a full replan. Set open_loop_steps=1 "
                "in your config to silence this warning.",
                stacklevel=2,
            )
        if getattr(cfg, "adaptive_lam", False):
            warnings.warn(
                "BatchedMPPI ignores adaptive_lam — using fixed lam={:.3f}. "
                "Tune `lam` manually for the warp path.".format(self.lam),
                stacklevel=2,
            )

        # ---- Noise model: shared across conditions (same Σ for every
        # condition). Mirrors CPU MPPI._build_noise_model. ----
        self._noise_diagonal = cfg.noise_cov is None
        self.noise_cov, self._noise_chol, self._noise_precision = (
            self._build_noise_model(env, cfg)
        )
        self.sigma = np.sqrt(np.diag(self.noise_cov))

        # ---- Per-condition nominal trajectory ----
        self.U = np.zeros((self.N, self.H, self.nu))

        # ---- Caches for GPS (filled after each plan_step) ----
        self._last_states: np.ndarray | None = None     # (N, K, H, nstate)
        self._last_actions: np.ndarray | None = None    # (N, K, H, nu)
        self._last_weights: np.ndarray | None = None    # (N, K)
        self._last_costs: np.ndarray | None = None      # (N, K)
        self._last_sensordata: np.ndarray | None = None # (N, K, H, ns)
        self._last_unbiased_weights: np.ndarray | None = None  # (N, K)
        self._last_info: dict | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Zero the nominal across all conditions. Call when the env(s) reset."""
        self.U[...] = 0.0
        self._last_info = None

    def reset_condition(self, n: int) -> None:
        """Zero the nominal for ONE condition. Use when condition ``n``
        terminates and is re-initialized while others are still running."""
        self.U[n] = 0.0

    def plan_step(
        self,
        states: np.ndarray,                         # (N, nstate)
        prior: Callable | None = None,              # see make_policy_prior
    ) -> tuple[np.ndarray, dict]:
        """Plan one step for each of the N conditions in parallel.

        Args:
            states: per-condition initial states, shape ``(N, nstate)``.
            prior:  optional callable ``(states, actions, sensordata) -> (B,)``
                    where ``B = N*K`` — caller's prior is evaluated on the
                    flattened batch; we reshape back to (N, K) internally.

        Returns:
            actions: shape ``(N, nu)`` — best per-condition action to execute.
            info:    aggregated diagnostics (per-condition arrays inside).
        """
        states = np.asarray(states)
        if states.shape[0] != self.N:
            raise ValueError(
                f"states.shape[0] = {states.shape[0]} != num_conditions {self.N}"
            )

        # ---- Sample noise (N, K, H, nu) ----
        if self._noise_diagonal:
            eps = np.random.randn(self.N, self.K, self.H, self.nu) * self.sigma
        else:
            standard = np.random.randn(self.N, self.K, self.H, self.nu)
            eps = np.einsum("nkhi,ji->nkhj", standard, self._noise_chol)

        U_perturbed = self.U[:, None, :, :] + eps                # (N, K, H, nu)
        U_clipped = np.clip(U_perturbed, self.act_low, self.act_high)

        # ---- Flatten for warp: (N*K, H, nu) and (N*K, nstate) ----
        B = self.N * self.K
        actions_flat = U_clipped.reshape(B, self.H, self.nu)
        # Tile each row of `states` K times so worlds [n*K : (n+1)*K]
        # all start from condition n's initial state.
        states_flat = np.repeat(states, self.K, axis=0)          # (N*K, nstate)

        # ---- Rollout via warp ----
        states_b, costs_b, sensordata_b = self.env._batch_rollout_warp(
            states_flat, actions_flat,
        )
        # Sanitize NaNs (one diverged sample shouldn't poison its
        # condition's softmin). Mirrors CPU MPPI's sanitization.
        costs_b = np.nan_to_num(costs_b, nan=1e12, posinf=1e12, neginf=1e12)

        # ---- Reshape back to (N, K, ...) ----
        nstate = states_b.shape[-1]
        ns = sensordata_b.shape[-1]
        states_nk = states_b.reshape(self.N, self.K, self.H, nstate)
        sensordata_nk = sensordata_b.reshape(self.N, self.K, self.H, ns)
        costs_nk = costs_b.reshape(self.N, self.K)

        # ---- IS correction: per-condition (N, K). Same Σ so we can broadcast. ----
        is_corr = self._is_correction(eps, self.lam)             # (N, K)

        # ---- Policy prior on flat batch, reshape (B,) → (N, K) ----
        track_nk = None
        if prior is not None:
            track_b = prior(states_b, actions_flat, sensordata_b)
            track_b = np.nan_to_num(track_b, nan=1e12, posinf=1e12, neginf=1e12)
            track_nk = np.asarray(track_b).reshape(self.N, self.K)

        S = costs_nk + is_corr + (track_nk if track_nk is not None else 0.0)

        # ---- Per-condition softmin: independent per row of (N, K) ----
        weights, n_eff = self._softmin_weights_batched(S, self.lam)  # (N, K), (N,)

        # ---- Per-condition weighted U update ----
        # einsum 'nk,nkha->nha' — for each condition n, weighted sum of eps[n].
        self.U = self.U + np.einsum("nk, nkha -> nha", weights, eps)
        self.U = np.clip(self.U, self.act_low, self.act_high)

        action_now = self.U[:, 0, :].copy()                       # (N, nu)

        # ---- Open-loop chunk advance: shift U left by 1 (open_loop_steps=1) ----
        self.U[:, :-1, :] = self.U[:, 1:, :]
        self.U[:, -1, :] = self.U[:, -2, :]

        # ---- Stash for GPS / KL estimator ----
        self._last_states = states_nk
        self._last_actions = U_clipped
        self._last_weights = weights
        self._last_costs = costs_nk
        self._last_sensordata = sensordata_nk
        # Unbiased weights: re-softmin without the prior contribution so
        # downstream KL estimator sees the cost-only teacher.
        if track_nk is not None:
            ub_S = S - track_nk
            self._last_unbiased_weights, _ = self._softmin_weights_batched(
                ub_S, self.lam,
            )
        else:
            self._last_unbiased_weights = weights

        info = {
            "cost_mean": float(np.mean(costs_nk)),
            "cost_min": float(np.min(costs_nk)),
            "cost_per_condition": costs_nk.min(axis=1).tolist(),
            "n_eff_mean": float(np.mean(n_eff)),
            "lam": self.lam,
            "replanned": True,
        }
        self._last_info = info
        return action_now, info

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _is_correction(self, eps: np.ndarray, lam: float) -> np.ndarray:
        """γ · Σ_t U_t^T Σ⁻¹ ε_{n,k,t} → (N, K). γ = λ.

        Diagonal: each U[n] aligns with eps[n] across (K, H, nu); broadcast
        multiply through.
        Full Σ: precision @ eps via einsum, then per-step dot with U.
        """
        if self._noise_diagonal:
            # U[None, :, :] : (1, N, H, nu) → (N, 1, H, nu) for broadcast with eps
            weighted = self.U[:, None, :, :] * eps / (self.sigma ** 2)  # (N, K, H, nu)
            return lam * weighted.sum(axis=(2, 3))                       # (N, K)

        prec_eps = np.einsum(
            "ij, nkhj -> nkhi", self._noise_precision, eps,
        )                                                                # (N, K, H, nu)
        return lam * (self.U[:, None, :, :] * prec_eps).sum(axis=(2, 3))  # (N, K)

    def _softmin_weights_batched(
        self,
        S: np.ndarray,        # (N, K)
        lam: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per-row softmin with min-baseline stabilization.

        Returns:
            weights: (N, K), each row sums to 1.
            n_eff:   (N,), per-condition effective sample size.
        """
        rho = S.min(axis=1, keepdims=True)                  # (N, 1)
        unnorm = np.exp(-(S - rho) / lam)                   # (N, K)
        eta = unnorm.sum(axis=1, keepdims=True)             # (N, 1)
        weights = unnorm / eta                              # (N, K)
        n_eff = 1.0 / np.clip((weights ** 2).sum(axis=1), 1e-12, None)
        return weights, n_eff

    def _build_noise_model(
        self,
        env: BaseEnv,
        cfg: MPPIConfig,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Same signature / semantics as MPPI._build_noise_model.

        Lifted here rather than reused-via-import to keep BatchedMPPI a
        standalone module (no inheritance from MPPI; their plan_step
        signatures already differ enough).
        """
        if cfg.noise_cov is not None:
            cov = np.asarray(cfg.noise_cov, dtype=np.float64)
            if cov.shape != (self.nu, self.nu):
                raise ValueError(
                    f"noise_cov must have shape ({self.nu}, {self.nu}), "
                    f"got {cov.shape}"
                )
            if not np.allclose(cov, cov.T, atol=1e-8):
                raise ValueError("noise_cov must be symmetric")
            try:
                chol = np.linalg.cholesky(cov)
            except np.linalg.LinAlgError as exc:
                raise ValueError(
                    "noise_cov must be positive-definite (cholesky failed)"
                ) from exc
            precision = np.linalg.inv(cov)
            return cov, chol, precision

        sigma_per_dim = np.asarray(
            cfg.noise_sigma * env.noise_scale, dtype=np.float64,
        )
        assert sigma_per_dim.shape == (self.nu,), (
            f"noise_sigma * env.noise_scale must be (action_dim={self.nu},), "
            f"got {sigma_per_dim.shape}"
        )
        cov = np.diag(sigma_per_dim ** 2)
        chol = np.diag(sigma_per_dim)
        precision = np.diag(1.0 / sigma_per_dim ** 2)
        return cov, chol, precision

    def get_rollout_data(self) -> dict:
        return {
            "states": self._last_states,
            "actions": self._last_actions,
            "weights": self._last_weights,
            "costs": self._last_costs,
            "sensordata": self._last_sensordata,
        }
