"""Batched MPPI: N independent control problems, K samples each.

Crams ``B = N × K`` worlds into one graph-captured ``mujoco_warp``
rollout. ``self.U[N, H, nu]``, noise ``(N, K, H, nu)``, perturbed
actions flattened to ``(N*K, H, nu)`` and dispatched via
``env._batch_rollout_warp``; per-condition softmin over the K axis.

Simplifications vs the CPU ``MPPI``:
- ``open_loop_steps`` forced to 1 (always full replan).
- ``adaptive_lam`` disabled — use a hand-tuned ``lam``.
- ``dry_run`` (DAgger relabel) not supported.

A WARNING is emitted if a config sets one of the ignored knobs.
"""
from __future__ import annotations

import warnings
from typing import Callable

import numpy as np

from src.envs.base import BaseEnv
from src.utils.config import MPPIConfig
from src.utils.math import effective_sample_size


class BatchedMPPI:
    """N-condition batched MPPI on the Warp rollout path.

    Args:
        env: Warp-backed env (``_warp_nworld == N * cfg.K``).
        cfg: standard ``MPPIConfig`` (``cfg.K`` is per-condition).
        num_conditions: N independent control problems.

    Caller advances the N CPU env states; this class only handles
    sampling, rollout dispatch, weighting, and ``U`` updates.
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

        # Noise: shared across conditions.
        self._noise_diagonal = cfg.noise_cov is None
        self.noise_cov, self._noise_chol, self._noise_precision = (
            self._build_noise_model(env, cfg)
        )
        self.sigma = np.sqrt(np.diag(self.noise_cov))

        self.U = np.zeros((self.N, self.H, self.nu))

        # Caches for GPS (filled after each plan_step).
        self._last_states: np.ndarray | None = None     # (N, K, H, nstate)
        self._last_actions: np.ndarray | None = None    # (N, K, H, nu)
        self._last_weights: np.ndarray | None = None    # (N, K)
        self._last_costs: np.ndarray | None = None      # (N, K)
        self._last_sensordata: np.ndarray | None = None # (N, K, H, ns)
        self._last_unbiased_weights: np.ndarray | None = None  # (N, K)
        self._last_info: dict | None = None

    def reset(self) -> None:
        """Zero the nominal across all conditions."""
        self.U[...] = 0.0
        self._last_info = None

    def reset_condition(self, n: int) -> None:
        """Zero the nominal for ONE condition (others keep running)."""
        self.U[n] = 0.0

    def plan_step(
        self,
        states: np.ndarray,                         # (N, nstate)
        prior: Callable | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Plan one step for each of N conditions in parallel.

        ``prior`` (optional): ``(states, actions, sensordata) -> (B,)``
        where ``B = N*K``; reshaped to ``(N, K)`` internally.

        Returns ``(actions[N, nu], info)``.
        """
        states = np.asarray(states)
        if states.shape[0] != self.N:
            raise ValueError(
                f"states.shape[0] = {states.shape[0]} != num_conditions {self.N}"
            )

        if self._noise_diagonal:
            eps = np.random.randn(self.N, self.K, self.H, self.nu) * self.sigma
        else:
            standard = np.random.randn(self.N, self.K, self.H, self.nu)
            eps = np.einsum("nkhi,ji->nkhj", standard, self._noise_chol)

        U_perturbed = self.U[:, None, :, :] + eps
        U_clipped = np.clip(U_perturbed, self.act_low, self.act_high)

        # Flatten for warp; worlds [n*K:(n+1)*K] start from condition n.
        B = self.N * self.K
        actions_flat = U_clipped.reshape(B, self.H, self.nu)
        states_flat = np.repeat(states, self.K, axis=0)

        states_b, costs_b, sensordata_b = self.env._batch_rollout_warp(
            states_flat, actions_flat,
        )
        costs_b = np.nan_to_num(costs_b, nan=1e12, posinf=1e12, neginf=1e12)

        nstate = states_b.shape[-1]
        ns = sensordata_b.shape[-1]
        states_nk = states_b.reshape(self.N, self.K, self.H, nstate)
        sensordata_nk = sensordata_b.reshape(self.N, self.K, self.H, ns)
        costs_nk = costs_b.reshape(self.N, self.K)

        is_corr = self._is_correction(eps, self.lam)

        track_nk = None
        if prior is not None:
            track_b = prior(states_b, actions_flat, sensordata_b)
            track_b = np.nan_to_num(track_b, nan=1e12, posinf=1e12, neginf=1e12)
            track_nk = np.asarray(track_b).reshape(self.N, self.K)

        S = costs_nk + is_corr + (track_nk if track_nk is not None else 0.0)

        weights, n_eff = self._softmin_weights_batched(S, self.lam)

        self.U = self.U + np.einsum("nk, nkha -> nha", weights, eps)
        self.U = np.clip(self.U, self.act_low, self.act_high)

        action_now = self.U[:, 0, :].copy()

        # Shift U left by 1 (open_loop_steps=1).
        self.U[:, :-1, :] = self.U[:, 1:, :]
        self.U[:, -1, :] = self.U[:, -2, :]

        self._last_states = states_nk
        self._last_actions = U_clipped
        self._last_weights = weights
        self._last_costs = costs_nk
        self._last_sensordata = sensordata_nk
        # Unbiased: re-softmin without the prior for the KL estimator.
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

    def _is_correction(self, eps: np.ndarray, lam: float) -> np.ndarray:
        """γ · Σ_t U_t^T Σ⁻¹ ε_{n,k,t} → (N, K), γ=λ."""
        if self._noise_diagonal:
            weighted = self.U[:, None, :, :] * eps / (self.sigma ** 2)
            return lam * weighted.sum(axis=(2, 3))

        prec_eps = np.einsum("ij, nkhj -> nkhi", self._noise_precision, eps)
        return lam * (self.U[:, None, :, :] * prec_eps).sum(axis=(2, 3))

    def _softmin_weights_batched(
        self,
        S: np.ndarray,        # (N, K)
        lam: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per-row softmin → (weights[N, K], n_eff[N])."""
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
        """Same semantics as ``MPPI._build_noise_model``."""
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
