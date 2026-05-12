"""Information-theoretic MPPI (Williams et al. 2018)."""
import numpy as np 
from src.envs.base import BaseEnv 
from src.utils.config import MPPIConfig
from src.utils.math import (
    compute_weights, effective_sample_size
)

class MPPI:
    
    def __init__(self, env: BaseEnv, cfg: MPPIConfig):
        self.env = env
        self.cfg = cfg
        self.K = cfg.K
        self.H = cfg.H
        self.lam = cfg.lam
        # Actions executed per full replan; clamped to [1, H].
        self.open_loop_steps = max(1, min(int(cfg.open_loop_steps), self.H))

        self.nu = env.action_dim
        self.act_low, self.act_high = env.action_bounds

        # Noise: diagonal (cfg.noise_cov is None) uses
        # noise_sigma * env.noise_scale per dim; full-Σ uses cfg.noise_cov
        # directly (and ignores noise_sigma / env.noise_scale).
        self._noise_diagonal = cfg.noise_cov is None
        self.noise_cov, self._noise_chol, self._noise_precision = (
            self._build_noise_model(env, cfg)
        )
        self.sigma = np.sqrt(np.diag(self.noise_cov))

        self.reset()

        self._last_states = None
        self._last_actions = None
        self._last_weights = None
        # Weights re-softmin'd over (S − track): what MPPI would produce
        # without the policy prior. Used by KL-adaptive α to measure the
        # unbiased teacher↔policy gap. Equals _last_weights when no prior.
        self._last_unbiased_weights = None
        self._last_costs = None
        self._last_sensordata = None

    def reset(self):
        self.U = np.zeros((self.H, self.nu))
        # 0 = next call must replan; >0 = serve nominal[cursor].
        self._plan_cursor = 0
        # Cached info from the last replan, returned on open-loop follow-ups.
        self._last_info: dict | None = None

    def plan_step(
            self,
            state: np.ndarray,
            prior = None,
            dry_run: bool = False,
    ) -> tuple[np.ndarray, dict]:
        """One MPPI step.

        prior: optional callable (states, actions, sensordata) → cost (K,)
        dry_run: compute action without mutating any persistent state;
            always takes the full replan path (ignores open-loop cadence).
            Used by GPS to relabel without disturbing the executor.

        Open-loop: only every ``open_loop_steps``-th call replans;
        intermediate calls serve from nominal ``U``.
        """
        # Open-loop follow-up: serve the next nominal action; skip rollout.
        if self._plan_cursor > 0:
            action = self.U[self._plan_cursor].copy()
            self._plan_cursor += 1
            if self._plan_cursor >= self.open_loop_steps:
                self._shift_horizon(self.open_loop_steps)
                self._plan_cursor = 0
            info = dict(self._last_info) if self._last_info is not None else {}
            info["replanned"] = False
            return action, info

        # Sample ε ~ N(0, Σ).
        if self._noise_diagonal:
            eps = np.random.randn(self.K, self.H, self.nu) * self.sigma
        else:
            standard = np.random.randn(self.K, self.H, self.nu)
            eps = np.einsum('khi,ji->khj', standard, self._noise_chol)
        U_perturbed = eps if dry_run else self.U[None, :, :] + eps
        U_clipped = np.clip(U_perturbed, self.act_low, self.act_high)

        # Sanitise NaN/Inf costs from diverged samples so they don't
        # poison the whole batch.
        states, costs, sensordata = self.env.batch_rollout(state, U_clipped)
        costs = np.nan_to_num(costs, nan=1e12, posinf=1e12, neginf=1e12)

        # S_k = S_env + λ · Σ_t u_t · Σ⁻¹ ε_{k,t} + track  (γ=λ)
        lam = self.lam
        is_corr = self._is_correction(eps, lam)
        track = (
            prior(states, U_clipped, sensordata) if prior is not None else None
        )
        if track is not None:
            track = np.nan_to_num(track, nan=1e12, posinf=1e12, neginf=1e12)
        S = costs + is_corr + (track if track is not None else 0.0)

        # w_k = exp(-(S_k - min S)/λ) / η
        weights, n_eff = self._softmin_weights(S, lam)

        # Adaptive λ (not in paper) keeps n_eff in a usable range.
        if self.cfg.adaptive_lam:
            for _ in range(5):
                if n_eff < self.cfg.n_eff_threshold:
                    lam *= 2.0
                elif n_eff > 0.75 * self.K:
                    lam *= 0.5
                else:
                    break
                lam = float(np.clip(lam, 0.01, 100.0))
                is_corr = self._is_correction(eps, lam)
                S = costs + is_corr + (track if track is not None else 0.0)
                weights, n_eff = self._softmin_weights(S, lam)
            self.lam = lam

        # Update on raw ε (not clipped U) to avoid clipping bias.
        U_updated = self.U + np.einsum('k, kha -> ha', weights, eps)
        U_updated = np.clip(U_updated, self.act_low, self.act_high)

        action = U_updated[0].copy()

        info = {
            'cost_mean': float(np.mean(costs)),
            'cost_min': float(np.min(costs)),
            # 'n_eff': n_eff,
            'lam': float(lam),
            'replanned': True,
        }

        if dry_run:
            return action, info

        self.U = U_updated
        self._last_weights = weights

        self._plan_cursor = 1
        if self._plan_cursor >= self.open_loop_steps:
            self._shift_horizon(self.open_loop_steps)
            self._plan_cursor = 0

        self._last_states = states
        self._last_actions = U_clipped
        self._last_weights = weights
        # Unbiased weights: re-softmin without the prior contribution.
        if track is not None:
            self._last_unbiased_weights, _ = self._softmin_weights(
                S - track, lam,
            )
        else:
            self._last_unbiased_weights = weights
        self._last_costs = costs
        self._last_sensordata = sensordata

        self._last_info = info
        return action, info

    def _build_noise_model(
        self, env: BaseEnv, cfg: MPPIConfig,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(cov, chol, precision)`` for the action noise.

        Diagonal: variance = (noise_sigma * env.noise_scale)² per dim.
        Full-Σ: cfg.noise_cov used directly; validated as PSD here so a
        bad matrix fails at construction.
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
            cfg.noise_sigma * env.noise_scale, dtype=np.float64
        )
        assert sigma_per_dim.shape == (self.nu,), (
            f"noise_sigma * env.noise_scale must be (action_dim={self.nu},), "
            f"got {sigma_per_dim.shape}"
        )
        cov = np.diag(sigma_per_dim ** 2)
        chol = np.diag(sigma_per_dim)
        precision = np.diag(1.0 / sigma_per_dim ** 2)
        return cov, chol, precision

    def _shift_horizon(self, shift: int) -> None:
        """Shift U left by ``shift``, padding with the last action."""
        if shift <= 0:
            return
        if shift >= self.H:
            self.U[:] = 0.0
            return
        self.U[:-shift] = self.U[shift:]
        self.U[-shift:] = self.U[-shift - 1]

    def _is_correction(self, eps: np.ndarray, lam: float) -> np.ndarray:
        """γ · Σ_t u_t^T Σ⁻¹ ε_{k,t} with γ=λ → (K,)."""
        if self._noise_diagonal:
            weighted = self.U[None, :, :] * eps / (self.sigma ** 2)
            return lam * weighted.sum(axis=(1, 2))
        prec_eps = np.einsum('ij,ktj->kti', self._noise_precision, eps)
        return lam * (self.U[None, :, :] * prec_eps).sum(axis=(1, 2))

    def _softmin_weights(self, S: np.ndarray, lam: float) -> tuple[np.ndarray, float]:
        """w_k = exp(-(S_k - min S)/λ) / η, with min-baseline stabilization."""
        rho = np.min(S)
        unnorm = np.exp(-(S - rho) / lam)
        eta = np.sum(unnorm)
        weights = unnorm / eta
        return weights, effective_sample_size(weights)
    
    def get_rollout_data(self) -> dict:
        return {
            'states': self._last_states,
            'actions': self._last_actions,
            'weights': self._last_weights,
            'costs': self._last_costs,
            'sensordata': self._last_sensordata, 
            }
