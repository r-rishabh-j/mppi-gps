"""information theoretic mppi (2018 Williams et al.)"""
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
        self.sigma = cfg.noise_sigma
        # open-loop chunk size: number of actions to execute per full replan.
        # Clamp to [1, H] so the chunk always fits inside the nominal trajectory.
        self.open_loop_steps = max(1, min(int(cfg.open_loop_steps), self.H))

        self.nu = env.action_dim
        self.act_low, self.act_high = env.action_bounds

        self.reset()

        self._last_states = None
        self._last_actions = None
        self._last_weights = None
        self._last_costs = None
        self._last_sensordata = None

    def reset(self):
        self.U = np.zeros((self.H, self.nu))
        # Cursor into the current open-loop chunk. 0 = next call must replan.
        self._plan_cursor = 0
        # Cached info from the last replan — returned on open-loop follow-up calls
        # so `info['lam']` etc. remain meaningful instead of going stale/NaN.
        self._last_info: dict | None = None

    def plan_step(
            self,
            state: np.ndarray,
            prior = None,
            dry_run: bool = False,
    ) -> tuple[np.ndarray, dict]:
        """running one MPPI iteration
        state: current environment state
        prior: this is an optional callable (states, actions) -> log_prob (K, )
        dry_run: if True, compute and return the planned action WITHOUT mutating
            any persistent MPPI state (`self.U`, `self._plan_cursor`, `self._last_*`,
            `self.lam`). Used by GPS's DAgger-style relabel path to query a fresh
            teacher action at a visited state without disturbing the executor's
            nominal trajectory or the KL rollout cache. Always takes the full
            replan path (ignores open-loop cadence) — a label call must reflect
            THIS state, not a cached chunk action from a prior-biased plan.

            Works at any `_plan_cursor` value; dry_run neither reads nor advances
            the cursor, so it coexists with an open-loop executor mid-chunk.
            WARNING: under `open_loop_steps > 1`, the executor's non-replan
            timesteps are cheap (cached action serving) but every dry_run label
            call forces a full rollout. You lose the open-loop speedup on the
            label side — wallclock per step becomes ~1 rollout instead of 1/N.
            The combination is correct, just not faster than `open_loop_steps=1`.

        Open-loop cadence: only every `open_loop_steps`-th call does a full MPPI
        replan (sample/rollout/weights/update). Intermediate calls return the next
        action from the already-planned nominal `U` and skip the expensive work.
        When the chunk is exhausted, `U` is shifted left by `open_loop_steps` so
        the next replan uses a correctly-aligned warm start.
        """
        if dry_run:
            self.reset()
        # --- Open-loop follow-up: serve the next action from the current plan.
        # No rollout, no weight update. `prior` is ignored on these calls since
        # it's only meaningful during the weight computation of a replan.
        # Skipped entirely on dry_run (which always does a fresh full replan).

        if self._plan_cursor > 0:
            action = self.U[self._plan_cursor].copy()
            self._plan_cursor += 1
            if self._plan_cursor >= self.open_loop_steps:
                self._shift_horizon(self.open_loop_steps)
                self._plan_cursor = 0
            info = dict(self._last_info) if self._last_info is not None else {}
            info["replanned"] = False
            return action, info

        # sample ε ~ N(0, σ² I), perturb, clamp for rollout
        eps = np.random.randn(self.K, self.H, self.nu) * self.sigma
        U_perturbed = self.U[None, :, :] + eps
        U_clipped = np.clip(U_perturbed, self.act_low, self.act_high)

        # rollout
        states, costs, sensordata = self.env.batch_rollout(state, U_clipped)

        # assemble S_k components (paper's Algorithm 2, γ=λ, Σ=σ²I):
        #    S_k = S_env + λ · Σ_t u_t · ε_{k,t}/σ² + (optional) λ_track · Σ_t ‖a-π‖²
        lam = self.lam
        is_corr = self._is_correction(eps, lam)
        # track = None
        track = prior(states, U_clipped) if prior is not None else None
        S = costs + is_corr + (track if track is not None else 0.0)

        # paper weights: ρ = min_k S_k, w_k = exp(-(S_k - ρ)/λ) / η
        weights, n_eff = self._softmin_weights(S, lam)

        # adaptive λ (not in paper; keeps n_eff in a sensible range)
        if self.cfg.adaptive_lam:
            for _ in range(5):
                if n_eff < self.cfg.n_eff_threshold:
                    lam *= 2.0
                elif n_eff > 0.75 * self.K:
                    lam *= 0.5
                else:
                    break
                lam = float(np.clip(lam, 0.01, 100.0))
                # γ=λ: is_corr depends on λ; track does not
                is_corr = self._is_correction(eps, lam)
                S = costs + is_corr + (track if track is not None else 0.0)
                weights, n_eff = self._softmin_weights(S, lam)
            self.lam = lam

        # weighted update on raw ε (not clipped U) to avoid clipping bias
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
            # Side-effect-free: no self.U / cursor / _last_* / _last_info writes.
            # Caller gets the action + info; MPPI's persistent state is untouched
            # so the next non-dry-run plan_step sees the same nominal as before.
            return action, info

        # --- Persist state for the non-dry-run path ---
        self.U = U_updated

        # Store final (post-adaptation) weights so GPS KL estimation sees the
        # same distribution MPPI actually used to update U.
        self._last_weights = weights

        # Advance the cursor. If open_loop_steps == 1 this also triggers the
        # (legacy) shift-by-1 so next replan's warm start is identical to the
        # pre-open-loop behaviour. For open_loop_steps > 1, subsequent calls
        # will serve U[1..N-1] before the shift-by-N fires.
        self._plan_cursor = 1
        if self._plan_cursor >= self.open_loop_steps:
            self._shift_horizon(self.open_loop_steps)
            self._plan_cursor = 0

        # store for GPS
        self._last_states = states
        self._last_actions = U_clipped
        self._last_weights = weights
        self._last_costs = costs
        self._last_sensordata = sensordata

        self._last_info = info
        return action, info

    def _shift_horizon(self, shift: int) -> None:
        """Shift the nominal trajectory left by `shift` steps, repeating the
        last action to pad. Keeps warm-start consistent after executing an
        open-loop chunk of `shift` actions."""
        if shift <= 0:
            return
        if shift >= self.H:
            # Whole horizon consumed — reset to a zero trajectory rather than
            # repeating a single stale action H times.
            self.U[:] = 0.0
            return
        self.U[:-shift] = self.U[shift:]
        self.U[-shift:] = self.U[-shift - 1]
    
    def _is_correction(self, eps: np.ndarray, lam: float) -> np.ndarray:
        """γ · Σ_t u_t^T Σ^{-1} ε_{k,t} with γ=λ, Σ=σ²I → (K,)."""
        return lam * np.sum(self.U[None, :, :] * eps, axis=(1, 2)) / (self.sigma ** 2)

    def _softmin_weights(self, S: np.ndarray, lam: float) -> tuple[np.ndarray, float]:
        """Paper's weight formula with min-baseline stabilization."""
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
    



                
                




