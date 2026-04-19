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
    ) -> tuple[np.ndarray, dict]:
        """running one MPPI iteration
        state: current environment state
        prior: this is an optional callable (states, actions) -> log_prob (K, )

        Open-loop cadence: only every `open_loop_steps`-th call does a full MPPI
        replan (sample/rollout/weights/update). Intermediate calls return the next
        action from the already-planned nominal `U` and skip the expensive work.
        When the chunk is exhausted, `U` is shifted left by `open_loop_steps` so
        the next replan uses a correctly-aligned warm start.
        """
        # --- Open-loop follow-up: serve the next action from the current plan.
        # No rollout, no weight update. `prior` is ignored on these calls since
        # it's only meaningful during the weight computation of a replan.
        if self._plan_cursor > 0:
            action = self.U[self._plan_cursor].copy()
            self._plan_cursor += 1
            if self._plan_cursor >= self.open_loop_steps:
                self._shift_horizon(self.open_loop_steps)
                self._plan_cursor = 0
            info = dict(self._last_info) if self._last_info is not None else {}
            info["replanned"] = False
            return action, info

        # --- Full replan path ---
        # sample from q = N(U_nominal, sigma^2 I)
        # noise is eps
        eps = np.random.randn(self.K, self.H, self.nu) * self.sigma
        U_perturbed = self.U[None, :, :] + eps

        # clamp before rollout (but keep raw eps for unbiased update)
        U_clipped = np.clip(U_perturbed, self.act_low, self.act_high)

        # rollout
        states, costs, sensordata = self.env.batch_rollout(state, U_clipped)
       
        # baseline warm start
        # q(V) = N(U, sigma^2 I), p(V) = N(0, sigma^2 I)
        # log p(V) - log q(V) = -(u · eps) / sigma^2 + const
        baseline_log_ratio = -np.sum(self.U[None, :, :] * eps, axis=(1, 2)) / (self.sigma ** 2)

        log_prior = baseline_log_ratio 
        log_proposal = None 

        if prior is not None:
            log_prior = log_prior + prior(states, U_clipped)

        # compute weights
        lam = self.lam
        weights = compute_weights(costs, lam, log_prior, log_proposal)
        self._last_weights = weights
        n_eff = effective_sample_size(weights)

        # you want to make sure that the weights don't collapse aka lambda is not too small 
        # if lambda is small then the policy isn't exploring
        if self.cfg.adaptive_lam:
            for _ in range(5):
                if n_eff < self.cfg.n_eff_threshold:
                    lam *= 2.0
                elif n_eff > 0.75 * self.K:
                    lam *= 0.5 
                else:
                    break 
                lam = np.clip(lam, 0.01, 100.0)
                weights = compute_weights(costs, lam, log_prior, log_proposal)
                n_eff = effective_sample_size(weights)
            self.lam = lam 

        # compute the weighted mean (weight raw perturbations to avoid clipping bias)
        self.U = self.U + np.einsum('k, kha -> ha', weights, eps)
        self.U = np.clip(self.U, self.act_low, self.act_high)

        # extract the first action of the newly planned chunk
        action = self.U[0].copy()

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

        info = {
            'cost_mean': float(np.mean(costs)),
            'cost_min': float(np.min(costs)),
            # 'n_eff': n_eff,
            'lam': float(lam),
            'replanned': True,
        }
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
    
    def get_rollout_data(self) -> dict:
        return {
            'states': self._last_states,
            'actions': self._last_actions,
            'weights': self._last_weights,
            'costs': self._last_costs,
            'sensordata': self._last_sensordata, 
            }
    



                
                




