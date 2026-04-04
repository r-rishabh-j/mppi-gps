"""information theoretic mppi (2018 Williams et al.)"""
import numpy as np 
from src.envs.base import BaseEnv 
from src.utils.config import MPPIConfig
from src.utils.math import (
    compute_weights, effective_sample_size, gaussian_log_prob
)

class MPPI:
    
    def __init__(self, env: BaseEnv, cfg: MPPIConfig):
        self.env = env 
        self.cfg = cfg
        self.K = cfg.K 
        self.H = cfg.H
        self.lam = cfg.lam
        self.sigma = cfg.noise_sigma

        self.nu = env.action_dim
        self.act_low, self.act_high = env.action_bounds

        self.reset()

        self._last_states = None
        self._last_actions = None
        self._last_weights = None
        self._last_costs = None

    def reset(self):
        self.U = np.zeros((self.H, self.nu))

    def plan_step(
            self, 
            state: np.ndarray, 
            prior = None,
    ) -> tuple[np.ndarray, dict]:
        """running one MPPI iteration
        state: current environment state 
        prior: this is an optional callable (states, actions) -> log_prob (K, )
        """
        # sample from q = N(U_nominal, sigma^2 I)
        # noise is eps 
        eps = np.random.randn(self.K, self.H, self.nu) * self.sigma
        U_perturbed = self.U[None, :, :] + eps

        # clamp before rollout (but keep raw eps for unbiased update)
        U_clipped = np.clip(U_perturbed, self.act_low, self.act_high)

        # rollout
        states, costs = self.env.batch_rollout(state, U_clipped)
       
        # compute the prior/ proposal 
        log_prior = None 
        log_proposal = None 

        if prior is not None:
            log_prior = prior(states, U_clipped)
            log_proposal = gaussian_log_prob(U_clipped, self.U, self.sigma)

        # compute weights 
        lam = self.lam 
        weights = compute_weights(costs, lam, log_prior, log_proposal)
        self._last_weights = weights
        # n_eff = effective_sample_size(weights)
        
        # you want to make sure that the weights don't collapse aka lambda is not too small 
        # if lambda is small then the policy isn't exploring
        # if self.cfg.adaptive_lam:
        #     for _ in range(5):
        #         if n_eff < self.cfg.n_eff_threshold:
        #             lam *= 2.0
        #         elif n_eff > 0.75 * self.K:
        #             lam *= 0.5 
        #         else:
        #             break 
        #         lam = np.clip(lam, 0.01, 100.0)
        #         weights = compute_weights(costs, lam, log_prior, log_proposal)
        #         n_eff = effective_sample_size(weights)
        #     self.lam = lam 

        # compute the weighted mean (weight raw perturbations to avoid clipping bias)
        self.U = self.U + np.einsum('k, kha -> ha', weights, eps)
        np.clip(self.U, self.act_low, self.act_high, out = self.U)

        # extract action 
        action = self.U[0].copy()

        # shift horizon 
        self.U[:-1] = self.U[1:]
        self.U[-1] = self.U[-2].copy()

        # store for GPS 
        self._last_states = states
        self._last_actions = U_clipped
        self._last_weights = weights
        self._last_costs = costs

        info = {
            'cost_mean': np.mean(costs),
            'cost_min': np.min(costs),
            # 'n_eff': n_eff,
            'lam': lam,
        }
        return action, info
    
    def get_rollout_data(self) -> dict:
        return {
            'states': self._last_states,
            'actions': self._last_actions,
            'weights': self._last_weights,
            'costs': self._last_costs,
            }
    



                
                





