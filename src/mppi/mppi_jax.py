"""JAX-native MPPI controller for GPU-accelerated trajectory optimisation.

Same public API as ``MPPI`` (``plan_step``, ``reset``, ``get_rollout_data``)
but the inner loop — noise sampling, batch rollout, cost computation, weight
computation, and trajectory update — runs as a single JIT-compiled function
on GPU.  Only the final action vector is transferred back to CPU.

The policy prior (PyTorch) is handled at the boundary: the JIT'd core runs
without the prior, then if a prior is provided the weights are recomputed
on CPU with the prior included.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from functools import partial

from src.envs.mjx_env import MJXEnv
from src.utils.config import MPPIConfig
from src.utils import math_jax


class MPPIJAX:
    """GPU-accelerated MPPI controller backed by JAX + MJX."""

    def __init__(self, env: MJXEnv, cfg: MPPIConfig):
        self.env = env
        self.cfg = cfg
        self.K = cfg.K
        self.H = cfg.H
        self.sigma = cfg.noise_sigma
        self.nu = env.action_dim

        # Use the same device as the MJX model (CPU fallback on Metal)
        self._device = env._mjx_device

        self.act_low = self._put(np.asarray(env.action_bounds[0], dtype=np.float32))
        self.act_high = self._put(np.asarray(env.action_bounds[1], dtype=np.float32))

        # Mutable state (JAX arrays, updated in-place between calls)
        self.U = self._put(np.zeros((self.H, self.nu), dtype=np.float32))
        self.lam = self._put(np.float32(cfg.lam))
        self._rng = self._put(jax.random.PRNGKey(0))

        # Cache for GPS data extraction
        self._last_states = None
        self._last_actions = None
        self._last_weights = None
        self._last_costs = None
        self._last_sensordata = None

        # JIT the core planning function.  ``env`` is captured as a closure
        # (its JIT'd batch_rollout is already compiled).
        self._jit_plan = jax.jit(self._plan_inner)

    def _put(self, arr):
        """Place array on the MJX device (CPU fallback on Metal).

        Accepts numpy arrays, JAX arrays, or scalars.
        """
        if isinstance(arr, np.ndarray):
            return jax.device_put(arr, self._device)
        # For JAX arrays/scalars, device_put handles transfer
        return jax.device_put(arr, self._device)

    def reset(self):
        self.U = self._put(np.zeros((self.H, self.nu), dtype=np.float32))

    # ------------------------------------------------------------------
    # Core JIT-compiled planning step (no prior — pure JAX)
    # ------------------------------------------------------------------

    def _plan_inner(self, rng, state, U, lam):
        """Pure-functional MPPI iteration.  All on device.

        Args:
            rng:   JAX PRNG key.
            state: (nstate,) initial physics state.
            U:     (H, nu) nominal action sequence.
            lam:   scalar temperature.

        Returns:
            (rng, action, U_new, weights, costs, states, sensordata, U_clipped, eps)
        """
        K, H, nu = self.K, self.H, self.nu

        # 1. Sample noise
        rng, noise_rng = jax.random.split(rng)
        eps = jax.random.normal(noise_rng, (K, H, nu)) * self.sigma

        # 2. Perturb and clip
        U_perturbed = U[None, :, :] + eps
        U_clipped = jnp.clip(U_perturbed, self.act_low, self.act_high)

        # 3. Batch rollout (GPU, JIT'd)
        states, costs, sensordata = self.env.batch_rollout_jax(state, U_clipped)

        # 4. Baseline log ratio: log p(V)/q(V) = -(U · eps) / sigma^2
        baseline_log_ratio = (
            -jnp.sum(U[None, :, :] * eps, axis=(1, 2)) / (self.sigma ** 2)
        )

        # 5. Compute weights (without prior — prior added on CPU if needed)
        weights = math_jax.compute_weights(costs, lam, baseline_log_ratio, None)

        # 6. Adaptive lambda
        if self.cfg.adaptive_lam:
            lam, weights = self._adaptive_lam(costs, baseline_log_ratio, lam)

        # 7. Weighted trajectory update (on raw eps to avoid clipping bias)
        U_new = U + jnp.einsum('k,kha->ha', weights, eps)
        U_new = jnp.clip(U_new, self.act_low, self.act_high)

        # 8. Extract action & shift horizon
        action = U_new[0]
        U_shifted = jnp.concatenate([U_new[1:], U_new[-1:]], axis=0)

        return rng, action, U_shifted, weights, costs, states, sensordata, U_clipped, eps

    def _adaptive_lam(self, costs, log_prior, lam):
        """Adapt temperature to maintain effective sample size.

        Uses ``jax.lax.while_loop`` so it traces once regardless of
        iteration count (no recompilation).
        """
        K = self.K
        threshold = jnp.float32(self.cfg.n_eff_threshold)

        def cond_fn(state):
            lam, weights, i = state
            n_eff = math_jax.effective_sample_size(weights)
            too_low = n_eff < threshold
            too_high = n_eff > 0.75 * K
            return (too_low | too_high) & (i < 5)

        def body_fn(state):
            lam, weights, i = state
            n_eff = math_jax.effective_sample_size(weights)
            lam = jnp.where(n_eff < threshold, lam * 2.0, lam)
            lam = jnp.where(n_eff > 0.75 * K, lam * 0.5, lam)
            lam = jnp.clip(lam, 0.01, 100.0)
            weights = math_jax.compute_weights(costs, lam, log_prior, None)
            return lam, weights, i + 1

        init_weights = math_jax.compute_weights(costs, lam, log_prior, None)
        lam, weights, _ = jax.lax.while_loop(
            cond_fn, body_fn, (lam, init_weights, jnp.int32(0))
        )
        return lam, weights

    # ------------------------------------------------------------------
    # Public API (matches MPPI.plan_step signature)
    # ------------------------------------------------------------------

    def plan_step(self, state, prior=None):
        """Run one MPPI planning iteration.

        Args:
            state: (nstate,) numpy array — current physics state.
            prior: Optional callable(states, actions) → (K,) numpy log-priors.
                   Used by GPS for policy-augmented cost.

        Returns:
            action: (nu,) numpy array — best action to execute.
            info:   dict with cost statistics.
        """
        state_jax = self._put(np.asarray(state, dtype=np.float32))

        (
            self._rng, action, self.U, weights, costs,
            states, sensordata, U_clipped, eps,
        ) = self._jit_plan(self._rng, state_jax, self.U, self.lam)

        # Handle policy prior (PyTorch on CPU) — bridge at boundary
        if prior is not None:
            states_np = np.asarray(states)
            actions_np = np.asarray(U_clipped)
            log_prior_extra = prior(states_np, actions_np)  # (K,) numpy

            # Recompute weights on CPU with prior included
            from src.utils.math import compute_weights, effective_sample_size
            baseline_np = np.asarray(
                -jnp.sum(self.U[None, :, :] * eps, axis=(1, 2))
                / (self.sigma ** 2)
            )
            # Note: self.U here is already the shifted U from _plan_inner,
            # but we need the pre-shift U.  We recover it:
            # U_shifted = [U_new[1:], U_new[-1:]], so U_new[0] = action.
            # Reconstruct U_new = [action, U_shifted[:-1]]
            U_new_np = np.asarray(
                jnp.concatenate([action[None, :], self.U[:-1]], axis=0)
            )
            baseline_np = -np.sum(
                U_new_np[None, :, :] * np.asarray(eps), axis=(1, 2)
            ) / (self.sigma ** 2)
            log_prior_total = baseline_np + log_prior_extra

            costs_np = np.asarray(costs)
            lam_np = float(self.lam)
            weights_np = compute_weights(costs_np, lam_np, log_prior_total, None)

            # Adaptive lambda with prior
            if self.cfg.adaptive_lam:
                n_eff = effective_sample_size(weights_np)
                for _ in range(5):
                    if n_eff < self.cfg.n_eff_threshold:
                        lam_np *= 2.0
                    elif n_eff > 0.75 * self.K:
                        lam_np *= 0.5
                    else:
                        break
                    lam_np = np.clip(lam_np, 0.01, 100.0)
                    weights_np = compute_weights(
                        costs_np, lam_np, log_prior_total, None
                    )
                    n_eff = effective_sample_size(weights_np)
                self.lam = jnp.float32(lam_np)

            # Recompute trajectory update with prior-adjusted weights
            eps_np = np.asarray(eps)
            U_new_np = U_new_np + np.einsum('k,kha->ha', weights_np, eps_np)
            lo, hi = np.asarray(self.act_low), np.asarray(self.act_high)
            U_new_np = np.clip(U_new_np, lo, hi)

            action_np = U_new_np[0].copy()
            self.U = jnp.asarray(np.concatenate([U_new_np[1:], U_new_np[-1:]], axis=0))
            weights = jnp.asarray(weights_np)
        else:
            action_np = np.asarray(action)

        # Store for GPS's get_rollout_data
        self._last_states = states
        self._last_actions = U_clipped
        self._last_weights = weights
        self._last_costs = costs
        self._last_sensordata = sensordata

        info = {
            "cost_mean": float(jnp.mean(costs)),
            "cost_min": float(jnp.min(costs)),
            "lam": float(self.lam),
        }
        return action_np, info

    def get_rollout_data(self) -> dict:
        """Return last rollout data as numpy arrays (for GPS compatibility)."""
        return {
            "states": np.asarray(self._last_states) if self._last_states is not None else None,
            "actions": np.asarray(self._last_actions) if self._last_actions is not None else None,
            "weights": np.asarray(self._last_weights) if self._last_weights is not None else None,
            "costs": np.asarray(self._last_costs) if self._last_costs is not None else None,
            "sensordata": np.asarray(self._last_sensordata) if self._last_sensordata is not None else None,
        }
