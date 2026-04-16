"""JAX versions of the numerical utilities in math.py.

These are used by the GPU (MJX) path so that weight computation stays
on-device without a JAX→numpy→JAX round-trip.
"""

import jax
import jax.numpy as jnp


def log_sum_exp(x: jax.Array) -> jax.Array:
    """Numerically stable log-sum-exp."""
    return jax.scipy.special.logsumexp(x)


def compute_weights(
    costs: jax.Array,
    lam: float | jax.Array,
    log_prior: jax.Array | None = None,
    log_proposal: jax.Array | None = None,
) -> jax.Array:
    """Information-theoretic importance weights (Williams et al. 2018).

    Mirrors ``src.utils.math.compute_weights`` but operates on JAX arrays.

    log w_k = -S_k / λ  +  log p(U_k)  -  log q(U_k)
    Returns normalised weights (K,).
    """
    log_w = -costs / lam
    if log_prior is not None:
        log_w = log_w + log_prior
    if log_proposal is not None:
        log_w = log_w - log_proposal
    log_w = log_w - jax.scipy.special.logsumexp(log_w)
    return jnp.exp(log_w)


def effective_sample_size(weights: jax.Array) -> jax.Array:
    """N_eff = 1 / sum(w_k^2).  Range [1, K]."""
    return 1.0 / jnp.sum(weights ** 2)
