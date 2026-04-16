"""MPPI-based Guided Policy Search.

Distills an MPPI controller into a reactive neural network policy via
constrained optimization with BADMM dual updates on the KL divergence.

The high-level loop (Algorithm 1 from the proposal) is:
  1. For each initial condition, run MPPI with a policy-augmented cost
     that biases trajectory samples toward actions the policy can represent.
  2. Collect the executed (obs, action) pairs from each condition.
  3. Distill those pairs into the policy via weighted maximum-likelihood
     (supervised learning on the MPPI teacher's demonstrations).
  4. Measure the KL divergence between the MPPI trajectory distribution
     and the policy, then adjust the BADMM dual variable nu accordingly.
  5. Optionally warm-start MPPI's nominal trajectory from the updated policy.
"""

import numpy as np
import torch
from dataclasses import dataclass, field

from src.envs.base import BaseEnv
from src.mppi.mppi import MPPI
from src.policy.gaussian_policy import GaussianPolicy
from src.utils.config import MPPIConfig, PolicyConfig, GPSConfig
from src.utils.math import weighted_mean_cov


# ---------------------------------------------------------------------------
# Policy prior for MPPI (Eq. 5 from the proposal)
# ---------------------------------------------------------------------------

def make_policy_prior(policy: GaussianPolicy, env: BaseEnv, alpha: float, nu: float):
    """Build a callable ``(states, actions) -> (K,)`` for MPPI's ``prior`` arg.

    In the MPPI importance-weight computation, log-weights are computed as:

        log w_k = -S_k / lambda  +  log_prior

    where S_k is the trajectory cost.  The prior is *added* to log-weights,
    so a positive value *reduces* the effective cost.  We return:

        prior(states, actions) = +alpha * nu * sum_t log pi(u_t | obs_t)

    This biases MPPI toward trajectories whose actions are likely under the
    current policy, which is the "policy-augmented cost" described in Eq. 5.

    Args:
        policy: The current policy network (used in eval / no-grad mode).
        env:    Environment instance — needed for state_to_obs conversion.
        alpha:  Base weight on the policy augmentation term (from GPSConfig).
        nu:     BADMM dual variable — scales how strongly the constraint is
                enforced.  Starts at badmm_init_nu and is updated each iter.
    """
    def prior_fn(states, actions) -> np.ndarray:
        # states:  (K, H, nstate) — full physics states from batch_rollout
        # actions: (K, H, act_dim) — clipped perturbed action sequences
        # Ensure numpy (MPPIJAX may pass JAX arrays from the prior bridge)
        states = np.asarray(states)
        actions = np.asarray(actions)
        K, H, _ = states.shape

        # Convert full physics states to policy-sized observations.
        # E.g. for Acrobot this extracts [qpos, qvel] from [time, qpos, qvel, …]
        obs = np.asarray(env.state_to_obs(states))  # (K, H, obs_dim)

        # Flatten the (K, H) grid into a single batch for the policy network,
        # evaluate log π(u | obs) for every (obs, action) pair, then reshape
        # back and sum over the horizon to get one scalar per trajectory.
        obs_flat = obs.reshape(K * H, -1)
        act_flat = actions.reshape(K * H, -1)
        lp = policy.log_prob_np(obs_flat, act_flat)  # (K*H,)
        lp = lp.reshape(K, H).sum(axis=1)            # (K,) — sum over time

        # Scale by alpha (base weight) and nu (dual variable).
        # Higher alpha  → MPPI samples are more strongly biased toward the policy.
        # Higher nu     → the BADMM constraint is tighter (KL is being penalised more).
        return alpha * nu * lp

    return prior_fn


# ---------------------------------------------------------------------------
# KL estimators
# ---------------------------------------------------------------------------

def _kl_diagonal_gaussian(mu_p, cov_p, mu_q, log_sigma_q):
    """Closed-form KL(N(mu_p, cov_p) || N(mu_q, diag(sigma_q^2))).

    This computes the KL divergence from the MPPI moment-matched distribution
    (the "p" distribution — full covariance) to the policy distribution
    (the "q" distribution — diagonal covariance from the network output).

    The standard formula for KL(p || q) between two multivariate Gaussians is:
        KL = 0.5 * [ tr(Σ_q^{-1} Σ_p)
                    + (μ_q - μ_p)^T Σ_q^{-1} (μ_q - μ_p)
                    - D
                    + ln(det Σ_q / det Σ_p) ]

    Since Σ_q is diagonal (the policy outputs independent per-dimension std),
    the inverse and determinant are cheap to compute.

    Args:
        mu_p:        (D,)   — mean of MPPI weighted distribution
        cov_p:       (D, D) — covariance of MPPI weighted distribution
        mu_q:        (D,)   — policy mean at this observation
        log_sigma_q: (D,)   — policy log-std at this observation
    Returns:
        Scalar KL value (clamped to >= 0 for numerical safety).
    """
    D = mu_p.shape[0]

    # Policy variance per dimension: sigma_q^2 = exp(2 * log_sigma_q)
    var_q = np.exp(2.0 * log_sigma_q)                # (D,)
    inv_var_q = 1.0 / (var_q + 1e-8)                 # avoid division by zero

    # Trace term: tr(Σ_q^{-1} Σ_p).
    # Since Σ_q is diagonal, this simplifies to sum of diag(Σ_p) / var_q.
    trace_term = np.sum(np.diag(cov_p) * inv_var_q)

    # Mahalanobis (mean difference) term: (μ_q - μ_p)^T Σ_q^{-1} (μ_q - μ_p)
    diff = mu_q - mu_p
    mahal = np.sum(diff ** 2 * inv_var_q)

    # Log-determinant ratio: ln(det Σ_q) - ln(det Σ_p)
    # For diagonal Σ_q: ln det = sum of ln(var_q) = 2 * sum(log_sigma_q)
    log_det_q = 2.0 * np.sum(log_sigma_q)
    # For general Σ_p: use slogdet for numerical stability
    sign, log_det_p = np.linalg.slogdet(cov_p + 1e-8 * np.eye(D))
    log_det_ratio = log_det_q - log_det_p

    kl = 0.5 * (trace_term + mahal - D + log_det_ratio)
    return max(kl, 0.0)  # clamp: KL is non-negative by definition


def compute_kl_moment_matched(
    episode_mppi_actions: list[np.ndarray],
    episode_mppi_weights: list[np.ndarray],
    episode_obs: list[np.ndarray],
    policy: GaussianPolicy,
) -> tuple[float, dict]:
    """Moment-matched KL divergence (Eq. 3 from the proposal).

    At each timestep t during the episode:
      1. Fit a Gaussian N(μ_t, Σ_t) to the K weighted MPPI first-step actions
         using weighted_mean_cov.
      2. Query the policy for its distribution π_θ(·|obs_t) = N(μ_π, σ_π²I).
      3. Compute KL(N(μ_t, Σ_t) || π_θ) in closed form.

    Return the KL averaged over all T timesteps.

    This estimator is stable (closed-form, no sampling variance) but assumes
    the MPPI distribution is well-approximated by a single Gaussian (unimodal).

    Args:
        episode_mppi_actions: T-length list, each element is (K, act_dim) —
                              the first-step actions from each MPPI sample.
        episode_mppi_weights: T-length list, each element is (K,) — normalised
                              importance weights from MPPI.
        episode_obs:          T-length list, each element is (obs_dim,) — the
                              observation at that timestep.
        policy:               The current policy network.
    Returns:
        (kl_avg, info_dict) where kl_avg is the mean KL over timesteps.
    """
    kl_sum = 0.0
    T = len(episode_obs)

    for t in range(T):
        actions_t = episode_mppi_actions[t]    # (K, act_dim)
        weights_t = episode_mppi_weights[t]    # (K,)
        obs_t = episode_obs[t]                 # (obs_dim,)

        # Step 1: Moment-match — fit a Gaussian to the weighted MPPI samples.
        # μ = Σ_k w_k u_k,   Σ = Σ_k w_k (u_k - μ)(u_k - μ)^T
        mu_mppi, cov_mppi = weighted_mean_cov(actions_t, weights_t)

        # Step 2: Get the policy's distribution parameters at this observation.
        obs_tensor = torch.as_tensor(obs_t, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            mu_pi, log_sigma_pi = policy(obs_tensor)
        mu_pi = mu_pi.squeeze(0).numpy()
        log_sigma_pi = log_sigma_pi.squeeze(0).numpy()

        # Step 3: Closed-form KL between the two Gaussians.
        kl_sum += _kl_diagonal_gaussian(mu_mppi, cov_mppi, mu_pi, log_sigma_pi)

    kl_avg = kl_sum / max(T, 1)
    return kl_avg, {"kl_sum": kl_sum, "T": T}


def compute_kl_sample_based(
    episode_mppi_actions: list[np.ndarray],
    episode_mppi_weights: list[np.ndarray],
    episode_obs: list[np.ndarray],
    policy: GaussianPolicy,
) -> tuple[float, dict]:
    """Sample-based KL estimate (Eq. 4 from the proposal).

    Instead of fitting a Gaussian, this directly estimates KL from the
    weighted MPPI particles:

        KL ≈ Σ_k w_k [ log w_k  -  log π_θ(u_k | x_k) ]

    This can capture multi-modal MPPI distributions but has higher variance
    than the moment-matched estimator.

    Args / Returns: same as compute_kl_moment_matched.
    """
    kl_sum = 0.0
    T = len(episode_obs)

    for t in range(T):
        actions_t = episode_mppi_actions[t]    # (K, act_dim)
        weights_t = episode_mppi_weights[t]    # (K,)
        obs_t = episode_obs[t]                 # (obs_dim,)

        K = actions_t.shape[0]
        # Broadcast the single observation to all K samples so we can
        # evaluate log π(u_k | obs) in a single batched call.
        obs_batch = np.broadcast_to(obs_t, (K, obs_t.shape[-1]))
        log_pi = policy.log_prob_np(obs_batch, actions_t)  # (K,)

        # Weighted sum of (log w_k - log π)
        log_w = np.log(weights_t + 1e-10)  # +eps to avoid log(0)
        kl_t = np.sum(weights_t * (log_w - log_pi))
        kl_sum += max(kl_t, 0.0)  # clamp for safety

    kl_avg = kl_sum / max(T, 1)
    return kl_avg, {"kl_sum": kl_sum, "T": T}


# ---------------------------------------------------------------------------
# Main GPS class
# ---------------------------------------------------------------------------

@dataclass
class GPSHistory:
    """Training metrics collected during GPS, one entry per iteration."""
    iteration_costs: list[float] = field(default_factory=list)   # mean episode cost
    iteration_kl: list[float] = field(default_factory=list)      # mean KL across conditions
    iteration_nu: list[float] = field(default_factory=list)      # BADMM dual variable
    distill_losses: list[float] = field(default_factory=list)    # last distillation loss


class MPPIGPS:
    """MPPI-GPS: distills an MPPI controller into a neural network policy.

    The training loop iterates between:
      (C-step) Running MPPI with a policy-augmented cost on each initial
               condition to collect expert demonstrations.
      (S-step) Distilling those demonstrations into the policy via weighted
               maximum-likelihood (supervised learning).

    A BADMM dual variable (nu) enforces a KL constraint between the MPPI
    trajectory distribution and the policy, preventing the two from
    diverging too much.
    """

    def __init__(
        self,
        env: BaseEnv,
        mppi_cfg: MPPIConfig,
        policy_cfg: PolicyConfig,
        gps_cfg: GPSConfig,
    ):
        self.env = env
        self.mppi_cfg = mppi_cfg
        self.policy_cfg = policy_cfg
        self.gps_cfg = gps_cfg

        # The global policy network that we are training.
        # obs_dim and act_dim come from the environment.
        self.policy = GaussianPolicy(env.obs_dim, env.action_dim, policy_cfg)

        # BADMM dual variable — controls how strongly the KL constraint
        # is enforced in the policy-augmented cost.  Starts at badmm_init_nu
        # and is adjusted each iteration based on whether KL > or < target.
        self.nu = gps_cfg.badmm_init_nu

        # A single MPPI controller instance, reused across conditions
        # (we reset its nominal U between conditions).
        # Use JAX-backed MPPI when backend="gpu".
        if mppi_cfg.backend == "gpu":
            from src.mppi.mppi_jax import MPPIJAX
            self.mppi = MPPIJAX(env, mppi_cfg)
        else:
            self.mppi = MPPI(env, mppi_cfg)

    # ----- helpers ---------------------------------------------------------

    def _sample_initial_conditions(self, n: int) -> list[np.ndarray]:
        """Sample n initial conditions by resetting the environment.

        Each reset produces a random initial state (the env's reset()
        randomises qpos/qvel).  We capture the full physics state via
        env.get_state() so we can restore it exactly later.

        Returns:
            List of n state vectors, each of shape (nstate,).
        """
        conditions = []
        for _ in range(n):
            self.env.reset()
            conditions.append(self.env.get_state().copy())
        return conditions

    def _distill_epoch(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        weights: np.ndarray,
    ) -> float:
        """Run multiple mini-batch gradient steps to distill MPPI into the policy.

        This is the S-step (supervised step) of GPS.  We train the policy
        to maximise the weighted log-likelihood of the MPPI teacher's actions:

            L = - Σ_i  w_i  log π_θ(a_i | o_i)

        We do `distill_epochs` full passes over the data, shuffling each time,
        with mini-batches of size `distill_batch_size`.

        Args:
            obs:     (N, obs_dim)  — observations from all conditions.
            actions: (N, act_dim)  — MPPI-executed actions.
            weights: (N,)          — importance weights (uniform for now).
        Returns:
            The loss value from the last mini-batch.
        """
        cfg = self.gps_cfg
        N = len(obs)
        indices = np.arange(N)
        last_loss = 0.0
        for _ in range(cfg.distill_epochs):
            np.random.shuffle(indices)
            for start in range(0, N, cfg.distill_batch_size):
                batch = indices[start : start + cfg.distill_batch_size]
                # train_weighted computes -Σ w log π / Σ w  and does one Adam step
                last_loss = self.policy.train_weighted(
                    obs[batch], actions[batch], weights[batch]
                )
        return last_loss

    def _update_badmm(self, kl_value: float):
        """Adjust the BADMM dual variable nu based on the current KL.

        The BADMM update rule is simple:
          - If KL > target → the policy is too far from MPPI → *increase* nu
            to penalise the divergence more in the next iteration's prior.
          - If KL < target → the constraint is easily satisfied → *decrease* nu
            to let MPPI explore more freely.

        nu is clamped to [1e-4, 1e4] to prevent numerical issues.
        """
        target = self.gps_cfg.kl_target
        step = self.gps_cfg.badmm_step_size
        if kl_value > target:
            self.nu *= step      # tighten: increase the penalty
        elif kl_value < target:
            self.nu /= step      # relax: decrease the penalty
        self.nu = np.clip(self.nu, 1e-4, 1e4)

    def _warm_start_mppi(self, initial_state: np.ndarray):
        """Warm-start MPPI's nominal action sequence from the current policy.

        After the policy has been updated (S-step), we roll it out from the
        initial state to produce a sequence of H actions.  This becomes
        MPPI's starting nominal trajectory U for the next C-step, so that
        MPPI doesn't have to search from scratch — it refines around what
        the policy already knows.

        Args:
            initial_state: Full physics state to reset the environment to.
        """
        self.env.reset(state=initial_state)
        actions = []
        for _ in range(self.mppi_cfg.H):
            obs = self.env._get_obs()
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                mu, _ = self.policy(obs_t)
            # Use the mean action (no sampling noise) for a clean warm-start
            action = mu.squeeze(0).numpy()
            actions.append(action)
            self.env.step(action)
        # Overwrite MPPI's nominal trajectory with the policy rollout
        self.mppi.U = np.array(actions)

    # ----- main loop -------------------------------------------------------

    def train(self, num_iterations: int | None = None) -> GPSHistory:
        """Run the full MPPI-GPS training loop.

        Each iteration:
          1. (C-step) For each initial condition, run MPPI with a policy-
             augmented prior for a full episode, collecting demonstrations.
          2. (S-step) Aggregate all (obs, action) pairs and distill them
             into the policy via mini-batched weighted MLE.
          3. Compute the KL divergence between MPPI and the policy.
          4. Update the BADMM dual variable nu.

        Args:
            num_iterations: Override for gps_cfg.num_iterations.
        Returns:
            GPSHistory with per-iteration metrics.
        """
        num_iterations = num_iterations or self.gps_cfg.num_iterations
        cfg = self.gps_cfg
        history = GPSHistory()

        # Sample a fixed set of initial conditions that persist across all
        # iterations.  This ensures the policy is trained to handle a diverse
        # but consistent set of starting states.
        initial_conditions = self._sample_initial_conditions(cfg.num_conditions)

        for iteration in range(num_iterations):
            alpha = cfg.policy_augmented_alpha

            # Accumulators for distillation data (aggregated across conditions)
            all_obs: list[np.ndarray] = []
            all_actions: list[np.ndarray] = []
            all_weights: list[np.ndarray] = []

            # Accumulators for per-timestep KL data (kept per-condition)
            all_kl_actions: list[list[np.ndarray]] = []
            all_kl_weights: list[list[np.ndarray]] = []
            all_kl_obs: list[list[np.ndarray]] = []

            condition_costs: list[float] = []

            # ========== C-STEP: Run MPPI on each initial condition ==========
            for ic_idx, ic_state in enumerate(initial_conditions):
                # Build the policy prior closure for this iteration.
                # This captures the *current* policy weights and nu value.
                prior_fn = make_policy_prior(
                    self.policy, self.env, alpha, self.nu,
                )

                # After the first iteration, seed MPPI's nominal trajectory
                # from the policy so it starts from a better guess.
                if iteration > 0 and cfg.warm_start_policy:
                    self._warm_start_mppi(ic_state)

                # Reset environment to this initial condition
                self.env.reset(state=ic_state)
                # Only reset MPPI's U to zeros on the very first iteration
                # when we're not warm-starting (otherwise keep the warm-start)
                self.mppi.reset() if iteration == 0 and not cfg.warm_start_policy else None

                episode_obs = []
                episode_actions = []
                episode_kl_actions = []   # per-timestep: (K, act_dim)
                episode_kl_weights = []   # per-timestep: (K,)
                episode_cost = 0.0

                for t in range(cfg.episode_length):
                    state = self.env.get_state()
                    obs = self.env._get_obs()

                    # Run one MPPI planning step with the policy-augmented prior.
                    # This samples K trajectories, evaluates cost + prior, and
                    # returns the weighted-mean action.
                    action, info = self.mppi.plan_step(state, prior=prior_fn)

                    # Retrieve the K rollout samples for KL computation.
                    # We only need the first-step actions (what MPPI considered
                    # at this timestep) and their importance weights.
                    rollout_data = self.mppi.get_rollout_data()
                    first_step_actions = np.asarray(rollout_data['actions'][:, 0, :])  # (K, act_dim)
                    weights = np.asarray(rollout_data['weights'])                       # (K,)

                    episode_obs.append(obs.copy())
                    episode_actions.append(action.copy())  # the executed action
                    episode_kl_actions.append(first_step_actions)
                    episode_kl_weights.append(weights.copy())

                    # Step the environment with the MPPI-chosen action
                    _, cost, done, _ = self.env.step(action)
                    episode_cost += cost

                    if done:
                        break

                # Store this condition's data for distillation.
                # We use uniform weights on the executed actions (simple SL).
                # The "teacher" action at each step is the MPPI weighted mean.
                all_obs.append(np.array(episode_obs))
                all_actions.append(np.array(episode_actions))
                all_weights.append(np.ones(len(episode_obs)))

                # Store per-timestep MPPI sample data for KL computation
                all_kl_actions.append(episode_kl_actions)
                all_kl_weights.append(episode_kl_weights)
                all_kl_obs.append(episode_obs)

                condition_costs.append(episode_cost)

            # ========== S-STEP: Distill into policy ==========
            # Concatenate data from all conditions into flat arrays
            obs_flat = np.concatenate(all_obs, axis=0)    # (total_steps, obs_dim)
            act_flat = np.concatenate(all_actions, axis=0) # (total_steps, act_dim)
            w_flat = np.concatenate(all_weights, axis=0)   # (total_steps,)

            # Multiple epochs of mini-batch gradient descent
            loss = self._distill_epoch(obs_flat, act_flat, w_flat)

            # ========== KL computation (average across conditions) ==========
            kl_values = []
            # Select the KL estimator based on config
            kl_fn = (compute_kl_moment_matched
                     if cfg.kl_estimator == "moment_matched"
                     else compute_kl_sample_based)
            for ic_idx in range(cfg.num_conditions):
                kl_val, _ = kl_fn(
                    all_kl_actions[ic_idx],
                    all_kl_weights[ic_idx],
                    all_kl_obs[ic_idx],
                    self.policy,
                )
                kl_values.append(kl_val)
            kl_mean = float(np.mean(kl_values))

            # ========== BADMM dual variable update ==========
            self._update_badmm(kl_mean)

            # ========== Logging ==========
            mean_cost = float(np.mean(condition_costs))
            history.iteration_costs.append(mean_cost)
            history.iteration_kl.append(kl_mean)
            history.iteration_nu.append(self.nu)
            history.distill_losses.append(loss)

            print(
                f"[GPS iter {iteration:3d}]  "
                f"cost={mean_cost:8.2f}  "
                f"distill_loss={loss:.4f}  "
                f"kl={kl_mean:.4f}  "
                f"nu={self.nu:.4f}"
            )

        return history
