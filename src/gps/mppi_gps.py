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

from pathlib import Path

import numpy as np
import torch
from dataclasses import dataclass, field
from tqdm.auto import tqdm

from src.envs.base import BaseEnv
from src.mppi.mppi import MPPI
from src.policy.gaussian_policy import GaussianPolicy
from src.utils.config import MPPIConfig, PolicyConfig, GPSConfig
from src.utils.evaluation import evaluate_policy
from src.utils.experiment import copy_as, save_checkpoint
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
        device = getattr(policy, "_device", torch.device("cpu"))
        obs_tensor = torch.as_tensor(obs_t, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            mu_pi, log_sigma_pi = policy(obs_tensor)
        mu_pi = mu_pi.squeeze(0).cpu().numpy()
        log_sigma_pi = log_sigma_pi.squeeze(0).cpu().numpy()

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
    # Mean C-step MPPI rollout cost (teacher-under-policy-prior) — diagnostic only.
    iteration_costs: list[float] = field(default_factory=list)
    # Greedy-policy eval cost; NaN on iterations where evaluation was skipped.
    # This is what `best.pt` is selected on.
    iteration_eval_costs: list[float] = field(default_factory=list)
    iteration_kl: list[float] = field(default_factory=list)      # mean KL across conditions
    iteration_nu: list[float] = field(default_factory=list)      # BADMM dual variable
    distill_losses: list[float] = field(default_factory=list)    # last distillation loss
    best_iter: int = -1
    best_cost: float = float("inf")


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
        device: torch.device | str | None = None,
    ):
        self.env = env
        self.mppi_cfg = mppi_cfg
        self.policy_cfg = policy_cfg
        self.gps_cfg = gps_cfg

        # The global policy network that we are training.
        # obs_dim and act_dim come from the environment.
        self.policy = GaussianPolicy(
            env.obs_dim,
            env.action_dim,
            policy_cfg,
            device=device,
            action_bounds=env.action_bounds,
        )

        # BADMM dual variable — controls how strongly the KL constraint
        # is enforced in the policy-augmented cost.  Starts at badmm_init_nu
        # and is adjusted each iteration based on whether KL > or < target.
        self.nu = gps_cfg.badmm_init_nu

        # A single MPPI controller instance, reused across conditions
        # (we reset its nominal U between conditions).
        self.mppi = MPPI(env, mppi_cfg)

        # Cross-iteration episode replay buffer. Each entry is one whole
        # sub-episode (dict with 'obs', 'actions', 'weights'), physically
        # contiguous (split at every `done` boundary during C-step). Empty
        # and unused when `gps_cfg.distill_buffer_cap == 0`.
        self._episode_buffer: list[dict] = []

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
                if cfg.distill_loss == "mse":
                    last_loss = self.policy.mse_step(obs[batch], actions[batch])
                else:
                    # train_weighted computes -Σ w log π / Σ w  and does one Adam step
                    last_loss = self.policy.train_weighted(
                        obs[batch], actions[batch], weights[batch]
                    )
        return last_loss

    # def _train_step_mse(self, obs: np.ndarray, actions: np.ndarray) -> float:
    #     """Plain MSE on the policy mean — used when distill_loss='mse' (BC-style).

    #     Reuse GaussianPolicy.mse_step so the running observation normalizer is
    #     updated exactly the same way as the standalone BC and DAgger paths.
    #     """
    #     return self.policy.mse_step(obs, actions)

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
        done = False
        for _ in range(self.mppi_cfg.H):
            if done:
                # Env has terminated — pad the remaining
                # horizon with zeros so we don't feed post-terminal garbage
                # states into the policy.
                actions.append(np.zeros(self.env.action_dim))
                continue
            obs = self.env._get_obs()
            action = self.policy.act_np(obs)
            actions.append(action)
            _, _, done, _ = self.env.step(action)
        self.mppi.U = np.array(actions)

    # ----- main loop -------------------------------------------------------

    def train(
        self,
        num_iterations: int | None = None,
        run_dir: Path | None = None,
    ) -> GPSHistory:
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
            run_dir: If set, save per-iter wrapped checkpoints as
                `{run_dir}/iter_{k:03d}.pt` and copy the best-by-policy-eval
                to `{run_dir}/best.pt`. "Best" is decided from the greedy
                policy's eval cost (see `gps_cfg.n_eval_eps` / `eval_ep_len`
                / `eval_every`), not from the C-step MPPI rollout cost.
        Returns:
            GPSHistory with per-iteration metrics.
        """
        num_iterations = num_iterations or self.gps_cfg.num_iterations
        cfg = self.gps_cfg
        history = GPSHistory()

        if run_dir is not None:
            run_dir = Path(run_dir)
            run_dir.mkdir(parents=True, exist_ok=True)

        # Sample a fixed set of initial conditions that persist across all
        # iterations.  This ensures the policy is trained to handle a diverse
        # but consistent set of starting states.
        initial_conditions = self._sample_initial_conditions(cfg.num_conditions)

        skip_kl = cfg.disable_kl_constraint

        outer_bar = tqdm(range(num_iterations), desc="GPS", unit="iter")
        for iteration in outer_bar:
            alpha = cfg.policy_augmented_alpha

            # Accumulators for distillation data (aggregated across conditions)
            all_obs: list[np.ndarray] = []
            all_actions: list[np.ndarray] = []
            all_weights: list[np.ndarray] = []

            # Accumulators for per-timestep KL data (kept per-condition).
            # Skipped entirely when the KL constraint is disabled — this also
            # avoids storing K * H * act_dim floats per timestep.
            all_kl_actions: list[list[np.ndarray]] = []
            all_kl_weights: list[list[np.ndarray]] = []
            all_kl_obs: list[list[np.ndarray]] = []

            condition_costs: list[float] = []

            # ========== C-STEP: Run MPPI on each initial condition ==========
            # Nested progress bar over total MPPI steps (num_conditions * episode_length).
            # `leave=False` so it clears after each iteration and the outer bar stays clean.
            c_step_total = cfg.num_conditions * cfg.episode_length
            c_bar = tqdm(
                total=c_step_total,
                desc=f"  iter {iteration:3d} C-step",
                unit="step",
                leave=False,
            )
            for ic_idx, ic_state in enumerate(initial_conditions):
                # Build the policy prior closure for this iteration.
                # This captures the *current* policy weights and nu value.
                prior_fn = make_policy_prior(
                    self.policy, self.env, alpha, self.nu,
                )

                # After the first iteration, seed MPPI's nominal trajectory
                # from the policy so it starts from a better guess.  On iter 0
                # (or if warm-starting is off) reset U to zeros so nominal
                # state doesn't leak across initial conditions.
                if iteration > 0 and cfg.warm_start_policy:
                    self._warm_start_mppi(ic_state)
                else:
                    self.mppi.reset()

                # Reset environment to this initial condition
                self.env.reset(state=ic_state)

                # --- Sub-episode accumulators (one sub-episode closes at each
                #     `done`; without auto_reset there's exactly one sub-episode
                #     per condition). Each closed sub-episode is a contiguous
                #     slice of (obs, action) — never spans a reset.
                condition_sub_episodes: list[dict] = []
                cur_obs: list[np.ndarray] = []
                cur_actions: list[np.ndarray] = []
                # KL data stays aggregated per-condition (the KL estimator
                # consumes a flat per-timestep list); sub-episode boundaries
                # don't matter for the KL average.
                episode_kl_actions = []   # per-timestep: (K, act_dim)
                episode_kl_weights = []   # per-timestep: (K,)
                episode_obs_for_kl: list[np.ndarray] = []  # per-timestep obs
                episode_cost = 0.0

                def _flush_sub_episode():
                    if not cur_obs:
                        return
                    condition_sub_episodes.append({
                        "obs": np.array(cur_obs),
                        "actions": np.array(cur_actions),
                        "weights": np.ones(len(cur_obs)),
                    })
                    cur_obs.clear()
                    cur_actions.clear()

                for t in range(cfg.episode_length):
                    state = self.env.get_state()
                    obs = self.env._get_obs()

                    # Run one MPPI planning step with the policy-augmented prior.
                    # This samples K trajectories, evaluates cost + prior, and
                    # returns the weighted-mean action.
                    action, info = self.mppi.plan_step(state, prior=prior_fn)

                    cur_obs.append(obs.copy())
                    cur_actions.append(action.copy())  # the executed action

                    if not skip_kl:
                        # Retrieve the K rollout samples for KL computation.
                        # Copy both arrays: MPPI reuses the _last_* slots on
                        # the next plan_step, and we must not hold views into
                        # buffers it may later overwrite.
                        rollout_data = self.mppi.get_rollout_data()
                        first_step_actions = np.array(rollout_data['actions'][:, 0, :])
                        weights = np.array(rollout_data['weights'])
                        episode_kl_actions.append(first_step_actions)
                        episode_kl_weights.append(weights)
                        episode_obs_for_kl.append(obs.copy())

                    # Step the environment with the MPPI-chosen action
                    _, cost, done, _ = self.env.step(action)
                    episode_cost += cost
                    c_bar.update(1)

                    if done and t < cfg.episode_length - 1:
                        # Close the current sub-episode BEFORE resetting so its
                        # contents reflect a single contiguous rollout.
                        _flush_sub_episode()
                        if cfg.auto_reset:
                            # Terminating env (hopper, etc.): re-seed to a fresh random
                            # init and keep collecting until episode_length steps are
                            # taken. MPPI nominal U is also reset so it doesn't carry
                            # post-fall state. Without this, the C-step would keep
                            # stepping MuJoCo from a fallen state and the S-step would
                            # distill on mostly-post-fall (obs, action) pairs.
                            self.env.reset()
                            self.mppi.reset()
                            continue
                        # No auto_reset: cut the episode short here. Pad the progress
                        # bar so totals stay consistent.
                        c_bar.update(cfg.episode_length - (t + 1))
                        break

                # Flush the tail sub-episode (the loop's last chunk, whether
                # it ran to `episode_length` or ended on the break above).
                _flush_sub_episode()

                # Route this condition's sub-episodes either into the buffer
                # (DAgger-style cross-iteration replay) or into the per-iter
                # lists (current on-policy-per-iter behaviour).
                if cfg.distill_buffer_cap > 0:
                    for sub_ep in condition_sub_episodes:
                        self._episode_buffer.append(sub_ep)
                    # FIFO eviction by rows — pop whole episodes from the front
                    # until total rows ≤ cap. Mirror DAgger's guard
                    # (`len(buffer) > 1`) so we never drop the *only* episode
                    # even if it alone exceeds the cap; otherwise the S-step
                    # would run with an empty buffer.
                    total_rows = sum(len(ep["obs"]) for ep in self._episode_buffer)
                    while (total_rows > cfg.distill_buffer_cap
                           and len(self._episode_buffer) > 1):
                        dropped = self._episode_buffer.pop(0)
                        total_rows -= len(dropped["obs"])
                else:
                    for sub_ep in condition_sub_episodes:
                        all_obs.append(sub_ep["obs"])
                        all_actions.append(sub_ep["actions"])
                        all_weights.append(sub_ep["weights"])

                if not skip_kl:
                    # Store per-timestep MPPI sample data for KL computation.
                    # KL is always current-iter-only regardless of the distill
                    # buffer (we want KL(MPPI_now || policy_now), not against
                    # stale rollouts).
                    all_kl_actions.append(episode_kl_actions)
                    all_kl_weights.append(episode_kl_weights)
                    all_kl_obs.append(episode_obs_for_kl)

                condition_costs.append(episode_cost)
                c_bar.set_postfix(
                    ic=f"{ic_idx + 1}/{cfg.num_conditions}",
                    last_cost=f"{episode_cost:.1f}",
                )
            c_bar.close()

            # ========== S-STEP: Distill into policy ==========
            # Pull training data from either the cross-iteration buffer
            # (DAgger-style replay) or the current iteration's per-condition
            # lists. `distill_buffer_cap == 0` → current per-iter behaviour.
            if cfg.distill_buffer_cap > 0:
                obs_flat = np.concatenate(
                    [ep["obs"] for ep in self._episode_buffer], axis=0)
                act_flat = np.concatenate(
                    [ep["actions"] for ep in self._episode_buffer], axis=0)
                w_flat = np.concatenate(
                    [ep["weights"] for ep in self._episode_buffer], axis=0)
            else:
                obs_flat = np.concatenate(all_obs, axis=0)     # (total_steps, obs_dim)
                act_flat = np.concatenate(all_actions, axis=0)  # (total_steps, act_dim)
                w_flat = np.concatenate(all_weights, axis=0)    # (total_steps,)

            # Multiple epochs of mini-batch gradient descent
            loss = self._distill_epoch(obs_flat, act_flat, w_flat)

            # ========== KL computation (average across conditions) ==========
            if skip_kl:
                # Policy-prior-only variant: no KL tracking, no BADMM update.
                # nu stays at badmm_init_nu; the MPPI prior weight is alpha * nu.
                kl_mean = float("nan")
            else:
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
            mppi_cost = float(np.mean(condition_costs))
            history.iteration_costs.append(mppi_cost)
            history.iteration_kl.append(kl_mean)
            history.iteration_nu.append(self.nu)
            history.distill_losses.append(loss)

            # ========== Policy evaluation (drives best-checkpoint selection) ==========
            # We evaluate the student policy on fresh env resets and use THAT cost
            # to pick best.pt. The C-step rollout cost above is the MPPI teacher
            # under a policy prior — not a faithful measure of the student.
            eval_every = max(int(getattr(cfg, "eval_every", 1)), 1)
            is_last = (iteration == num_iterations - 1)
            do_eval = is_last or ((iteration + 1) % eval_every == 0)

            eval_cost = float("nan")
            if do_eval:
                was_training = self.policy.training
                self.policy.eval()
                eval_stats = evaluate_policy(
                    self.policy, self.env,
                    n_episodes=cfg.n_eval_eps,
                    episode_len=cfg.eval_ep_len,
                    seed=iteration,  # vary conditions across iters but deterministic per-iter
                )
                if was_training:
                    self.policy.train()
                eval_cost = float(eval_stats["mean_cost"])
            history.iteration_eval_costs.append(eval_cost)

            # ========== Checkpointing ==========
            if run_dir is not None:
                iter_path = run_dir / f"iter_{iteration:03d}.pt"
                save_checkpoint(
                    iter_path, self.policy,
                    iteration=iteration,
                    mppi_cost=mppi_cost,
                    eval_cost=eval_cost,
                    distill_loss=loss,
                    kl=kl_mean,
                    nu=self.nu,
                )
                # Best is selected on eval cost only (NaN comparisons are false,
                # so skipped-eval iters never overwrite best.pt).
                if do_eval and eval_cost < history.best_cost:
                    history.best_cost = eval_cost
                    history.best_iter = iteration
                    copy_as(iter_path, run_dir / "best.pt")

            postfix = {
                "mppi_cost": f"{mppi_cost:.2f}",
                "loss": f"{loss:.3f}",
                "kl": f"{kl_mean:.3f}",
                "nu": f"{self.nu:.3f}",
                "best": history.best_iter if history.best_iter >= 0 else "-",
            }
            if do_eval:
                postfix["eval_cost"] = f"{eval_cost:.2f}"
            outer_bar.set_postfix(**postfix)

            base_line = (
                f"[GPS iter {iteration:3d}]  "
                f"mppi_cost={mppi_cost:8.2f}  "
                f"distill_loss={loss:.4f}  "
                f"kl={kl_mean:.4f}  "
                f"nu={self.nu:.4f}"
            )
            if cfg.distill_buffer_cap > 0:
                n_eps = len(self._episode_buffer)
                n_rows = sum(len(ep["obs"]) for ep in self._episode_buffer)
                base_line += f"  buf={n_eps}eps/{n_rows}rows"
            if do_eval:
                base_line += f"  eval_cost={eval_cost:8.2f}"
            tqdm.write(base_line)

        outer_bar.close()
        return history
