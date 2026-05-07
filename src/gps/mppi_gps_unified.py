"""Unified MPPI-GPS for GaussianPolicy and DeterministicPolicy students.

Subsumes ``mppi_gps_clip.py`` (Gaussian, NLL distill, optional PPO clip) and
``mppi_gps_det.py`` (Deterministic, MSE distill, grad-norm clip). The C-step
rollout, replay buffer, sub-episode flush, EMA / hard-sync / Adam-reset
stabilisers, dagger-relabel, warm-start, auto-reset, and eval window are all
shared. Policy-class-specific branches:

* **Policy prior** (``GPSConfig.policy_prior_type``):
    - ``"auto"`` (default) → resolves to ``"nll"`` for Gaussian,
      ``"mean_distance"`` for Deterministic.
    - ``"nll"``: ``prior(s, a) = -alpha * Σ_t log π(a_t | o_t)``.
      Gaussian-only — needs ``log_prob``; raises for Deterministic.
    - ``"mean_distance"``: ``prior(s, a) = alpha * Σ_t ‖a_t − π.action(o_t)‖²``.
      Uses ``policy.action(...)`` which returns the mean for both classes.

* **Distill loss** is implicit in the policy class:
    - DeterministicPolicy → MSE always (``mse_step`` with optional
      ``grad_clip_norm``). ``cfg.distill_loss`` is ignored.
    - GaussianPolicy → weighted NLL (``train_weighted``) with optional
      ``cfg.prev_iter_kl_coef`` trust region. When ``cfg.clip_ratio > 0``,
      the per-batch surrogate switches to the PPO ratio-clipped form
      (mirroring ``mppi_gps_clip._train_step_clipped``'s Gaussian branch).
"""

from __future__ import annotations

import copy
import csv
import math
import os
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from dataclasses import dataclass, field
from tqdm.auto import tqdm

from src.envs.base import BaseEnv
from src.mppi.mppi import MPPI
from src.policy.deterministic_policy import DeterministicPolicy
from src.policy.gaussian_policy import GaussianPolicy
from src.utils.config import GPSConfig, MPPIConfig, PolicyConfig
from src.utils.evaluation import evaluate_policy
from src.utils.experiment import copy_as, save_checkpoint


# ---------------------------------------------------------------------------
# Alpha schedule
# ---------------------------------------------------------------------------

def schedule_alpha(iteration: int, cfg: GPSConfig) -> float:
    """Per-iteration α value for the policy-prior weight.

    Returns ``cfg.policy_augmented_alpha`` (constant) when the schedule
    is "constant" or warmup is disabled — preserves legacy behavior.
    Otherwise interpolates from ``cfg.alpha_start`` at iter 0 to
    ``cfg.policy_augmented_alpha`` at iter ``cfg.alpha_warmup_iters``,
    holding constant thereafter.

    Schedule shapes (x = iteration / alpha_warmup_iters, clipped to [0, 1]):
      * ``"linear"``     — ``x``
      * ``"smoothstep"`` — ``x²(3 − 2x)``    (default ramp; deriv 0 at endpoints)
      * ``"cosine"``     — ``0.5(1 − cos πx)`` (visually similar to smoothstep)
    """
    if cfg.alpha_schedule == "constant" or cfg.alpha_warmup_iters <= 0:
        return cfg.policy_augmented_alpha
    if iteration >= cfg.alpha_warmup_iters:
        return cfg.policy_augmented_alpha

    x = iteration / cfg.alpha_warmup_iters
    if cfg.alpha_schedule == "linear":
        ramp = x
    elif cfg.alpha_schedule == "smoothstep":
        ramp = x * x * (3.0 - 2.0 * x)
    elif cfg.alpha_schedule == "cosine":
        ramp = 0.5 * (1.0 - math.cos(math.pi * x))
    else:
        raise ValueError(
            f"unknown alpha_schedule={cfg.alpha_schedule!r}; "
            f"expected one of: constant | linear | smoothstep | cosine"
        )
    return cfg.alpha_start + (cfg.policy_augmented_alpha - cfg.alpha_start) * ramp


# ---------------------------------------------------------------------------
# KL-adaptive α (MDGPS-style dual variable)
# ---------------------------------------------------------------------------

def estimate_kl_p_to_policy(
    obs: np.ndarray,
    mu_p_phys: np.ndarray,
    var_p_phys: np.ndarray,
    policy: GaussianPolicy,
    sigma_p_floor_norm: float | np.ndarray = 0.05,
) -> float:
    """E_state[ KL(N(μ_p, σ_p²) ‖ π_θ(·|s)) ] over visited states.

    ``(μ_p, σ_p²)`` are the weighted mean and per-dim variance of MPPI's
    first-step action samples at each state, in **physical** action space.
    The policy's head outputs in **normalized** space when ``USE_ACT_NORM``
    is on, so we move μ_p and σ_p into normalized space before evaluating
    the closed-form Gaussian KL — same reason the distill loss normalizes
    its labels (the network's μ_θ and σ_θ live there).

    ``sigma_p_floor_norm`` (scalar or per-dim, in NORMALIZED action space)
    bounds σ_p from below. Critical because MPPI's importance weights can
    become near-one-hot when a single sample dominates the softmin (low
    effective sample size), driving the weighted variance toward zero. A
    log(σ_θ/σ_p) term then explodes regardless of mean alignment, which
    biases ``kl_est`` upward by tens-to-hundreds of nats per state and
    makes the dual update saturate at ``kl_alpha_max`` even when the
    distributions are reasonably aligned. The floor encodes "the local
    policy is at least as wide as a fraction of MPPI's proposal noise" —
    which is honest, since the policy will face that proposal noise during
    execution.

    Diagonal-Gaussian assumption (no off-diagonals in the local-policy
    covariance — fine for a per-dim variance estimate from MPPI samples).
    """
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=policy.device)
    with torch.no_grad():
        mu_th_t, log_sigma_th_t = policy._head(obs_t)
    mu_th = mu_th_t.cpu().numpy()
    sigma_th = log_sigma_th_t.exp().cpu().numpy()

    if policy._has_act_norm:
        scale = policy._act_scale.detach().cpu().numpy()
        bias = policy._act_bias.detach().cpu().numpy()
        mu_p = (mu_p_phys - bias) / scale
        var_p = var_p_phys / (scale ** 2)
    else:
        mu_p = mu_p_phys
        var_p = var_p_phys

    # Floor on σ_p prevents a near-one-hot MPPI weight distribution from
    # collapsing weighted-sample variance to ~0 (would otherwise blow up
    # log(σ_θ/σ_p)). 1e-12 → tens of nats per dim of bias — see docstring.
    floor = np.asarray(sigma_p_floor_norm, dtype=var_p.dtype) ** 2
    sigma_p = np.sqrt(np.maximum(var_p, floor))

    # KL(N(μ_p, σ_p²) ‖ N(μ_θ, σ_θ²)) per dim, summed across action dims:
    #   = log(σ_θ/σ_p) + (σ_p² + (μ_p − μ_θ)²) / (2 σ_θ²)  − 0.5
    kl_per_dim = (
        np.log(sigma_th / sigma_p)
        + (sigma_p ** 2 + (mu_p - mu_th) ** 2) / (2.0 * sigma_th ** 2 + 1e-12)
        - 0.5
    )
    kl_per_state = kl_per_dim.sum(axis=-1)        # (N,)
    return float(kl_per_state.mean())              # scalar — average over states


def update_alpha_kl_adaptive(
    alpha: float,
    kl_est: float,
    cfg: GPSConfig,
) -> float:
    """Multiplicative dual update — but with the unbiased KL operand the
    direction is inverted relative to vanilla MDGPS.

    `kl_est` here is ``E_state[ KL(p_unbiased ‖ π_θ) ]`` — the gap
    between the *unbiased* MPPI teacher and the global policy. Large
    `kl_est` ⇒ policy is bad ⇒ we want the executor MPPI to **ignore**
    the policy prior and explore freely so the C-step gathers unbiased
    data. Hence:

      kl_est > target ⇒ SHRINK α (let MPPI escape the bad policy)
      kl_est < target ⇒ GROW α   (policy is reliable; lean on it more)

    Contrast with the biased-operand variant, where α would grow to pull
    biased MPPI ever closer to π_θ regardless of policy quality — a fixed
    point of "MPPI shadows policy", not "policy improves". Multiplicative
    in α-space (symmetric in log-α). Clipped to [kl_alpha_min,
    kl_alpha_max] to avoid degenerate fixed points.
    """
    if kl_est > cfg.kl_target:
        new = alpha / cfg.kl_step_rate
    else:
        new = alpha * cfg.kl_step_rate
    return float(np.clip(new, cfg.kl_alpha_min, cfg.kl_alpha_max))


# ---------------------------------------------------------------------------
# Policy priors
# ---------------------------------------------------------------------------

def make_mean_distance_prior(
    policy,
    env: BaseEnv,
    alpha: float,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """``prior(s, a) = alpha * Σ_t ‖a_t − π.action(o_t)‖²``.

    Works for any policy that exposes ``action(obs_tensor) -> Tensor`` of
    shape (B, act_dim). For GaussianPolicy this is the mean head (no σ, no
    sampling — sampling adds variance without changing the expectation).
    For DeterministicPolicy this is the regression output directly.
    """
    device = policy._device
    # Per-dim half-range from the env (1.0s for envs that don't override).
    # Used to rescale the per-dim residual so a fixed *fraction-of-range*
    # error costs the same on every dim, regardless of whether that dim
    # is a 0.2 m arm slide or a 2.0 rad finger joint. Without this, the
    # raw L2 prior is dominated by whichever dims have the largest
    # ctrlrange, biasing MPPI toward "match the policy on the loud dims,
    # ignore the quiet ones" — exactly the asymmetry that makes the
    # high-DoF Adroit policy's arm look weak relative to its fingers.
    # For envs whose noise_scale is `np.ones(action_dim)` (the default),
    # this is a no-op — behaviour matches the previous unweighted L2.
    inv_scale = 1.0 / np.asarray(env.noise_scale, dtype=np.float64)

    def prior_cost(
        states: np.ndarray,
        actions: np.ndarray,
        sensordata: np.ndarray | None = None,
    ) -> np.ndarray:
        states = np.asarray(states)
        actions = np.asarray(actions)
        K, H, act_dim = actions.shape

        obs = np.asarray(env.state_to_obs(states, sensordata))  # (K, H, obs_dim)
        obs_flat = obs.reshape(K * H, -1)
        with torch.no_grad():
            obs_t = torch.as_tensor(obs_flat, dtype=torch.float32, device=device)
            mu_flat = policy.action(obs_t).cpu().numpy()
        mu = mu_flat.reshape(K, H, act_dim)
        # ((a − μ) / scale)² — broadcasts over (K, H, act_dim); sum over
        # time + dim → (K,). Equivalent to comparing actions and mu in
        # normalized action space, but keeps mu in physical so the rest
        # of the policy API (clip_eps surrogate, DAgger MSE, eval) is
        # unchanged.
        sq = (((actions - mu) * inv_scale) ** 2).sum(axis=(1, 2))   # (K,)
        return alpha * sq

    return prior_cost


def make_nll_prior(
    policy: GaussianPolicy,
    env: BaseEnv,
    alpha: float,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """``prior(s, a) = -alpha * Σ_t log π(a_t | o_t)``. GaussianPolicy only.

    Negative log-likelihood as a cost contribution. Sigma-aware — actions
    near a low-σ mean carry more weight than the same residual at a high-σ
    mean. Effective contribution to ``log w`` is ``+(alpha/lambda) * Σ log π``.

    Note: no per-dim weighting is applied here (unlike `mean_distance`).
    The diagonal Gaussian NLL already normalizes per-dim by ``σ_i²`` —
    σ adapts to the data scale per dim during training, so residuals are
    intrinsically equalized across heterogeneous action ranges. Adding
    extra weighting would double-count.
    """
    if not isinstance(policy, GaussianPolicy):
        raise TypeError(
            f"nll prior requires GaussianPolicy (got {type(policy).__name__})"
        )

    def prior_fn(
        states: np.ndarray,
        actions: np.ndarray,
        sensordata: np.ndarray | None = None,
    ) -> np.ndarray:
        states = np.asarray(states)
        actions = np.asarray(actions)
        K, H, _ = states.shape

        obs = np.asarray(env.state_to_obs(states, sensordata))  # (K, H, obs_dim)
        obs_flat = obs.reshape(K * H, -1)
        act_flat = actions.reshape(K * H, -1)
        lp = policy.log_prob_np(obs_flat, act_flat)     # (K*H,)
        lp = lp.reshape(K, H).sum(axis=1)               # (K,)
        return -alpha * lp

    return prior_fn


def make_policy_prior(
    policy,
    env: BaseEnv,
    alpha: float,
    prior_type: str,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Dispatch to the requested prior. Caller is responsible for resolving
    ``"auto"`` to a concrete type (the trainer does this in ``__init__`` so
    incompatible combinations raise eagerly)."""
    if prior_type == "mean_distance":
        return make_mean_distance_prior(policy, env, alpha)
    if prior_type == "nll":
        return make_nll_prior(policy, env, alpha)
    raise ValueError(
        f"unknown policy_prior_type: {prior_type!r} "
        f"(expected 'nll' or 'mean_distance')"
    )


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------

@dataclass
class GPSHistory:
    """Per-iteration metrics."""
    iteration_costs: list[float] = field(default_factory=list)
    iteration_eval_costs: list[float] = field(default_factory=list)
    distill_losses: list[float] = field(default_factory=list)
    iteration_ema_drift: list[float] = field(default_factory=list)
    iteration_prev_iter_kl: list[float] = field(default_factory=list)
    iteration_alpha: list[float] = field(default_factory=list)
    iteration_kl_est: list[float] = field(default_factory=list)   # MDGPS KL(p ‖ π_θ); NaN when adaptive α is off
    best_iter: int = -1
    best_cost: float = float("inf")


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class MPPIGPS:
    """Unified MPPI-GPS trainer for both Gaussian and Deterministic policies.

    Construct with ``deterministic=True`` to use a ``DeterministicPolicy``;
    default is ``GaussianPolicy``. The prior shape and distill loss are
    derived from the policy class plus ``GPSConfig.policy_prior_type``.
    """

    def __init__(
        self,
        env: BaseEnv,
        mppi_cfg: MPPIConfig,
        policy_cfg: PolicyConfig,
        gps_cfg: GPSConfig,
        device: torch.device | str | None = None,
        deterministic: bool = False,
    ):
        self.env = env
        self.mppi_cfg = mppi_cfg
        self.policy_cfg = policy_cfg
        self.gps_cfg = gps_cfg
        self._deterministic = deterministic

        # ---- Policy ----
        if deterministic:
            self.policy = DeterministicPolicy(
                env.obs_dim, env.action_dim, policy_cfg,
                device=device, action_bounds=env.action_bounds,
            )
        else:
            self.policy = GaussianPolicy(
                env.obs_dim, env.action_dim, policy_cfg,
                device=device, action_bounds=env.action_bounds,
            )

        # ---- Resolve "auto" prior type and validate combo ----
        prior_type = getattr(gps_cfg, "policy_prior_type", "auto")
        if prior_type == "auto":
            prior_type = "mean_distance" if deterministic else "nll"
        if prior_type == "nll" and deterministic:
            raise ValueError(
                "policy_prior_type='nll' requires GaussianPolicy; "
                "DeterministicPolicy has no log_prob."
            )
        if prior_type not in ("nll", "mean_distance"):
            raise ValueError(
                f"unknown policy_prior_type: {prior_type!r} "
                f"(expected 'auto', 'nll', or 'mean_distance')"
            )
        self._prior_type = prior_type

        self.mppi = MPPI(env, mppi_cfg)
        # Cross-iteration episode replay buffer (rows-capped, FIFO eviction
        # by whole episodes). Empty when distill_buffer_cap == 0.
        self._episode_buffer: list[dict] = []

        # KL-adaptive α (MDGPS dual variable). Lazily seeded from the
        # schedule on the first iter that the adaptive rule activates;
        # carried iter-to-iter once active. None means "schedule path".
        self._kl_alpha: float | None = None

        # EMA shadow over trainable params. Eval and best.pt selection happen
        # inside an ema_swapped_in() window, so the on-disk checkpoint matches
        # the reported eval cost.
        if gps_cfg.ema_decay > 0.0:
            self.policy.attach_ema(gps_cfg.ema_decay)

    # ----- helpers ---------------------------------------------------------

    def _sample_initial_conditions(self, n: int) -> list[np.ndarray]:
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
        prev_policy=None,
    ) -> float:
        """Mini-batch distillation. Loss type derived from policy class.

        * Deterministic → MSE via ``policy.mse_step`` with optional
          ``cfg.grad_clip_norm`` clipping the gradient L2 norm. When
          ``cfg.clip_eps > 0``, the per-batch MPPI label is also clamped
          to ``[old_policy.action(o) ± clip_eps]`` before the MSE loss
          (action-space trust region; mirrors ``mppi_gps_clip``'s MSE
          branch). ``weights`` and ``prev_policy`` are ignored either way.
        * Gaussian → weighted NLL via ``policy.train_weighted`` with
          optional ``cfg.prev_iter_kl_coef`` trust region against
          ``prev_policy``. If ``cfg.clip_ratio > 0``, every batch goes
          through the PPO ratio-clipped surrogate instead.
        """
        cfg = self.gps_cfg
        N = len(obs)
        indices = np.arange(N)
        last_loss = 0.0

        if self._deterministic:
            grad_clip_norm = float(getattr(cfg, "grad_clip_norm", 0.0))
            clip_eps = float(getattr(cfg, "clip_eps", 0.0))

            # Snapshot for action-target clip (frozen, eval, no-grad).
            # Skipped when clip_eps == 0 to avoid the deepcopy cost when
            # the trust region is disabled. Same skip-if-zero pattern
            # as the Gaussian PPO branch below.
            old_policy = None
            if clip_eps > 0.0:
                old_policy = copy.deepcopy(self.policy)
                old_policy.eval()
                for p in old_policy.parameters():
                    p.requires_grad_(False)

            device = self.policy.device

            for _ in range(cfg.distill_epochs):
                np.random.shuffle(indices)
                for start in range(0, N, cfg.distill_batch_size):
                    batch = indices[start : start + cfg.distill_batch_size]
                    obs_b = obs[batch]
                    act_b = actions[batch]
                    if old_policy is not None:
                        # Clamp the label to ±clip_eps around the old
                        # policy's prediction. Per-element np.clip with
                        # array bounds (NOT a single scalar) so each
                        # action dim is independently trust-regioned.
                        with torch.no_grad():
                            o_t = torch.as_tensor(
                                obs_b, dtype=torch.float32, device=device,
                            )
                            old_pred = old_policy.action(o_t).cpu().numpy()
                        act_b = np.clip(
                            act_b, old_pred - clip_eps, old_pred + clip_eps,
                        )
                    last_loss = self.policy.mse_step(
                        obs_b, act_b,
                        grad_clip_norm=grad_clip_norm,
                    )
            return last_loss

        # ---- Gaussian path ----
        clip_ratio = float(getattr(cfg, "clip_ratio", 0.2))
        use_prev_kl = (
            prev_policy is not None
            and cfg.prev_iter_kl_coef > 0.0
        )

        # Snapshot for PPO clip (frozen, eval, no-grad). Skipped when
        # clip_ratio == 0 to avoid the deepcopy + per-batch forward.
        old_policy = None
        if clip_ratio > 0.0:
            old_policy = copy.deepcopy(self.policy)
            old_policy.eval()
            for p in old_policy.parameters():
                p.requires_grad_(False)

        for _ in range(cfg.distill_epochs):
            np.random.shuffle(indices)
            for start in range(0, N, cfg.distill_batch_size):
                batch = indices[start : start + cfg.distill_batch_size]
                if old_policy is not None:
                    last_loss = self._train_step_ppo_clip(
                        obs[batch], actions[batch], weights[batch],
                        old_policy, clip_ratio,
                    )
                else:
                    last_loss = self.policy.train_weighted(
                        obs[batch], actions[batch], weights[batch],
                        prev_policy=prev_policy if use_prev_kl else None,
                        prev_kl_coef=cfg.prev_iter_kl_coef if use_prev_kl else 0.0,
                    )
        return last_loss

    def _train_step_ppo_clip(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        weights: np.ndarray,
        old_policy: GaussianPolicy,
        clip_ratio: float,
    ) -> float:
        """PPO ratio-clipped surrogate. Mirrors mppi_gps_clip's Gaussian branch.

        ``L = - E_w[ min(r·w, clip(r, 1-eps, 1+eps)·w) ]`` where
        ``r = π_θ(a|o) / π_old(a|o)``.
        """
        device = self.policy.device
        o_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
        a_t = torch.as_tensor(actions, dtype=torch.float32, device=device)
        w_t = torch.as_tensor(weights, dtype=torch.float32, device=device)

        if self.policy.normalizer is not None:
            self.policy.normalizer.update(o_t)

        curr_log_prob = self.policy.log_prob(o_t, a_t)
        with torch.no_grad():
            old_log_prob = old_policy.log_prob(o_t, a_t)

        # Clamp the log-ratio so exp() can't overflow to +inf. When σ
        # collapses (log_sigma near log_sigma_min), per-row log-prob
        # magnitudes grow into the millions and inter-batch differences
        # easily exceed 88 (= log of float32 max for exp). An overflowed
        # ratio creates a 0·∞ in the autograd chain (torch.min picks the
        # finite surr2, giving the surr1 branch outer-grad 0; but ratio's
        # local grad d_exp/d_x = exp(x) = inf, so 0 · inf = NaN).
        # 20 caps ratio at exp(20) ≈ 5e8 — well below any clip threshold,
        # so behaviour is unchanged in the regime where PPO actually
        # matters (ratio ≈ 1±eps).
        log_ratio = torch.clamp(curr_log_prob - old_log_prob, min=-20.0, max=20.0)
        ratio = torch.exp(log_ratio)
        surr1 = ratio * w_t
        surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * w_t
        loss = -torch.min(surr1, surr2).sum() / w_t.sum().clamp(min=1e-8)

        self.policy.optimizer.zero_grad()
        loss.backward()
        # NaN-loss guard: same rationale as in GaussianPolicy.train_weighted.
        if not torch.isfinite(loss):
            self.policy.optimizer.zero_grad()
            return float("nan")
        self.policy.optimizer.step()
        # Mirror mse_step / train_weighted's EMA hook so the shadow stays
        # in sync regardless of which loss path runs.
        if self.policy.ema is not None:
            self.policy.ema.update(self.policy)
        return float(loss.item())

    def _warm_start_mppi(self, initial_state: np.ndarray):
        """Seed MPPI nominal U with a policy rollout from ``initial_state``."""
        self.env.reset(state=initial_state)
        actions: list[np.ndarray] = []
        done = False
        for _ in range(self.mppi_cfg.H):
            if done:
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
        num_iterations = num_iterations or self.gps_cfg.num_iterations
        cfg = self.gps_cfg
        history = GPSHistory()

        if run_dir is not None:
            run_dir = Path(run_dir)
            run_dir.mkdir(parents=True, exist_ok=True)

        # ---- Per-iter CSV log (crash-safe) ---------------------------------
        # Opened once, header written once, one row per iter with explicit
        # flush + fsync so a SIGKILL / OOM during iter N still leaves rows
        # 0..N-1 readable on disk. Uniform schema across runs (NaN for
        # inactive columns like kl_est when adaptive is off, ema_drift
        # when EMA is off, etc.) so downstream code can pd.read_csv all
        # runs without per-run column gymnastics.
        csv_columns = [
            "iter", "alpha", "mppi_cost", "distill_loss",
            "kl_est", "kl_target",
            "ema_drift", "prev_iter_kl",
            "buffer_eps", "buffer_rows",
            "eval_cost", "best_iter", "best_cost",
        ]
        csv_file = None
        csv_writer = None
        if run_dir is not None:
            csv_path = run_dir / "gps_log.csv"
            csv_file = open(csv_path, "w", newline="")
            csv_writer = csv.DictWriter(csv_file, fieldnames=csv_columns)
            csv_writer.writeheader()
            csv_file.flush()

        initial_conditions = self._sample_initial_conditions(cfg.num_conditions)

        # KL-adaptive α activates only for Gaussian policies — needs σ for
        # the closed-form Gaussian KL. Deterministic falls back to schedule.
        kl_adaptive_enabled = (
            cfg.kl_target > 0.0 and not self._deterministic
        )

        outer_bar = tqdm(range(num_iterations), desc="GPS", unit="iter")
        for iteration in outer_bar:
            # Pick α for this iter:
            #  * during warmup (or when adaptive disabled): use the schedule
            #  * after warmup (when adaptive enabled): use the dual variable
            #    self._kl_alpha, which is initialized lazily from the
            #    schedule on the first adaptive iter and updated at the end
            #    of each iter via dual gradient ascent.
            use_kl_adaptive_this_iter = (
                kl_adaptive_enabled and iteration >= cfg.alpha_warmup_iters
            )
            if use_kl_adaptive_this_iter:
                if self._kl_alpha is None:
                    self._kl_alpha = schedule_alpha(iteration, cfg)
                alpha = self._kl_alpha
            else:
                alpha = schedule_alpha(iteration, cfg)

            all_obs: list[np.ndarray] = []
            all_actions: list[np.ndarray] = []
            all_weights: list[np.ndarray] = []
            condition_costs: list[float] = []

            # KL-adaptive accumulator. Per plan_step we record (obs at
            # plan time, weighted-mean of MPPI's first-step actions,
            # weighted-var of same) — used after the C-step to estimate
            # KL(p ‖ π_θ) and update the dual α. Empty when adaptive is
            # disabled, kept tiny in size (3 small arrays per timestep).
            kl_obs_buf: list[np.ndarray] = []
            kl_mu_p_buf: list[np.ndarray] = []
            kl_var_p_buf: list[np.ndarray] = []

            # ========== C-STEP ==========
            # eval() so the prior_fn forward (which goes through Dropout +
            # LayerNorm in BOTH policy classes) is deterministic. Train
            # mode would inject Dropout noise into MPPI's cost.
            self.policy.eval()

            c_step_total = cfg.num_conditions * cfg.episode_length
            c_bar = tqdm(
                total=c_step_total,
                desc=f"  iter {iteration:3d} C-step",
                unit="step",
                leave=False,
            )
            for ic_idx, ic_state in enumerate(initial_conditions):
                # Fresh prior closure each condition — captures current
                # policy weights and current alpha.
                if alpha > 0.0:
                    prior_fn = make_policy_prior(
                        self.policy, self.env, alpha, self._prior_type,
                    )
                else:
                    prior_fn = None

                if iteration > 0 and cfg.warm_start_policy:
                    self._warm_start_mppi(ic_state)
                else:
                    self.mppi.reset()

                self.env.reset(state=ic_state)

                # Sub-episodes split at every `done` boundary so a single
                # condition with auto_reset=True yields multiple entries.
                condition_sub_episodes: list[dict] = []
                cur_obs: list[np.ndarray] = []
                cur_actions: list[np.ndarray] = []
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

                    # Executor: MPPI with the policy prior. Steers the env.
                    action_exec, _info = self.mppi.plan_step(state, prior=prior_fn)

                    # KL-adaptive: snapshot MPPI's per-state first-step
                    # action distribution. weighted mean / variance over
                    # the K samples; only collected on full-replan steps
                    # (open-loop follow-up calls don't refresh _last_*).
                    #
                    # Use UNBIASED weights (cost-only posterior, prior
                    # contribution stripped) — measures the true
                    # teacher-student gap. The biased weights would
                    # collapse to π_θ by construction as α grows, making
                    # the dual update self-referential. See
                    # ``MPPI._last_unbiased_weights``.
                    if use_kl_adaptive_this_iter and _info.get("replanned", True):
                        ws = self.mppi._last_unbiased_weights
                        first_actions = self.mppi._last_actions[:, 0, :]   # (K, act_dim)
                        mu_p = (ws[:, None] * first_actions).sum(axis=0)
                        var_p = (
                            ws[:, None] * (first_actions - mu_p) ** 2
                        ).sum(axis=0)
                        kl_obs_buf.append(obs.copy())
                        kl_mu_p_buf.append(mu_p)
                        kl_var_p_buf.append(var_p)

                    # DAgger-style relabel: a fresh prior-free MPPI call
                    # whose action is the training label. dry_run=True keeps
                    # MPPI's persistent state intact so the executor is
                    # undisturbed. Gated on alpha > 0 (no point relabeling
                    # when the prior is already inactive).
                    if cfg.dagger_relabel and alpha > 0.0:
                        action_label, _ = self.mppi.plan_step(
                            state, prior=None, dry_run=True,
                        )
                    else:
                        action_label = action_exec

                    cur_obs.append(obs.copy())
                    cur_actions.append(action_label.copy())

                    _, cost, done, _ = self.env.step(action_exec)
                    episode_cost += cost
                    c_bar.update(1)

                    if done and t < cfg.episode_length - 1:
                        _flush_sub_episode()
                        if cfg.auto_reset:
                            self.env.reset()
                            self.mppi.reset()
                            continue
                        c_bar.update(cfg.episode_length - (t + 1))
                        break

                _flush_sub_episode()

                # Route into either the cross-iter buffer or per-iter lists.
                if cfg.distill_buffer_cap > 0:
                    for sub_ep in condition_sub_episodes:
                        self._episode_buffer.append(sub_ep)
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

                condition_costs.append(episode_cost)
                c_bar.set_postfix(
                    ic=f"{ic_idx + 1}/{cfg.num_conditions}",
                    last_cost=f"{episode_cost:.1f}",
                )
            c_bar.close()

            # ========== S-STEP ==========
            # train() so Dropout + LayerNorm-running-stats behave correctly
            # for gradient steps.
            self.policy.train()

            if cfg.distill_buffer_cap > 0:
                obs_flat = np.concatenate(
                    [ep["obs"] for ep in self._episode_buffer], axis=0)
                act_flat = np.concatenate(
                    [ep["actions"] for ep in self._episode_buffer], axis=0)
                w_flat = np.concatenate(
                    [ep["weights"] for ep in self._episode_buffer], axis=0)
            else:
                obs_flat = np.concatenate(all_obs, axis=0)
                act_flat = np.concatenate(all_actions, axis=0)
                w_flat = np.concatenate(all_weights, axis=0)

            # Snapshot of pre-S-step policy for the trust-region KL-to-prev
            # penalty. Gaussian-only (deterministic has no distribution); we
            # still tolerate the field being set with a det policy by simply
            # skipping the snapshot.
            prev_policy = None
            if (
                not self._deterministic
                and cfg.prev_iter_kl_coef > 0.0
            ):
                prev_policy = copy.deepcopy(self.policy)
                prev_policy.eval()
                for p in prev_policy.parameters():
                    p.requires_grad_(False)

            loss = self._distill_epoch(obs_flat, act_flat, w_flat, prev_policy=prev_policy)

            # Diagnostic: post-distill EMA drift. Compute BEFORE any hard-sync
            # so the diagnostic reflects how far theta actually moved this
            # S-step (post-sync drift would always be 0).
            ema_drift = (
                self.policy.ema_l2_drift() if cfg.ema_decay > 0.0 else float("nan")
            )

            # prev-iter KL diagnostic — Gaussian only, requires kl_to_np.
            if (
                prev_policy is not None
                and not self._deterministic
                and len(obs_flat) > 0
                and hasattr(self.policy, "kl_to_np")
            ):
                diag_n = min(len(obs_flat), 4096)
                diag_idx = np.random.choice(len(obs_flat), size=diag_n, replace=False)
                prev_iter_kl_diag = self.policy.kl_to_np(
                    obs_flat[diag_idx], prev_policy,
                )
            else:
                prev_iter_kl_diag = float("nan")

            # End-of-S-step stabilisers: hard-sync first, then Adam reset.
            if cfg.ema_decay > 0.0 and cfg.ema_hard_sync:
                self.policy.ema_sync()
            if cfg.reset_optim_per_iter:
                self.policy.reset_optimizer()

            # ========== KL-adaptive α update (MDGPS dual ascent) =========
            # Estimate KL(p ‖ π_θ) from the C-step data, then take a dual
            # gradient step on α to drive KL toward kl_target. Uses an
            # ema-swapped policy snapshot for consistency with eval — the
            # KL we measure is against the policy that's about to be
            # checkpointed and used as next iter's prior.
            kl_est_iter = float("nan")
            if use_kl_adaptive_this_iter and kl_obs_buf:
                # σ_p floor in NORMALIZED action space. MPPI's per-dim
                # proposal std is ``cfg.noise_sigma * env.noise_scale``;
                # divided by ``policy._act_scale`` (also a per-dim
                # half-range) it reduces to roughly ``cfg.noise_sigma`` —
                # a scalar in normalized space when noise_scale matches
                # the action-bound half-range (true for Adroit). Frac
                # 0 disables flooring (legacy 1e-6 behaviour).
                if cfg.kl_sigma_floor_frac > 0.0:
                    if self.policy._has_act_norm:
                        scale_np = self.policy._act_scale.detach().cpu().numpy()
                        sigma_p_floor = (
                            self.mppi.sigma * cfg.kl_sigma_floor_frac / scale_np
                        )
                    else:
                        sigma_p_floor = self.mppi.sigma * cfg.kl_sigma_floor_frac
                else:
                    sigma_p_floor = 1e-6
                with self.policy.ema_swapped_in():
                    self.policy.eval()
                    kl_est_iter = estimate_kl_p_to_policy(
                        np.stack(kl_obs_buf),
                        np.stack(kl_mu_p_buf),
                        np.stack(kl_var_p_buf),
                        self.policy,
                        sigma_p_floor_norm=sigma_p_floor,
                    )
                self._kl_alpha = update_alpha_kl_adaptive(alpha, kl_est_iter, cfg)

            # ========== Logging ==========
            mppi_cost = float(np.mean(condition_costs))
            history.iteration_costs.append(mppi_cost)
            history.distill_losses.append(loss)
            history.iteration_ema_drift.append(ema_drift)
            history.iteration_prev_iter_kl.append(prev_iter_kl_diag)
            history.iteration_alpha.append(float(alpha))
            history.iteration_kl_est.append(kl_est_iter)

            # ========== Eval (drives best.pt) ==========
            eval_every = max(int(getattr(cfg, "eval_every", 1)), 1)
            is_last = (iteration == num_iterations - 1)
            do_eval = is_last or ((iteration + 1) % eval_every == 0)

            eval_cost = float("nan")
            with self.policy.ema_swapped_in():
                if do_eval:
                    self.policy.eval()
                    eval_stats = evaluate_policy(
                        self.policy, self.env,
                        n_episodes=cfg.n_eval_eps,
                        episode_len=cfg.eval_ep_len,
                        seed=iteration,
                    )
                    eval_cost = float(eval_stats["mean_cost"])
                history.iteration_eval_costs.append(eval_cost)

                if run_dir is not None:
                    iter_path = run_dir / f"iter_{iteration:03d}.pt"
                    save_checkpoint(
                        iter_path, self.policy,
                        iteration=iteration,
                        mppi_cost=mppi_cost,
                        eval_cost=eval_cost,
                        distill_loss=loss,
                        ema_drift=ema_drift,
                        prev_iter_kl=prev_iter_kl_diag,
                    )
                    if do_eval and eval_cost < history.best_cost:
                        history.best_cost = eval_cost
                        history.best_iter = iteration
                        copy_as(iter_path, run_dir / "best.pt")

            postfix = {
                "mppi_cost": f"{mppi_cost:.2f}",
                "loss": f"{loss:.3f}",
                "best": history.best_iter if history.best_iter >= 0 else "-",
            }
            schedule_active = (
                cfg.alpha_schedule != "constant" and cfg.alpha_warmup_iters > 0
            )
            if schedule_active or use_kl_adaptive_this_iter:
                postfix["alpha"] = f"{alpha:.3f}"
            if use_kl_adaptive_this_iter:
                postfix["kl"] = f"{kl_est_iter:.3f}"
            if do_eval:
                postfix["eval_cost"] = f"{eval_cost:.2f}"
            outer_bar.set_postfix(**postfix)

            tag = "GPS-det" if self._deterministic else "GPS"
            base_line = (
                f"[{tag} iter {iteration:3d}]  "
                f"alpha={alpha:.4f}  "
                f"mppi_cost={mppi_cost:8.2f}  "
                f"distill_loss={loss:.4f}"
            )
            if use_kl_adaptive_this_iter:
                base_line += f"  kl_est={kl_est_iter:7.4f}/tgt={cfg.kl_target:.3f}"
            if cfg.distill_buffer_cap > 0:
                n_eps = len(self._episode_buffer)
                n_rows = sum(len(ep["obs"]) for ep in self._episode_buffer)
                base_line += f"  buf={n_eps}eps/{n_rows}rows"
            if cfg.ema_decay > 0.0:
                base_line += f"  ema_drift={ema_drift:.4f}"
            if not self._deterministic and cfg.prev_iter_kl_coef > 0.0:
                base_line += f"  prev_kl={prev_iter_kl_diag:.4f}"
            if do_eval:
                base_line += f"  eval_cost={eval_cost:8.2f}"
            tqdm.write(base_line)

            # ---- Crash-safe per-iter CSV row -------------------------------
            # Uniform schema: NaN for currently-inactive columns. flush()
            # + fsync() so a hard kill during iter N+1 still leaves iter N
            # readable.
            if csv_writer is not None:
                if cfg.distill_buffer_cap > 0:
                    n_eps = len(self._episode_buffer)
                    n_rows = sum(len(ep["obs"]) for ep in self._episode_buffer)
                else:
                    n_eps = float("nan")
                    n_rows = float("nan")
                row = {
                    "iter": iteration,
                    "alpha": float(alpha),
                    "mppi_cost": mppi_cost,
                    "distill_loss": float(loss),
                    "kl_est": kl_est_iter,
                    "kl_target": (
                        float(cfg.kl_target) if use_kl_adaptive_this_iter
                        else float("nan")
                    ),
                    "ema_drift": ema_drift,
                    "prev_iter_kl": prev_iter_kl_diag,
                    "buffer_eps": n_eps,
                    "buffer_rows": n_rows,
                    "eval_cost": eval_cost,
                    "best_iter": history.best_iter,
                    "best_cost": (
                        history.best_cost
                        if history.best_cost != float("inf")
                        else float("nan")
                    ),
                }
                csv_writer.writerow(row)
                csv_file.flush()
                os.fsync(csv_file.fileno())

        outer_bar.close()
        if csv_file is not None:
            csv_file.close()
        return history
