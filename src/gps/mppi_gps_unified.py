"""Unified MPPI-GPS for GaussianPolicy and DeterministicPolicy students.

Policy-class branches:
  * Prior (``GPSConfig.policy_prior_type``):
      - ``"auto"`` → ``"nll"`` (Gaussian), ``"mean_distance"`` (Det.).
      - ``"nll"``: ``-α · Σ_t log π(a_t|o_t)`` (Gaussian only).
      - ``"mean_distance"``: ``α · Σ_t ‖a_t − π.action(o_t)‖²``.
  * Distill loss: Deterministic → MSE; Gaussian → NLL with optional
    ``prev_iter_kl_coef`` trust region or ``clip_ratio`` PPO surrogate.
"""

from __future__ import annotations

import copy
import csv
import math
import os
from pathlib import Path
from typing import Callable

import h5py
import numpy as np
import torch
from dataclasses import dataclass, field
from tqdm.auto import tqdm

from src.envs.base import BaseEnv
from src.mppi.mppi import MPPI
from src.policy.deterministic_policy import DeterministicPolicy
from src.policy.gaussian_policy import GaussianPolicy
from src.utils.config import GPSConfig, MPPIConfig, PolicyConfig
from src.utils.evaluation import evaluate_mppi, evaluate_policy
from src.utils.experiment import copy_as, save_checkpoint


# α=0 rollout cache: BC-seed-compatible h5 written on iters where the
# resolved α is 0. Schema matches `scripts/collect_bc_demos.py`
# (states (M, T, obs_dim), actions (M, T, act_dim), costs (M, T),
# sensordata if present). Flushed after each iter's append.

def _try_capture_sensordata(env) -> np.ndarray | None:
    """Return `env.data.sensordata` snapshot or None for envs without it."""
    data = getattr(env, "data", None)
    if data is None:
        return None
    sd = getattr(data, "sensordata", None)
    if sd is None or sd.size == 0:
        return None
    return sd.copy()


# ---------------------------------------------------------------------------
# Alpha schedule
# ---------------------------------------------------------------------------

def schedule_alpha(iteration: int, cfg: GPSConfig) -> float:
    """Per-iteration α.

    "constant" → fixed ``policy_augmented_alpha``. Otherwise ramp from
    ``alpha_start`` over ``alpha_warmup_iters``: linear / smoothstep
    (x²(3-2x)) / cosine (0.5(1-cos πx)).
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
    """E_state[KL(N(μ_p, σ_p²) ‖ π_θ(·|s))], diagonal Gaussian.

    ``(μ_p, σ_p²)`` are the weighted mean/var of MPPI's first-step
    actions in PHYSICAL space; converted to NORMALIZED space when
    USE_ACT_NORM is on. ``sigma_p_floor_norm`` bounds σ_p from below to
    keep ``log(σ_θ/σ_p)`` finite when MPPI's softmin is near-one-hot.
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

    # σ_p floor: guard against near-one-hot MPPI weights collapsing var.
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


# ---------------------------------------------------------------------------
# Policy-trust α dampener (adapted from jaiselsingh1/mppi-gps `gps_train.py`)
# ---------------------------------------------------------------------------

def compute_policy_trust(
    *,
    policy_cost: float | None,
    raw_mppi_cost: float | None,
    eval_episode_len: int,
    cfg: GPSConfig,
) -> float:
    """Trust ∈ [min, max] from per-step (policy_cost, raw_mppi_cost).

    Linear interpolation in per-step cost space (so invariant to
    ``eval_ep_len``). Adaptive disabled → ``policy_trust_max``. Cost
    missing → ``policy_trust_min`` (defensive).
    """
    if not cfg.adaptive_policy_trust:
        return float(cfg.policy_trust_max)
    if policy_cost is None or raw_mppi_cost is None:
        return float(cfg.policy_trust_min)

    j_policy = policy_cost / max(eval_episode_len, 1)
    j_mppi = raw_mppi_cost / max(eval_episode_len, 1)
    j_bad = cfg.policy_trust_bad_cost_per_step
    denom = max(j_bad - j_mppi, 1e-8)
    quality = float(np.clip((j_bad - j_policy) / denom, 0.0, 1.0))
    return float(
        cfg.policy_trust_min
        + (cfg.policy_trust_max - cfg.policy_trust_min) * quality
    )


def update_alpha_kl_adaptive(
    alpha: float,
    kl_est: float,
    cfg: GPSConfig,
) -> float:
    """Multiplicative α update (inverted vs vanilla MDGPS).

    Large kl_est → policy bad → shrink α (let MPPI explore).
    Small kl_est → policy reliable → grow α. Clipped to
    [kl_alpha_min, kl_alpha_max].
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
    """``prior(s, a) = α · Σ_t ‖(a_t − π.action(o_t)) / scale‖²``.

    ``1/scale = 1/env.noise_scale`` equalizes per-dim contribution.
    """
    device = policy._device
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
        sq = (((actions - mu) * inv_scale) ** 2).sum(axis=(1, 2))   # (K,)
        return alpha * sq

    return prior_cost


def make_nll_prior(
    policy: GaussianPolicy,
    env: BaseEnv,
    alpha: float,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """``prior(s, a) = -α · Σ_t log π(a_t | o_t)``. GaussianPolicy only.

    Sigma-aware: diagonal Gaussian normalizes per-dim by σ_i² already.
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
    """Dispatch to the requested prior. Caller resolves ``"auto"``."""
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
    iteration_prev_iter_kl: list[float] = field(default_factory=list)
    iteration_alpha: list[float] = field(default_factory=list)
    iteration_kl_est: list[float] = field(default_factory=list)   # NaN when adaptive α off
    iteration_policy_trust: list[float] = field(default_factory=list)
    iteration_raw_mppi_eval_cost: list[float] = field(default_factory=list)
    best_iter: int = -1
    best_cost: float = float("inf")


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class MPPIGPS:
    """Unified MPPI-GPS trainer.

    ``deterministic=True`` → DeterministicPolicy + MSE; default → GaussianPolicy
    + NLL. Prior derived from policy + ``GPSConfig.policy_prior_type``.
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
        # Cross-iter replay (FIFO row-capped). Empty when cap == 0.
        self._episode_buffer: list[dict] = []

        # KL-adaptive α (MDGPS dual). None → schedule path.
        self._kl_alpha: float | None = None

        # Policy-trust dampener. Adaptive cold-start at min; else constant max.
        self._policy_trust: float = (
            float(gps_cfg.policy_trust_min)
            if gps_cfg.adaptive_policy_trust
            else float(gps_cfg.policy_trust_max)
        )

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
        """Distill: Det → MSE (+ grad clip / label clamp); Gauss → NLL
        (+ prev-iter KL trust region) or PPO clip when ``clip_ratio > 0``."""
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

        # Clamp log-ratio so exp() can't overflow → 0·∞ NaN gradients
        # via torch.min's autograd chain. exp(20) ≈ 5e8 is well outside
        # any clip threshold, so the PPO regime is unaffected.
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

        # Per-iter CSV log (flushed per row so a hard kill keeps prior rows).
        csv_columns = [
            "iter", "alpha", "base_alpha", "policy_trust", "policy_trust_next",
            "raw_mppi_eval_cost",
            "mppi_cost", "distill_loss",
            "kl_est", "kl_target",
            "prev_iter_kl",
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

        # α=0 rollout h5 cache; lazy-init on first α=0 iter. Sensors
        # dataset omitted when the env exposes none (matches collect_bc_demos).
        h5_file: h5py.File | None = None
        h5_cache_path = run_dir / "mppi_rollouts.h5" if run_dir is not None else None
        nsensor = 0
        probe_sd = _try_capture_sensordata(self.env)
        if probe_sd is not None:
            nsensor = int(probe_sd.shape[-1])

        initial_conditions = self._sample_initial_conditions(cfg.num_conditions)

        # KL-adaptive α: Gaussian only (needs σ for closed-form KL).
        kl_adaptive_enabled = (
            cfg.kl_target > 0.0 and not self._deterministic
        )

        outer_bar = tqdm(range(num_iterations), desc="GPS", unit="iter")
        for iteration in outer_bar:
            # α: schedule during warmup / when adaptive off; dual var
            # ``self._kl_alpha`` after warmup if adaptive on.
            use_kl_adaptive_this_iter = (
                kl_adaptive_enabled and iteration >= cfg.alpha_warmup_iters
            )
            if use_kl_adaptive_this_iter:
                if self._kl_alpha is None:
                    self._kl_alpha = schedule_alpha(iteration, cfg)
                base_alpha = self._kl_alpha
            else:
                base_alpha = schedule_alpha(iteration, cfg)

            # Multiplicative policy-trust dampener on top of α.
            policy_trust_iter = float(self._policy_trust)
            alpha = float(base_alpha) * policy_trust_iter

            all_obs: list[np.ndarray] = []
            all_actions: list[np.ndarray] = []
            all_weights: list[np.ndarray] = []
            condition_costs: list[float] = []

            # KL-adaptive: (obs, μ_p, σ_p²) per plan_step. Empty if disabled.
            kl_obs_buf: list[np.ndarray] = []
            kl_mu_p_buf: list[np.ndarray] = []
            kl_var_p_buf: list[np.ndarray] = []

            # ========== C-STEP ==========
            # eval() so prior_fn's forward is deterministic (no Dropout).
            self.policy.eval()

            c_step_total = cfg.num_conditions * cfg.episode_length
            c_bar = tqdm(
                total=c_step_total,
                desc=f"  iter {iteration:3d} C-step",
                unit="step",
                leave=False,
            )

            # α=0 → executor MPPI ≡ plain MPPI; cache to the h5.
            cache_iter = (alpha == 0.0) and (run_dir is not None)
            if cache_iter:
                iter_states = np.zeros(
                    (cfg.num_conditions, cfg.episode_length, self.env.obs_dim),
                    dtype=np.float32,
                )
                iter_actions = np.zeros(
                    (cfg.num_conditions, cfg.episode_length, self.env.action_dim),
                    dtype=np.float32,
                )
                iter_costs = np.zeros(
                    (cfg.num_conditions, cfg.episode_length), dtype=np.float32,
                )
                iter_sensors = (
                    np.zeros(
                        (cfg.num_conditions, cfg.episode_length, nsensor),
                        dtype=np.float32,
                    )
                    if nsensor > 0 else None
                )
            else:
                iter_states = iter_actions = iter_costs = iter_sensors = None

            for ic_idx, ic_state in enumerate(initial_conditions):
                # Fresh prior closure (captures current policy weights + α).
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

                # Sub-episodes split at every `done`.
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

                    # Capture sensordata BEFORE plan_step (row alignment).
                    if cache_iter and iter_sensors is not None:
                        sd = _try_capture_sensordata(self.env)
                        if sd is not None:
                            iter_sensors[ic_idx, t] = sd

                    # Executor (MPPI with prior).
                    action_exec, _info = self.mppi.plan_step(state, prior=prior_fn)

                    # KL-adaptive snapshot from unbiased posterior;
                    # full-replan steps only (open-loop skips _last_*).
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

                    # DAgger relabel: prior-free dry_run; no-op when α==0.
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

                    if cache_iter:
                        iter_states[ic_idx, t] = obs
                        iter_actions[ic_idx, t] = action_exec
                        iter_costs[ic_idx, t] = cost

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

            # α=0 rollout cache append (lazy-create on first α=0 iter).
            if cache_iter:
                T = cfg.episode_length
                M_iter = cfg.num_conditions
                if h5_file is None:
                    h5_file = h5py.File(h5_cache_path, "w")
                    h5_file.create_dataset(
                        "states", data=iter_states,
                        maxshape=(None, T, self.env.obs_dim), chunks=True,
                    )
                    h5_file.create_dataset(
                        "actions", data=iter_actions,
                        maxshape=(None, T, self.env.action_dim), chunks=True,
                    )
                    h5_file.create_dataset(
                        "costs", data=iter_costs,
                        maxshape=(None, T), chunks=True,
                    )
                    if iter_sensors is not None:
                        h5_file.create_dataset(
                            "sensordata", data=iter_sensors,
                            maxshape=(None, T, nsensor), chunks=True,
                        )
                    h5_file.attrs["env"] = type(self.env).__name__
                    h5_file.attrs["T"] = T
                    h5_file.attrs["obs_dim"] = self.env.obs_dim
                    h5_file.attrs["act_dim"] = self.env.action_dim
                    h5_file.attrs["M"] = M_iter
                    h5_file.attrs["source"] = "run_gps_alpha0"
                else:
                    n0 = h5_file["states"].shape[0]
                    n1 = n0 + M_iter
                    h5_file["states"].resize((n1, T, self.env.obs_dim))
                    h5_file["states"][n0:n1] = iter_states
                    h5_file["actions"].resize((n1, T, self.env.action_dim))
                    h5_file["actions"][n0:n1] = iter_actions
                    h5_file["costs"].resize((n1, T))
                    h5_file["costs"][n0:n1] = iter_costs
                    if iter_sensors is not None and "sensordata" in h5_file:
                        h5_file["sensordata"].resize((n1, T, nsensor))
                        h5_file["sensordata"][n0:n1] = iter_sensors
                    h5_file.attrs["M"] = n1
                h5_file.flush()

            # ========== S-STEP ==========
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

            # Pre-S-step snapshot for prev-iter KL (Gaussian-only).
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

            # Diagnostic prev-iter KL, capped at 4096 rows.
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

            if cfg.reset_optim_per_iter:
                self.policy.reset_optimizer()

            # MDGPS dual ascent on α toward kl_target.
            kl_est_iter = float("nan")
            if use_kl_adaptive_this_iter and kl_obs_buf:
                # σ_p floor in NORMALIZED action space. MPPI's per-dim
                # proposal std `cfg.noise_sigma * env.noise_scale` divided
                # by `_act_scale` (also a per-dim half-range) ≈
                # `cfg.noise_sigma` when noise_scale matches the action-
                # bound half-range (true for Adroit). 0 disables.
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
            history.iteration_prev_iter_kl.append(prev_iter_kl_diag)
            history.iteration_alpha.append(float(alpha))
            history.iteration_kl_est.append(kl_est_iter)

            # ========== Eval (drives best.pt) ==========
            eval_every = max(int(getattr(cfg, "eval_every", 1)), 1)
            is_last = (iteration == num_iterations - 1)
            do_eval = is_last or ((iteration + 1) % eval_every == 0)

            eval_cost = float("nan")
            raw_mppi_eval_cost: float = float("nan")
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

            # Raw-MPPI baseline (no prior); feeds the trust update.
            if do_eval and cfg.policy_trust_eval_mppi_eps > 0:
                mppi_stats = evaluate_mppi(
                    self.env, self.mppi,
                    n_episodes=cfg.policy_trust_eval_mppi_eps,
                    episode_len=cfg.eval_ep_len,
                    seed=iteration,
                )
                raw_mppi_eval_cost = float(mppi_stats["mean_cost"])
            history.iteration_raw_mppi_eval_cost.append(raw_mppi_eval_cost)

            if run_dir is not None:
                iter_path = run_dir / f"iter_{iteration:03d}.pt"
                save_checkpoint(
                    iter_path, self.policy,
                    iteration=iteration,
                    mppi_cost=mppi_cost,
                    eval_cost=eval_cost,
                    distill_loss=loss,
                    prev_iter_kl=prev_iter_kl_diag,
                )
                if do_eval and eval_cost < history.best_cost:
                    history.best_cost = eval_cost
                    history.best_iter = iteration
                    copy_as(iter_path, run_dir / "best.pt")

            # Trust update post-eval; history logs trust USED this iter.
            history.iteration_policy_trust.append(policy_trust_iter)
            if do_eval and cfg.adaptive_policy_trust:
                policy_cost_for_trust = (
                    eval_cost if not math.isnan(eval_cost) else None
                )
                raw_mppi_cost_for_trust = (
                    raw_mppi_eval_cost
                    if not math.isnan(raw_mppi_eval_cost) else None
                )
                self._policy_trust = compute_policy_trust(
                    policy_cost=policy_cost_for_trust,
                    raw_mppi_cost=raw_mppi_cost_for_trust,
                    eval_episode_len=cfg.eval_ep_len,
                    cfg=cfg,
                )

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
            if cfg.adaptive_policy_trust:
                postfix["trust"] = f"{policy_trust_iter:.2f}→{self._policy_trust:.2f}"
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
            if not self._deterministic and cfg.prev_iter_kl_coef > 0.0:
                base_line += f"  prev_kl={prev_iter_kl_diag:.4f}"
            if cfg.adaptive_policy_trust:
                base_line += (
                    f"  trust={policy_trust_iter:.3f}→{self._policy_trust:.3f}"
                )
            if do_eval:
                base_line += f"  eval_cost={eval_cost:8.2f}"
                if not math.isnan(raw_mppi_eval_cost):
                    base_line += f"  raw_mppi_cost={raw_mppi_eval_cost:8.2f}"
            tqdm.write(base_line)

            # Per-iter CSV row (NaN for inactive cols).
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
                    "base_alpha": float(base_alpha),
                    "policy_trust": float(policy_trust_iter),
                    "policy_trust_next": float(self._policy_trust),
                    "raw_mppi_eval_cost": raw_mppi_eval_cost,
                    "mppi_cost": mppi_cost,
                    "distill_loss": float(loss),
                    "kl_est": kl_est_iter,
                    "kl_target": (
                        float(cfg.kl_target) if use_kl_adaptive_this_iter
                        else float("nan")
                    ),
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
        if h5_file is not None:
            h5_file.close()
        return history
