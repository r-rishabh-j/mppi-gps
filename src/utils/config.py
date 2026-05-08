"""dataclass configurations for mppi-gps"""
import json
from dataclasses import dataclass
from pathlib import Path

_CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"

@dataclass 
class MPPIConfig:
    K: int = 256 # number of samples 
    H: int = 50 # planning horizon 
    lam: float = 1.0 # temperature parameter 
    # this parameter essentially helps you know how much you want to focus on specific samples vs others 
    noise_sigma: float = 0.5 # exploration noise std
    adaptive_lam: bool = False # adapt lam in order to maintain the n_eff
    n_eff_threshold: float = 64.0 # number of samples that you want to contribute to the weighted mean
    # open-loop execution horizon: replan once, then execute this many actions from the
    # resulting nominal U before replanning. 1 (default) = replan every step (MPC).
    # Intermediate calls skip the sample/rollout/weight pipeline, so wall-clock drops
    # roughly linearly in open_loop_steps at the cost of reactivity. Must be in [1, H].
    open_loop_steps: int = 1
    # Optional full action covariance for the MPPI sampling distribution.
    # When None (default), the noise model is the per-dim diagonal
    # `diag((noise_sigma * env.noise_scale)²)` — current behaviour, byte-
    # identical for all existing envs.
    # When set, must be a list-of-lists of shape `(action_dim, action_dim)`
    # — symmetric, positive-definite. Used directly as Σ for sampling
    # `ε ~ N(0, Σ)` (Cholesky) and for the IS correction `u^T·Σ⁻¹·ε`
    # (precision matrix). `noise_sigma` and `env.noise_scale` are
    # **ignored** when `noise_cov` is provided — pre-bake any per-dim
    # scaling into the matrix you supply.
    # Use case: correlate exploration across action dims (e.g. "extend
    # arm + close fingers together" via off-diagonal entries between
    # arm-z and finger-close dims) to discover coordinated grasp+lift
    # moves that independent per-dim noise rarely samples.
    noise_cov: list[list[float]] | None = None

    @staticmethod
    def load(env_name: str) -> "MPPIConfig":
        """Load best tuned params from configs/<env_name>_best.json.

        Keys starting with ``_`` are treated as documentation metadata
        (e.g. ``_generated_by``, ``_hand_synergy`` from the noise-cov
        generator) and stripped before instantiating the dataclass, so
        configs can carry self-documenting fields without breaking the
        loader.
        """
        path = _CONFIGS_DIR / f"{env_name}_best.json"
        params = json.loads(path.read_text())
        params = {k: v for k, v in params.items() if not k.startswith("_")}
        return MPPIConfig(**params)

@dataclass
class PolicyConfig:
    hidden_dims: tuple[int, ...] = (256, 256)
    lr: float = 5e-4
    activation: str = "relu"
    obs_norm: bool = True
    log_sigma_min: float = -3.0
    log_sigma_max: float = 0.0

    @classmethod
    def for_env(cls, env_name: str) -> "PolicyConfig":
        """Per-env defaults so the small/2D envs and the 30-DoF Adroit hand
        don't have to share one capacity setting.

        The fields are a strict superset for both ``GaussianPolicy`` and
        ``DeterministicPolicy``: ``hidden_dims``, ``lr``, ``activation``,
        ``obs_norm`` are read by both; ``log_sigma_*`` is Gaussian-only and
        silently ignored by the deterministic head. So a single per-env
        instance works for either policy class.

        Adroit-specific bumps:
        * ``hidden_dims=(512, 512, 512)`` — 30-DoF action with 70+-D obs
          needs more capacity than the (256, 256) default that's tuned for
          the small envs (acrobot, point_mass, hopper).
        * ``lr=3e-4`` — slightly safer with the deeper net.
        """
        # if env_name.startswith("adroit"):
        #     return cls(hidden_dims=(512, 512), lr=8e-4)
        return cls()

@dataclass
class GPSConfig:
    num_iterations: int = 50
    num_conditions: int = 5         # number of initial states
    episode_length: int = 500       # steps per episode during GPS training
    # Target weight on the policy prior in MPPI cost. With the default
    # `alpha_schedule="constant"` this is the per-iter alpha verbatim.
    # When a schedule is set (linear / smoothstep / cosine), this is the
    # *plateau* value reached after `alpha_warmup_iters` iterations —
    # see the schedule fields below and `mppi_gps_unified.schedule_alpha`.
    policy_augmented_alpha: float = 0.1
    distill_batch_size: int = 1024   # mini-batch size for policy distillation
    distill_epochs: int = 30         # gradient epochs per GPS iteration
    warm_start_policy: bool = False  # warm-start MPPI nominal U from policy rollout
    # NOTE: ``distill_loss`` is vestigial after the project-wide
    # "always NLL for Gaussian" sweep. Gaussian S-step always uses
    # weighted NLL via ``GaussianPolicy.train_weighted``; Deterministic
    # always uses MSE via ``DeterministicPolicy.mse_step``. The field is
    # kept (default "nll") so legacy ``config.json`` files load without
    # dataclass init errors, but it no longer drives any branch in the
    # trainers (``mppi_gps_unified.py``, ``mppi_gps.py``, ``mppi_gps_clip.py``).
    distill_loss: str = "nll"
    # Policy-prior shape used by the unified MPPIGPS trainer
    # (`src/gps/mppi_gps_unified.py`). Choices:
    #   * "auto" → "nll" for GaussianPolicy, "mean_distance" for DeterministicPolicy.
    #   * "nll" → prior = -alpha * Σ_t log π(u|o). Gaussian-only; raises for det.
    #   * "mean_distance" → prior = alpha * Σ_t ‖a − π.action(o)‖². Works for both.
    # The legacy trainers (mppi_gps.py / mppi_gps_clip.py / mppi_gps_det.py) ignore
    # this field — each hard-codes its own prior.
    policy_prior_type: str = "auto"
    auto_reset: bool = False        # On env termination mid-episode during the C-step,
                                    # reset to a fresh random init and keep collecting until
                                    # episode_length steps are taken. Required for terminating
                                    # envs like hopper; otherwise dataset size collapses early.
    # ---- Periodic policy evaluation during training ------------------------
    # After the S-step of each iteration, roll out the (greedy) policy for
    # `n_eval_eps` episodes of up to `eval_ep_len` steps and record its mean
    # cost. `best.pt` is selected on this eval cost — NOT on the C-step MPPI
    # rollout cost, which reflects the teacher (with policy prior), not the
    # student. `eval_every=1` evaluates every iteration; set higher to skip.
    # The final iteration is always evaluated.
    n_eval_eps: int = 10
    eval_ep_len: int = 1000
    eval_every: int = 1
    # ---- Cross-iteration distillation replay buffer ------------------------
    # Capacity measured in (obs, action) rows (same convention as DAggerConfig.
    # buffer_cap). 0 = disabled (current per-iter behaviour: data discarded
    # after S-step). >0 = aggregate sub-episodes across GPS iterations and
    # distill from the buffer. Sub-episodes split at each `done` boundary, so
    # with auto_reset=True a single condition can contribute many variable-
    # length sub-episodes. FIFO eviction — oldest WHOLE episode popped first
    # until total rows ≤ cap (we never split an episode mid-way).
    distill_buffer_cap: int = 0
    # ---- Stabilisers ------------------------------------------------------
    # EMA smoothing of the policy's trainable parameters. 0.0 = disabled.
    # Typical: 0.99–0.9999. When > 0 the S-step updates an exponential moving
    # average after every Adam step (`ema ← d·ema + (1-d)·θ`). Eval and
    # best.pt selection use the EMA snapshot, so checkpoints on disk reflect
    # the smoothed policy, not the noisy training weights.
    ema_decay: float = 0.99
    # When True (and ema_decay > 0): at the end of each S-step, permanently
    # overwrite θ with the EMA shadow (θ ← EMA). The next iteration's C-step
    # MPPI prior and the next S-step both start from the smoothed weights,
    # not the noisy end-of-S-step training trajectory. Without this flag the
    # EMA is shadow-only (just a smoothed snapshot for eval / best.pt).
    # Strongly recommended to pair with `reset_optim_per_iter=True` so Adam's
    # running moments don't carry stale estimates from the pre-sync θ.
    ema_hard_sync: bool = True
    # When True: recreate the policy's Adam optimizer at the end of each
    # GPS iteration's S-step, wiping accumulated m/v moments. Rationale:
    # (1) each GPS iter is a new supervised task (C-step data + prior shift
    # iter-to-iter), so momentum built against the prior iter's gradient
    # field can point in a stale direction; (2) when `ema_hard_sync` is on,
    # θ moves non-trivially at the boundary and Adam's moments were
    # estimated at the pre-sync θ; (3) accumulated momentum can blow
    # through the `prev_iter_kl_coef` trust region on the first few steps.
    # Default False preserves current behaviour.
    reset_optim_per_iter: bool = True
    # Trust-region-style KL penalty against the previous-iteration policy.
    # 0.0 = disabled (current behaviour). >0 = add
    #     coef * E_obs[ KL(π_θ(·|o) || π_{prev iter}(·|o)) ]
    # to the S-step distill loss. A deep copy of the policy at the start of
    # each S-step acts as π_prev. Only applied when distill_loss == "nll"
    # (closed-form diag-Gaussian KL); silently a no-op for MSE since the
    # deterministic mean-only loss has no policy distribution to constrain.
    prev_iter_kl_coef: float = 0.1
    # When True (and policy prior strength alpha > 0), the C-step runs TWO
    # MPPI calls per timestep: the first with the policy prior (executes the
    # env), the second without (side-effect-free dry_run, its action is the
    # training label). Decouples "steer where we visit" from "what the
    # teacher says" — fixes the self-reinforcing loop where the executed MPPI
    # action is already tilted toward the current policy. DAgger-style
    # relabeling inside the GPS loop. Default False preserves current conflated
    # behaviour. Correct under any MPPIConfig.open_loop_steps value, but with
    # open_loop_steps > 1 the label call forces a full rollout every step
    # (cached chunk actions are prior-biased and can't serve as plain labels),
    # so you lose the open-loop speedup — effective cost becomes ~1 rollout/step.
    dagger_relabel: bool = False
    # Action-target clip used by the deterministic / MSE S-step in
    # `mppi_gps_unified._distill_epoch` (and the legacy
    # `mppi_gps_clip._train_step_clipped`). At the start of each S-step
    # the trainer snapshots the policy as `old_policy`; per batch the
    # MPPI-supplied label is clamped to `[old_policy.action(o) ± clip_eps]`
    # before the MSE loss. This is a trust region on per-update label
    # movement — bounds how far each gradient step can pull the policy
    # mean. 0.0 = disabled (default; plain MSE on the raw MPPI label).
    # Typical 0.05–0.2 if you want to use it. NOTE: opt-in default is
    # deliberate — the previous 0.1 default silently activated the clip
    # for every deterministic GPS run, which can bias the fixed point
    # of the regression. `grad_clip_norm` is loss-agnostic and a less
    # biased alternative; prefer that unless you specifically want the
    # action-space trust region semantics.
    clip_eps: float = 0.3
    # Gradient-norm clip applied between backward() and optimizer.step() in
    # the deterministic policy's mse_step (used by `mppi_gps_det.MPPIGPSDet`).
    # Bounds typical per-update parameter movement; loss-agnostic. 0.0 =
    # disabled. 1.0 is a sensible default for normalized-action envs;
    # sweep {0.5, 1.0, 5.0} if the policy learns too slowly or the loss
    # spikes. No effect on the Gaussian path.
    grad_clip_norm: float = 1.0
    # ---- Alpha schedule for the policy-prior weight -----------------------
    # When `alpha_schedule == "constant"` (default) or `alpha_warmup_iters
    # <= 0`, behavior is byte-identical to the previous fixed-α path:
    # every iter uses `policy_augmented_alpha`. Otherwise, alpha ramps
    # from `alpha_start` (default 0.0) to `policy_augmented_alpha` over
    # the first `alpha_warmup_iters` iterations and stays constant
    # thereafter.
    #
    # Why: at iter 0 the policy is untrained, so the prior pulls MPPI
    # toward random behavior — poisoning the C-step dataset for several
    # iters until it FIFO-evicts. Starting α at 0 lets MPPI explore
    # freely while the policy bootstraps; ramping the prior in later
    # gives the on-policy state coverage a stable α provides.
    #
    # Choices: "constant" | "linear" | "smoothstep" | "cosine".
    # `smoothstep` (cubic, x²(3−2x)) is the default ramp shape: derivative
    # 0 at both endpoints, so α stays near 0 longer at the start and
    # plateaus into α_max smoothly with no jump-discontinuity.
    alpha_schedule: str = "constant"
    alpha_warmup_iters: int = 0     # ramp duration (in GPS iters); 0 disables
    alpha_start: float = 0.0        # starting α (typically 0.0)

    # ---- KL-adaptive α (MDGPS-style dual variable, unbiased operand) ------
    # When kl_target > 0 (and policy is Gaussian), the trainer treats α as
    # a DUAL VARIABLE auto-adjusted each iter to maintain a target KL
    # divergence between MPPI's *unbiased* teacher distribution and the
    # global policy. The schedule fields above are used only during the
    # warmup window (`alpha_warmup_iters`); after that, the KL-adaptive
    # rule takes over and `alpha_schedule` / `policy_augmented_alpha` /
    # `alpha_start` are ignored.
    #
    # Algorithm (per GPS iter, after the C-step):
    #   kl_est = E_state[ KL(N(μ_p(s), σ_p²(s)) || π_θ(·|s)) ]
    # where (μ_p, σ_p²) is the weighted mean/var of MPPI's first-step
    # action samples re-weighted by the COST-ONLY posterior (the prior
    # contribution is stripped from the importance weights — see
    # `MPPI._last_unbiased_weights`). Then dual gradient update in
    # log-α space, INVERTED relative to the biased-operand variant:
    #   α ← α / kl_step_rate    if kl_est > kl_target  (let MPPI escape a bad policy)
    #   α ← α · kl_step_rate    if kl_est < kl_target  (lean on policy when it matches teacher)
    #   α ← clip(α, kl_alpha_min, kl_alpha_max)
    #
    # Why this direction: with the *unbiased* operand, large `kl_est`
    # means "policy is far from the cost-optimal teacher" — i.e. the
    # policy is bad. We want MPPI to ignore it and explore (small α).
    # With the *biased* operand the direction would be the opposite, but
    # that variant has a degenerate fixed point: as α → ∞, biased MPPI
    # → π_θ by construction so KL → 0 mechanically, regardless of policy
    # quality. Stripping the prior breaks the self-reference.
    #
    # Why: replaces the manual α + schedule tuning with a data-driven
    # rule. For the "MPPI-trapped-in-bad-policy" failure mode: bad policy
    # → kl_unbiased large → α shrinks → MPPI explores freely next iter →
    # finds lift trajectories → buffer fills with unbiased data → policy
    # improves → kl_unbiased shrinks → α can grow → executor stabilises
    # around the now-good policy. The trap breaks by construction.
    #
    # 0 = disabled (default; standard schedule path).
    # ``kl_target`` is the per-state KL **summed over action dims** — so
    # high-DoF envs need proportionally larger targets. A useful rule of
    # thumb is ``act_dim × 0.05`` (≈ "policy mean within 0.3 σ_θ of MPPI
    # per dim, on average"):
    #     point_mass / acrobot (2-D)        ≈ 0.10
    #     hopper (3-D)                       ≈ 0.15
    #     adroit_pen / adroit_relocate (24/30-D) ≈ 1.2 — 1.5
    # Below ``act_dim × 0.05`` the dual update tends to saturate at
    # ``kl_alpha_max`` and the KL constraint becomes effectively
    # unsatisfiable (see ``kl_sigma_floor_frac`` for the σ_p collapse
    # that aggravates this).
    # Implemented Gaussian-only — needs σ from the policy for the closed-
    # form KL. Silently ignored under --deterministic.
    kl_target: float = 0.0
    kl_alpha_min: float = 0.001    # lower bound on dual α (allows multiplicative escape)
    kl_alpha_max: float = 0.5      # upper bound (prevents runaway). 0.5 is
                                   # a conservative cap; α ≫ 0.1 typically
                                   # crushes MPPI's exploration regardless
                                   # of the KL constraint, so growing past
                                   # it is rarely productive.
    kl_step_rate: float = 1.5      # multiplicative update rate per iter
    # Per-dim lower bound on the **local policy** std σ_p, expressed as a
    # fraction of MPPI's proposal std (``cfg.noise_sigma * env.noise_scale``,
    # in normalized action space ≈ ``cfg.noise_sigma``). When MPPI's softmin
    # concentrates onto a few samples, the weighted-sample variance crashes
    # toward zero and ``log(σ_θ/σ_p)`` in the closed-form Gaussian KL
    # explodes, biasing ``kl_est`` upward by tens-to-hundreds of nats per
    # state regardless of mean alignment. Flooring σ_p at e.g. 0.5 × the
    # proposal std encodes a sensible "the policy will face proposal noise
    # during execution, so its local distribution is at least that wide"
    # prior. 0 disables the floor (legacy behaviour: estimator clamps at
    # 1e-6 and returns near-pathological KL whenever ESS is low).
    kl_sigma_floor_frac: float = 0.5

    # PPO-style probability ratio clip for the Gaussian S-step surrogate
    # in `mppi_gps_unified._train_step_ppo_clip`. 0.0 = disabled (default;
    # falls through to plain NLL via `train_weighted`, which is what most
    # users expect). When > 0, snapshots the policy at the start of each
    # S-step and uses `L = -E[ min(r·w, clip(r, 1-eps, 1+eps)·w) ]` per
    # batch, where r = π_θ / π_old. Typical 0.1–0.3; standard PPO uses 0.2.
    # IMPORTANT: must default to 0.0 — `getattr(cfg, "clip_ratio", 0.2)`
    # in the trainer used to silently activate PPO when this field was
    # missing, and that path has a 0·∞ NaN-gradient bug when σ collapses
    # (extreme log-ratios overflow exp() to inf, the torch.min picks the
    # finite branch with gradient 0, and 0 · inf in the chain rule
    # produces NaN gradients that nuke every parameter in one step).
    clip_ratio: float = 0.2


@dataclass
class DAggerConfig:
    dagger_iters: int = 10               # number of DAgger rounds
    rollouts_per_iter: int = 20          # trajectories collected per round
    episode_len: int = 200               # steps per rollout trajectory
    beta_schedule: str = "linear"        # "linear" (1→0 by iter K/2) or "constant_zero"
    buffer_cap: int = 200_000            # max aggregated (obs, a*) pairs
    distill_epochs: int = 20             # finetune epochs per round
    batch_size: int = 4096
    val_frac: float = 0.1                # held-out fraction of most-recent round
    n_eval_eps: int = 10
    eval_ep_len: int = 500
    seed: int = 0
    auto_reset: bool = False             # on termination, reset and keep collecting
                                         # until episode_len steps are taken for the slot
                                         # (useful for terminating envs like hopper)
    # EMA smoothing of the policy's trainable parameters. 0.0 = disabled.
    # Typical: 0.99–0.9999. Eval uses the EMA snapshot and best.pt is copied
    # while the EMA weights are swapped in, so the checkpoint on disk matches
    # the reported eval cost. See src/policy/ema.py.
    ema_decay: float = 0.0
    # Hard-promote θ ← EMA at end of each DAgger round; see GPSConfig.
    ema_hard_sync: bool = False
    # Wipe Adam moments at end of each DAgger round; see GPSConfig.
    reset_optim_per_iter: bool = False
    # L2 gradient-norm clip applied inside the deterministic policy's
    # mse_step (between backward and optimizer.step). Bounds per-update
    # parameter movement; loss-agnostic, no biased estimator. Applied in
    # both `warmup()` and `finetune()`. ONLY active when `self.policy` is
    # a DeterministicPolicy — Gaussian dagger uses `clip_ratio` instead.
    # 0.0 = disabled. Default 1.0 matches `GPSConfig.grad_clip_norm`.
    grad_clip_norm: float = 1.0
    # PPO-style probability ratio clip for the GAUSSIAN dagger finetune,
    # mirroring `mppi_gps_clip._train_step_clipped`'s Gaussian branch.
    # At the start of `finetune()` we snapshot the policy; per batch the
    # surrogate
    #     L = - E[ min(r, clip(r, 1-eps, 1+eps)) ],   r = pi_theta / pi_old
    # is minimized — i.e. each (obs, expert_action) pair gets its log-
    # likelihood boosted up to ratio = 1+eps per step, then saturates.
    # Trust region in distribution space. ONLY active when `self.policy`
    # is GaussianPolicy AND `clip_ratio > 0`. Otherwise Gaussian dagger
    # falls through to plain MSE-on-mean. NOT applied during `warmup()`
    # (random-init policy → meaningless ratio). 0.0 = disabled (default
    # = plain MSE). Typical 0.1–0.3; standard PPO uses 0.2.
    clip_ratio: float = 0.0
    # NOTE: ``loss_type`` field removed in the "always NLL for Gaussian"
    # sweep. Gaussian DAgger now always uses NLL (via
    # ``GaussianPolicy.train_weighted`` with uniform weights), or the
    # PPO clip surrogate when ``clip_ratio > 0``. Deterministic DAgger
    # is always MSE on the mean — no σ to fit. Loading a config json
    # that includes ``loss_type`` will fail dataclass strict-init; if
    # you have such a file, drop the key (no other code reads it now).

