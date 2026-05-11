"""dataclass configurations for mppi-gps"""
import json
from dataclasses import dataclass
from pathlib import Path

_CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"

@dataclass
class MPPIConfig:
    K: int = 256                  # number of samples
    H: int = 50                   # planning horizon
    lam: float = 1.0              # temperature: lower → more focus on best samples
    noise_sigma: float = 0.5      # exploration noise std
    adaptive_lam: bool = False    # adapt lam to maintain n_eff
    n_eff_threshold: float = 64.0 # target effective sample count
    # Open-loop execution: replan once, then execute this many actions from
    # the resulting nominal U before replanning. 1 = replan every step (MPC).
    open_loop_steps: int = 1
    # Optional full action covariance for MPPI sampling. None → per-dim
    # diagonal `diag((noise_sigma * env.noise_scale)²)`. When set, must be
    # symmetric PD `(action_dim, action_dim)`; `noise_sigma` and
    # `env.noise_scale` are ignored (pre-bake any per-dim scaling into it).
    # Use to correlate exploration across dims (e.g. arm + finger close
    # together) so MPPI discovers coordinated motions.
    noise_cov: list[list[float]] | None = None

    @staticmethod
    def load(env_name: str) -> "MPPIConfig":
        """Load best tuned params from configs/<env_name>_best.json.

        Keys starting with `_` are treated as documentation metadata and
        stripped before instantiating the dataclass.
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
    # Tanh squash on the action head (Gaussian: on `mu` only; Deterministic:
    # appended to the MLP). Output is bounded to (-1, 1) in NORMALIZED
    # action space; `_to_phys` denormalizes. `eval_checkpoint`, `run_dagger`,
    # `test_sl`, and `run_gps` all expose `--disable-tanh`.
    tanh_squash: bool = True
    # Input featurization: "running_norm" (default; learned RunningNormalizer)
    # or "hand_crafted" (env-aware transform in `src.policy.featurize_obs`).
    # In "hand_crafted" mode the RunningNormalizer is bypassed.
    featurize: str = "running_norm"
    # Per-hidden-layer regularisation. On by default — biased toward
    # robustness for high-DoF envs. Flip both off for small envs (point_mass,
    # acrobot) to match upstream's plain MLP.
    use_dropout: bool = True
    use_layernorm: bool = True
    # Dropout probability (when `use_dropout=True`). None → class default
    # (0.2 det, 0.1 gaussian).
    dropout_p: float | None = None

    @classmethod
    def for_env(cls, env_name: str) -> "PolicyConfig":
        """Per-env defaults. Strict superset of fields for both policy
        classes — `log_sigma_*` is Gaussian-only and silently ignored by
        the deterministic head."""
        return cls()

@dataclass
class GPSConfig:
    num_iterations: int = 50
    num_conditions: int = 5         # number of initial states
    episode_length: int = 500       # steps per episode during GPS training
    # Target weight on the policy prior in MPPI cost. With the default
    # `alpha_schedule="constant"` this is the per-iter alpha verbatim.
    # With a schedule, this is the plateau value after `alpha_warmup_iters`.
    policy_augmented_alpha: float = 0.1
    distill_batch_size: int = 1024
    distill_epochs: int = 3
    warm_start_policy: bool = False  # warm-start MPPI nominal U from policy rollout
    # Vestigial: Gaussian S-step always uses NLL via `train_weighted`;
    # Deterministic always uses MSE. Field kept so legacy config.json
    # files load without dataclass init errors.
    distill_loss: str = "nll"
    # Policy-prior shape for `MPPIGPS`:
    #   "auto" → "nll" for Gaussian, "mean_distance" for Deterministic.
    #   "nll" → -alpha * Σ_t log π(u|o). Gaussian-only.
    #   "mean_distance" → alpha * Σ_t ‖a − π.action(o)‖². Both classes.
    policy_prior_type: str = "auto"
    # On env termination mid-episode: reset to a fresh init and keep collecting
    # until episode_length steps. Required for terminating envs (hopper).
    auto_reset: bool = False
    # Periodic policy eval. `best.pt` is selected on this cost — NOT on the
    # C-step MPPI rollout cost (teacher under prior is not the student).
    n_eval_eps: int = 10
    eval_ep_len: int = 1000
    eval_every: int = 1
    # Cross-iteration distillation replay buffer in (obs, action) rows.
    # 0 = disabled. >0 = aggregate sub-episodes across iters with FIFO
    # eviction by whole episode (never split mid-way).
    distill_buffer_cap: int = 0
    # Recreate Adam at end of each GPS iter, wiping m/v moments. Each iter
    # is a new supervised task (new C-step data + prior shift), so stale
    # momentum can point in a stale direction and fight `prev_iter_kl_coef`.
    reset_optim_per_iter: bool = True
    # Trust-region KL penalty against the previous-iter policy. Adds
    # `coef * E_obs[KL(π_θ || π_prev)]` to the S-step loss. Gaussian-only
    # (closed-form diag-Gaussian KL); silent no-op for deterministic.
    prev_iter_kl_coef: float = 0.1
    # DAgger-style relabel: at each C-step timestep, run MPPI twice — once
    # with the policy prior (executor, steers env) and once without
    # (side-effect-free, its action is the training label). Decouples
    # "where we visit" from "what teacher says". With open_loop_steps > 1
    # the label call forces a full rollout every step.
    dagger_relabel: bool = False
    # Action-target clip used by the deterministic MSE S-step. At the start
    # of each S-step, snapshot the policy as `old_policy`; per batch the
    # MPPI label is clamped to `[old_policy.action(o) ± clip_eps]` before
    # MSE. Action-space trust region on per-update label movement. 0 disables.
    # Prefer `grad_clip_norm` unless you specifically want action-space TR.
    clip_eps: float = 0.3
    # L2 grad-norm clip in deterministic policy's mse_step (between
    # backward and optimizer.step). 0 = disabled. 1.0 is a sensible default
    # for normalized-action envs.
    grad_clip_norm: float = 1.0
    # Alpha schedule. "constant" (default) or warmup_iters <= 0 → fixed α.
    # Otherwise α ramps from `alpha_start` to `policy_augmented_alpha` over
    # the first `alpha_warmup_iters` iters and holds. Shapes:
    #   linear      — x
    #   smoothstep  — x²(3 − 2x)  (default ramp; deriv 0 at endpoints)
    #   cosine      — 0.5(1 − cos πx)
    # Rationale: at iter 0 the policy is untrained, so the prior pulls
    # MPPI toward random behavior — starting α at 0 lets MPPI explore
    # freely while the policy bootstraps.
    alpha_schedule: str = "constant"
    alpha_warmup_iters: int = 0     # ramp duration; 0 disables
    alpha_start: float = 0.0

    # KL-adaptive α (MDGPS-style dual variable, unbiased operand).
    # When kl_target > 0 (and policy is Gaussian), α is auto-adjusted each
    # iter to maintain a target KL between MPPI's unbiased teacher and the
    # global policy. The schedule fields above are used only during warmup;
    # after that, the KL-adaptive rule takes over.
    #
    # Per iter (after C-step):
    #   kl_est = E_state[KL(N(μ_p(s), σ_p²(s)) || π_θ(·|s))]
    # where (μ_p, σ_p²) is the weighted mean/var of MPPI's first-step
    # action samples re-weighted by the COST-ONLY posterior (prior
    # stripped — see `MPPI._last_unbiased_weights`). Then in log-α space:
    #   α ← α / kl_step_rate    if kl_est > kl_target  (let MPPI escape bad policy)
    #   α ← α · kl_step_rate    if kl_est < kl_target  (lean on policy when reliable)
    #   α ← clip(α, kl_alpha_min, kl_alpha_max)
    #
    # Direction inverted vs vanilla MDGPS: the biased operand variant has
    # a degenerate fixed point (as α → ∞, biased MPPI → π_θ so KL → 0
    # regardless of policy quality). Stripping the prior breaks the
    # self-reference.
    #
    # `kl_target` is summed over action dims — high-DoF envs need
    # proportionally larger targets (~act_dim × 0.05). Gaussian-only.
    kl_target: float = 0.0
    kl_alpha_min: float = 0.001
    kl_alpha_max: float = 0.5
    kl_step_rate: float = 1.5
    # Per-dim lower bound on the local policy std σ_p, as a fraction of
    # MPPI's proposal std (in normalized action space). When MPPI's softmin
    # concentrates onto a few samples, weighted-sample variance crashes
    # toward zero and log(σ_θ/σ_p) explodes, biasing `kl_est` upward.
    # 0 disables the floor.
    kl_sigma_floor_frac: float = 0.5

    # Policy-trust α dampener (port of upstream's `compute_policy_trust`).
    # Multiplicative scale on whatever α the schedule (or KL-adaptive rule)
    # returns. Recomputed each eval iter from the policy↔MPPI cost gap so
    # the prior is off while the policy is bad and ramps in as it catches up.
    #
    # Per eval iter:
    #     j_policy = policy_eval_cost / eval_ep_len     (per-step)
    #     j_mppi   = raw_mppi_eval_cost / eval_ep_len   (per-step)
    #     j_bad    = policy_trust_bad_cost_per_step
    #     quality  = clip((j_bad - j_policy) / max(j_bad - j_mppi, 1e-8), 0, 1)
    #     trust    = policy_trust_min + (policy_trust_max - policy_trust_min) * quality
    # Effective α for the next C-step is `trust * base_alpha`. Trust is
    # frozen on non-eval iters.
    #
    # `adaptive_policy_trust=False` (default) → trust ≡ `policy_trust_max`.
    adaptive_policy_trust: bool = False
    policy_trust_min: float = 0.0
    policy_trust_max: float = 1.0
    # Per-step cost above which the policy is "as bad as no policy". Set
    # a healthy margin above raw MPPI's per-step cost.
    policy_trust_bad_cost_per_step: float = 0.25
    # Number of raw-MPPI baseline episodes per eval iter, used to feed the
    # trust update. 0 disables (under adaptive, trust then pins at min).
    policy_trust_eval_mppi_eps: int = 1

    # PPO-style probability ratio clip for the Gaussian S-step surrogate.
    # 0.0 = disabled (falls through to plain NLL via `train_weighted`).
    # When > 0, the per-batch loss is
    #   L = -E[min(r·w, clip(r, 1-eps, 1+eps)·w)],  r = π_θ / π_old.
    # Typical 0.1–0.3; standard PPO uses 0.2.
    clip_ratio: float = 0.2


@dataclass
class DAggerConfig:
    dagger_iters: int = 10
    rollouts_per_iter: int = 20
    episode_len: int = 200
    beta_schedule: str = "linear"        # "linear" (1→0 by iter K/2) or "constant_zero"
    buffer_cap: int = 200_000
    distill_epochs: int = 20
    batch_size: int = 4096
    val_frac: float = 0.1
    n_eval_eps: int = 10
    eval_ep_len: int = 500
    seed: int = 0
    auto_reset: bool = False             # keep collecting through termination up to episode_len
    # Wipe Adam moments at end of each DAgger round. Useful when buffer
    # shifts distribution enough that stale momentum becomes a liability.
    reset_optim_per_iter: bool = False
    # L2 grad-norm clip inside the deterministic policy's mse_step
    # (warmup + finetune). 0 disables. Only active with DeterministicPolicy
    # — Gaussian dagger uses `clip_ratio` instead.
    grad_clip_norm: float = 1.0
    # PPO-style ratio clip for the GAUSSIAN dagger finetune. At round
    # start we snapshot the policy; per batch the surrogate
    #     L = -E[min(r, clip(r, 1-eps, 1+eps))],  r = π_θ / π_old
    # caps how much each (obs, expert action) pair can boost its log-
    # likelihood per step. 0 = plain MSE-on-mean. NOT applied in `warmup`
    # (random-init policy → meaningless ratio). Only active with GaussianPolicy.
    clip_ratio: float = 0.0
