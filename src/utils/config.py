"""Dataclass configs for MPPI / GPS / DAgger / Policy."""
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
    # Actions per replan; 1 = MPC (replan every step).
    open_loop_steps: int = 1
    # Full action covariance. None → diagonal from noise_sigma · noise_scale.
    # When set: symmetric PD; noise_sigma / env.noise_scale are ignored.
    noise_cov: list[list[float]] | None = None

    @staticmethod
    def load(env_name: str) -> "MPPIConfig":
        """Load configs/<env_name>_best.json. Underscore keys are metadata."""
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
    # Tanh squash → (-1, 1) in normalized action space; ``_to_phys`` rescales.
    tanh_squash: bool = True
    # "running_norm" (default) or "hand_crafted" (bypasses normalizer).
    featurize: str = "running_norm"
    # Per-layer regularisation. Default on (high-DoF). Off for small envs.
    use_dropout: bool = True
    use_layernorm: bool = True
    # None → class default (0.2 det, 0.1 gaussian).
    dropout_p: float | None = None

    @classmethod
    def for_env(cls, env_name: str) -> "PolicyConfig":
        """Per-env defaults; superset of fields for both policy classes."""
        return cls()

@dataclass
class GPSConfig:
    num_iterations: int = 50
    num_conditions: int = 5         # number of initial states
    episode_length: int = 500       # steps per episode during GPS training
    # Plateau α; per-iter when ``alpha_schedule="constant"``.
    policy_augmented_alpha: float = 0.1
    distill_batch_size: int = 1024
    distill_epochs: int = 3
    warm_start_policy: bool = False
    # Legacy field: Gaussian → NLL, Deterministic → MSE always.
    distill_loss: str = "nll"
    # "auto" → "nll" (Gaussian) / "mean_distance" (Det.).
    policy_prior_type: str = "auto"
    # Required for terminating envs (hopper).
    auto_reset: bool = False
    # Eval drives best.pt; not the C-step MPPI rollout cost.
    n_eval_eps: int = 10
    eval_ep_len: int = 1000
    eval_every: int = 1
    # Cross-iter (obs, action) replay buffer, FIFO whole-episode eviction.
    distill_buffer_cap: int = 0
    # Wipe Adam moments per iter (each iter is a new supervised task).
    reset_optim_per_iter: bool = True
    # ``coef · E_obs[KL(π_θ || π_prev)]`` in the S-step loss; Gaussian only.
    prev_iter_kl_coef: float = 0.1
    # Per-step run MPPI twice: with prior (executor) + without (label).
    dagger_relabel: bool = False
    # Det MSE: clamp label to ``[old_policy.action(o) ± clip_eps]``.
    clip_eps: float = 0.3
    # L2 grad-norm clip in det mse_step. 0 disables.
    grad_clip_norm: float = 1.0
    # α schedule: "constant" / "linear" / "smoothstep" / "cosine".
    # Ramp from ``alpha_start`` to ``policy_augmented_alpha`` over
    # ``alpha_warmup_iters``; 0 disables.
    alpha_schedule: str = "constant"
    alpha_warmup_iters: int = 0
    alpha_start: float = 0.0

    # KL-adaptive α (MDGPS dual on unbiased teacher).
    # ``kl_est = E_state[KL(N(μ_p, σ_p²) || π_θ)]`` where (μ_p, σ_p²) is
    # the cost-only posterior over MPPI's first-step samples. Update:
    #   α ← α / kl_step_rate    if kl_est > kl_target
    #   α ← α · kl_step_rate    if kl_est < kl_target
    #   α ← clip(α, kl_alpha_min, kl_alpha_max)
    # Direction inverted vs vanilla MDGPS to break the biased fixed point.
    # ``kl_target`` is summed over action dims (~act_dim × 0.05). Gaussian only.
    kl_target: float = 0.0
    kl_alpha_min: float = 0.001
    kl_alpha_max: float = 0.5
    kl_step_rate: float = 1.5
    # Per-dim σ_p floor as a fraction of MPPI's proposal std. 0 disables.
    kl_sigma_floor_frac: float = 0.5

    # Policy-trust dampener: multiplicative on α, recomputed each eval iter
    # from the policy↔MPPI per-step cost gap (linear interp in [min, max]).
    # adaptive_policy_trust=False → trust ≡ policy_trust_max (constant).
    adaptive_policy_trust: bool = False
    policy_trust_min: float = 0.0
    policy_trust_max: float = 1.0
    # Per-step cost above which the policy is "as bad as no policy".
    policy_trust_bad_cost_per_step: float = 0.25
    # Raw-MPPI baseline episodes per eval iter; 0 disables.
    policy_trust_eval_mppi_eps: int = 1

    # PPO ratio clip for the Gaussian S-step. 0 → plain NLL.
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
    auto_reset: bool = False             # collect through termination up to episode_len
    # Wipe Adam moments per round.
    reset_optim_per_iter: bool = False
    # Det L2 grad-norm clip in mse_step. 0 disables.
    grad_clip_norm: float = 1.0
    # PPO ratio clip for the Gaussian DAgger finetune; not applied in warmup.
    clip_ratio: float = 0.0
