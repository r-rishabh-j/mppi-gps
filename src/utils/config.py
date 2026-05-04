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

    @staticmethod
    def load(env_name: str) -> "MPPIConfig":
        """Load best tuned params from configs/<env_name>_best.json."""
        path = _CONFIGS_DIR / f"{env_name}_best.json"
        params = json.loads(path.read_text())
        return MPPIConfig(**params)

@dataclass
class PolicyConfig:
    hidden_dims: tuple[int, ...] = (256, 256)
    lr: float = 5e-4
    activation: str = "relu"
    obs_norm: bool = True
    log_sigma_min: float = -5.0
    log_sigma_max: float = 2.0

@dataclass
class GPSConfig:
    num_iterations: int = 50
    num_conditions: int = 5         # number of initial states
    episode_length: int = 500       # steps per episode during GPS training
    policy_augmented_alpha: float = 0.1 # weight on the policy prior in MPPI cost
    distill_batch_size: int = 256   # mini-batch size for policy distillation
    distill_epochs: int = 30         # gradient epochs per GPS iteration
    warm_start_policy: bool = False  # warm-start MPPI nominal U from policy rollout
    distill_loss: str = "nll"       # "nll" (weighted NLL on policy log-prob) or "mse" (MSE on mean)
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
    # Legacy: action-target clip used only by the MSE branch of
    # `mppi_gps_clip._train_step_clipped` (clamps the label to
    # [pi_old(o) ± clip_eps] before the MSE loss). Kept for that branch's
    # back-compat; `mppi_gps_det.MPPIGPSDet` no longer uses it (was
    # removed in favour of `grad_clip_norm` — loss-agnostic, no biased
    # fixed point at the boundary).
    clip_eps: float = 0.1
    # Gradient-norm clip applied between backward() and optimizer.step() in
    # the deterministic policy's mse_step (used by `mppi_gps_det.MPPIGPSDet`).
    # Bounds typical per-update parameter movement; loss-agnostic. 0.0 =
    # disabled. 1.0 is a sensible default for normalized-action envs;
    # sweep {0.5, 1.0, 5.0} if the policy learns too slowly or the loss
    # spikes. No effect on the Gaussian path.
    grad_clip_norm: float = 1.0


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

