"""Train a reactive policy via MPPI-GPS and evaluate against baselines.

Creates a run dir `experiments/gps/<timestamp>_<env>_<name>/` containing:
  - config.json    CLI args + all dataclass configs + metadata
  - iter_<k>.pt    per-iter wrapped checkpoints
  - best.pt / final.pt
  - curves.json / curves.png
  - <env>.mp4      eval video

Usage:
    python -m scripts.run_gps --env acrobot --exp-name baseline
    python -m scripts.run_gps --env hopper --gps-iters 30 --alpha 0.05
    python -m scripts.run_gps --env acrobot --deterministic --alpha 0.05
    python -m scripts.run_gps --env acrobot --policy-prior mean_distance \
        --alpha 0.1 --gps-iters 20 --exp-name gauss_mean_dist
    python -m scripts.run_gps --env acrobot --exp-name warmstart \
        --init-ckpt checkpoints/dagger/<run>/best.pt
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import mediapy
import numpy as np
import torch

from src.envs import make_env
from src.gps.mppi_gps_unified import MPPIGPS
from src.gps.mppi_gps_warp import WarpMPPIGPS
from src.mppi.mppi import MPPI
from src.utils.config import GPSConfig, MPPIConfig, PolicyConfig
from src.utils.device import pick_device
from src.utils.evaluation import evaluate_mppi, evaluate_policy
from src.utils.seeding import seed_everything
from src.utils.experiment import (
    copy_as,
    git_sha,
    load_checkpoint,
    load_state_dict_into,
    make_run_dir,
    save_checkpoint,
    update_config,
    write_config,
)


_ENVS = ["acrobot", "adroit_pen", "adroit_relocate", "point_mass", "hopper", "ur5_push"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="acrobot", choices=_ENVS)
    p.add_argument("--warp", action="store_true",
                   help="Use mujoco_warp GPU batch rollout for MPPI's C-step. "
                        "Only `adroit_relocate` has a warp variant currently. "
                        "Requires `uv pip install warp-lang mujoco-warp` and "
                        "an NVIDIA GPU + CUDA (graph replay = the speedup). "
                        "`nworld` is pinned to MPPIConfig.K at env construction; "
                        "changing K means re-running the script.")
    p.add_argument("--gps-iters", type=int, default=None,
                   help="Number of GPS iterations (default: from GPSConfig)")
    p.add_argument("--num-conditions", type=int, default=None,
                   help="Number of initial conditions to train across")
    p.add_argument("--episode-length", type=int, default=None,
                   help="Steps per episode during GPS training")
    p.add_argument("--alpha", type=float, default=0.1,
                   help="Policy-augmented cost weight (0 = no augmentation). "
                        "When --alpha-schedule is set, this is the *plateau* "
                        "value reached after --alpha-warmup-iters; otherwise "
                        "the per-iter constant.")
    p.add_argument("--alpha-schedule", default='smoothstep',
                   choices=["constant", "linear", "smoothstep", "cosine"],
                   help="Per-iter α schedule. 'constant' (default) uses --alpha "
                        "verbatim every iter. 'smoothstep' / 'cosine' / 'linear' "
                        "ramp from --alpha-start to --alpha over the first "
                        "--alpha-warmup-iters iterations and stay constant "
                        "thereafter. Useful at iter 0 when the policy is "
                        "untrained: starting α at 0 lets MPPI explore freely "
                        "while the policy bootstraps, ramping the prior in "
                        "later for on-policy state coverage.")
    p.add_argument("--alpha-warmup-iters", type=int, default=None,
                   help="GPS iterations to ramp α from --alpha-start to "
                        "--alpha (default 0 = schedule disabled). Ignored "
                        "when --alpha-schedule is 'constant'.")
    p.add_argument("--alpha-start", type=float, default=0,
                   help="Starting α value for the ramp (default 0.0). "
                        "Ignored when --alpha-schedule is 'constant'.")
    p.add_argument("--kl-target", type=float, default=None,
                   help="MDGPS-style KL-adaptive α (Gaussian only). When > 0, "
                        "α becomes a dual variable auto-adjusted each iter to "
                        "drive E_state[KL(N(μ_p,σ_p²) ‖ π_θ)] toward this "
                        "target. The schedule fields above are still used for "
                        "the warmup window (alpha_warmup_iters); after that "
                        "the adaptive rule takes over. Per-state KL is summed "
                        "over action dims, so scale with act_dim — rule of "
                        "thumb act_dim × 0.05 (e.g. ~0.1 for acrobot 2-D, "
                        "~1.5 for adroit_relocate 30-D). 0 (default) disables "
                        "and uses the schedule path.")
    p.add_argument("--kl-alpha-min", type=float, default=None,
                   help="Lower bound on the KL-adaptive α (default 0.001). "
                        "Allows multiplicative escape from a tight constraint.")
    p.add_argument("--kl-alpha-max", type=float, default=None,
                   help="Upper bound on the KL-adaptive α (default 0.5). "
                        "Prevents runaway when MPPI is far from policy. "
                        "α ≫ 0.1 typically crushes MPPI's exploration "
                        "regardless of the constraint, so growing past 0.5 "
                        "is rarely productive.")
    p.add_argument("--kl-step-rate", type=float, default=None,
                   help="Multiplicative update rate for the dual α step "
                        "(default 1.5). Larger = faster adaptation but more "
                        "iter-to-iter oscillation; 2.0+ if you need to escape "
                        "a sticky α regime quickly.")
    p.add_argument("--kl-sigma-floor-frac", type=float, default=None,
                   help="Per-dim floor on the local-policy std σ_p in the "
                        "KL estimator, expressed as a fraction of MPPI's "
                        "proposal std (default 0.5). Prevents log(σ_θ/σ_p) "
                        "from exploding when MPPI's softmin concentrates "
                        "onto a few samples and weighted-sample variance "
                        "collapses toward zero. 0 disables the floor "
                        "(legacy behaviour: clamps at 1e-6, biases kl_est "
                        "upward by tens of nats per state).")
    p.add_argument("--policy-prior", default=None,
                   choices=["nll", "mean_distance"],
                   help="Policy prior shape used in the MPPI cost. Unset = "
                        "auto: nll for Gaussian, mean_distance for "
                        "--deterministic. Explicit values: 'nll' for "
                        "-alpha*Σ log π (Gaussian only), 'mean_distance' for "
                        "alpha*Σ‖a−π.action(o)‖² (works for both classes; the "
                        "only choice for --deterministic).")
    p.add_argument("--auto-reset", action="store_true",
                   help="On env termination during C-step rollout, reset to a fresh random "
                        "init and keep collecting until episode_length steps are taken for "
                        "the condition. Recommended for terminating envs like hopper.")
    p.add_argument("--warm-start-policy", action="store_true",
                   help="(Advanced) Seed MPPI nominal U from a policy rollout each condition. "
                        "Off by default — double-biases MPPI toward the policy and pads zeros "
                        "past termination; only useful late in training on non-terminating envs.")
    p.add_argument("--device", default="auto", help="auto | cpu | mps | cuda (policy only)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-eval", type=int, default=10,
                   help="Number of evaluation episodes for the final eval "
                        "(end-of-training comparison against MPPI).")
    p.add_argument("--eval-len", type=int, default=1000,
                   help="Max steps per evaluation episode (final eval + in-loop eval).")
    p.add_argument("--eval-every", type=int, default=1,
                   help="Run the per-iter policy eval every N iterations "
                        "(default from GPSConfig.eval_every). Last iter is always evaluated.")
    p.add_argument("--distill-buffer-cap", type=int, default=0,
                   help="Cross-iteration distillation buffer capacity, measured in "
                        "(obs, action) rows (matches DAgger's --buffer-cap convention). "
                        "0/unset = no buffering. Sub-episodes split at each done boundary, "
                        "so lengths vary with --auto-reset. FIFO eviction — oldest WHOLE "
                        "episode popped first until total rows ≤ cap.")
    p.add_argument("--ema-decay", type=float, default=0,
    # p.add_argument("--ema-decay", type=float, default=0.99,
                   help="Exponential moving average decay over policy trainable params "
                        "(e.g. 0.999). 0 or unset disables EMA. Eval and best.pt selection "
                        "use the EMA snapshot. Without --ema-hard-sync the shadow is purely "
                        "passive; MPPI's prior keeps using fresh training weights.")
    p.add_argument("--ema-hard-sync", action="store_true", default=True,
                   help="At end of each S-step, copy EMA shadow into the live policy (θ ← EMA) "
                        "so the next iteration's MPPI prior and S-step both start from the "
                        "smoothed weights. No effect unless --ema-decay > 0. Recommended to "
                        "pair with --reset-optim-per-iter for Adam-state consistency.")
    p.add_argument("--reset-optim-per-iter", action="store_true", default=True,
                   help="Recreate the policy's Adam optimizer at end of each GPS iteration, "
                        "wiping m/v moments. Recommended with --ema-hard-sync (moments are "
                        "stale after a hard-sync) and with --prev-iter-kl-coef (accumulated "
                        "momentum can blow through the trust region on the first few steps).")
    p.add_argument("--prev-iter-kl-coef", type=float, default=0,
    # p.add_argument("--prev-iter-kl-coef", type=float, default=0.05,
                   help="Trust-region-style KL penalty to the previous iteration's policy "
                        "(Gaussian only — no policy distribution under --deterministic). "
                        "Adds `coef * E_obs[KL(π_θ || π_prev)]` to the S-step loss. 0/unset "
                        "disables. Typical small values (e.g. 0.01-0.1). Silently ignored "
                        "with --deterministic.")
    p.add_argument("--dagger-relabel", action="store_true",
                   help="DAgger-style decoupled relabel: per C-step timestep, run MPPI once "
                        "WITH the policy prior (executor, steers the env) and a second time "
                        "WITHOUT the prior (side-effect-free dry_run, its action is the "
                        "training label). Removes the self-reinforcing loop where the "
                        "executed MPPI action is already tilted toward the current policy. "
                        "No-op when alpha == 0. Note: under open_loop_steps > 1 the label "
                        "call forces a full rollout every step (cached chunk actions are "
                        "prior-biased, so can't serve as plain labels) — wallclock per step "
                        "becomes ~1 rollout instead of 1/N.")
    p.add_argument("--deterministic", action="store_true",
                   help="Use a DeterministicPolicy student. Forces MSE distillation "
                        "(no NLL — there's no policy distribution). The policy prior "
                        "is locked to 'mean_distance' (alpha * Σ‖a − π.action(o)‖²); "
                        "--policy-prior nll raises if combined with this flag. "
                        "--prev-iter-kl-coef is silently ignored (no distribution).")
    p.add_argument("--grad-clip-norm", type=float, default=None,
                   help="L2 gradient-norm clip applied inside the deterministic "
                        "policy's MSE step (used by --deterministic / mppi_gps_det). "
                        "Bounds per-update parameter movement directly — loss-"
                        "agnostic, no biased estimator. 0 = disabled. Default 1.0 "
                        "(see GPSConfig.grad_clip_norm). Sweep {0.5, 1.0, 5.0} if "
                        "learning is too slow or loss spikes. Ignored without "
                        "--deterministic.")
    p.add_argument("--clip-eps", type=float, default=0.3,
                   help="Action-space trust region for the deterministic S-step "
                        "(--deterministic). At the start of each S-step the "
                        "trainer snapshots the policy as π_old; per batch the "
                        "MPPI label is clamped to [π_old.action(o) ± clip_eps] "
                        "before the MSE loss. 0/unset = disabled (default). "
                        "Typical 0.05–0.2. Mirrors mppi_gps_clip's MSE branch. "
                        "Ignored without --deterministic. Prefer --grad-clip-norm "
                        "unless you specifically want action-space (vs parameter-"
                        "space) trust-region semantics.")
    # ---- Policy architecture knobs (port from upstream `gaussian_policy.py` /
    # `deterministic_policy.py` — see PolicyConfig docstrings) ----
    p.add_argument("--disable-tanh", action="store_true",
                   help="Disable the default tanh squash on the policy's "
                        "action head (Gaussian: tanh on mu; Deterministic: "
                        "appended after the MLP). With tanh on (default), "
                        "the network output is bounded to (-1, 1) in "
                        "normalized action space and `_to_phys` denormalizes "
                        "to physical bounds — matches upstream's "
                        "`DeterministicPolicy`. Pass this to revert to the "
                        "legacy unbounded head + `act_np` clamp behaviour "
                        "(use when reproducing pre-tanh experiments).")
    p.add_argument("--featurize", default=None,
                   choices=["running_norm", "hand_crafted"],
                   help="Input pre-processor for the policy. 'running_norm' "
                        "(default) uses a learned per-dim RunningNormalizer; "
                        "'hand_crafted' uses the env-aware feature transform "
                        "in src.policy.featurize_obs — matches upstream's "
                        "`featurize_obs`. Supports 4-D acrobot (→ sin/cos + "
                        "scaled vel) and 6-D point_mass (→ delta-to-goal + "
                        "vel + goal, all normalized to ~[-1, 1]). For other "
                        "obs dims it's a no-op (identity).")
    p.add_argument("--no-dropout", action="store_true",
                   help="Disable Dropout between hidden layers of the policy. "
                        "Default keeps dropout on (0.2 det, 0.1 gaussian) — "
                        "useful for high-DoF envs. Flip off to match "
                        "upstream's plain-MLP setup on tiny tasks like "
                        "point_mass / acrobot where dropout over-smooths.")
    p.add_argument("--no-layernorm", action="store_true",
                   help="Disable LayerNorm between hidden layers of the "
                        "policy. Default keeps layernorm on. Mirrors "
                        "--no-dropout — together they reproduce upstream's "
                        "plain MLP architecture.")
    p.add_argument("--dropout-p", type=float, default=None,
                   help="Override the dropout probability (only relevant "
                        "when --no-dropout is not set). Default is 0.2 for "
                        "DeterministicPolicy, 0.1 for GaussianPolicy.")
    # ---- Policy-trust α dampener (port of upstream `gps_train.py`) ----
    p.add_argument("--adaptive-policy-trust", action="store_true",
                   help="Adaptively scale alpha by a `policy_trust` dual scalar that "
                        "reflects how close the policy's eval cost is to raw MPPI's "
                        "eval cost. Effective alpha = base_alpha * policy_trust; "
                        "policy_trust starts at --policy-trust-min (cold start: prior "
                        "off, MPPI explores freely) and ramps toward --policy-trust-max "
                        "as the policy closes the gap to MPPI. Recomputed each eval "
                        "iter; frozen between evals. Adapted from upstream's "
                        "`compute_policy_trust`/`make_collection_bias` (where it scaled "
                        "lambda_policy_track). With this flag OFF (default), trust ≡ "
                        "--policy-trust-max (1.0 by default → no scaling at all). "
                        "Composes multiplicatively with --kl-target (KL-adaptive α) "
                        "and --alpha-schedule.")
    p.add_argument("--policy-trust-min", type=float, default=None,
                   help="Lower bound on the dual trust scalar (default 0.0 — bad policy "
                        "shuts the prior off completely). Used as the cold-start value "
                        "for adaptive trust.")
    p.add_argument("--policy-trust-max", type=float, default=None,
                   help="Upper bound on the dual trust scalar (default 1.0 — good policy "
                        "passes the prior through unchanged). Trust converges here when "
                        "the policy matches raw MPPI's per-step cost.")
    p.add_argument("--policy-trust-bad-cost-per-step", type=float, default=None,
                   help="Per-step cost threshold treated as 'as bad as no policy' (trust → "
                        "min). Env-specific; pick a margin above raw MPPI's per-step cost "
                        "so the linear interpolation has headroom. Default 0.25 (matches "
                        "upstream's point_mass / acrobot default).")
    p.add_argument("--policy-trust-eval-mppi-eps", type=int, default=None,
                   help="Number of raw-MPPI baseline episodes evaluated on each eval iter "
                        "to feed the trust update. 0 disables the baseline (under "
                        "--adaptive-policy-trust, trust then pins at --policy-trust-min). "
                        "Default 1. Each episode runs --eval-len plan_step calls — keep "
                        "small on expensive envs (adroit), can bump on cheap ones.")
    p.add_argument("--init-ckpt", default=None,
                   help="path to a policy checkpoint (wrapped or raw state_dict) "
                        "to load into gps.policy before the training loop")
    p.add_argument("--exp-name", default="run",
                   help="Human-readable experiment name (used in the run dir name).")
    p.add_argument("--exp-dir", default="checkpoints/gps",
                   help="Parent dir under which a <timestamp>_<env>_<name>/ run dir is created.")
    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    # Load MPPI cfg + resolve final num_conditions BEFORE env construction:
    # the warp env's nworld is pinned to N×K at build time. We can't move
    # this after the gps_cfg overrides without re-instantiating the env,
    # which would re-allocate every GPU buffer.
    mppi_cfg = MPPIConfig.load(args.env)
    policy_cfg = PolicyConfig.for_env(args.env)
    # Policy-architecture overrides from CLI. Each only fires if the user
    # actually set the flag, so the per-env defaults from `for_env(...)`
    # are preserved otherwise.
    if args.disable_tanh:
        policy_cfg.tanh_squash = False
    if args.featurize is not None:
        policy_cfg.featurize = args.featurize
    if args.no_dropout:
        policy_cfg.use_dropout = False
    if args.no_layernorm:
        policy_cfg.use_layernorm = False
    if args.dropout_p is not None:
        policy_cfg.dropout_p = args.dropout_p
    gps_cfg = GPSConfig()

    if args.gps_iters is not None:
        gps_cfg.num_iterations = args.gps_iters
    if args.num_conditions is not None:
        gps_cfg.num_conditions = args.num_conditions

    env_kwargs: dict = {}
    if args.warp:
        # Warp path: BatchedMPPI runs N=num_conditions control problems in
        # parallel, K samples per condition → N*K mjw worlds total. The
        # standalone CPU path (no --warp) doesn't need this — env stays
        # single-condition and MPPI loops over conditions sequentially.
        env_kwargs.update(
            use_warp=True,
            nworld=gps_cfg.num_conditions * mppi_cfg.K,
        )
    env = make_env(args.env, **env_kwargs)
    if args.episode_length is not None:
        gps_cfg.episode_length = args.episode_length
    if args.alpha is not None:
        gps_cfg.policy_augmented_alpha = args.alpha
    if args.alpha_schedule is not None:
        gps_cfg.alpha_schedule = args.alpha_schedule
    if args.alpha_warmup_iters is not None:
        gps_cfg.alpha_warmup_iters = args.alpha_warmup_iters
    if args.alpha_start is not None:
        gps_cfg.alpha_start = args.alpha_start
    if args.kl_target is not None:
        gps_cfg.kl_target = args.kl_target
    if args.kl_alpha_min is not None:
        gps_cfg.kl_alpha_min = args.kl_alpha_min
    if args.kl_alpha_max is not None:
        gps_cfg.kl_alpha_max = args.kl_alpha_max
    if args.kl_step_rate is not None:
        gps_cfg.kl_step_rate = args.kl_step_rate
    if args.kl_sigma_floor_frac is not None:
        gps_cfg.kl_sigma_floor_frac = args.kl_sigma_floor_frac
    if args.policy_prior is not None:
        gps_cfg.policy_prior_type = args.policy_prior
    if args.auto_reset:
        gps_cfg.auto_reset = True
    if args.warm_start_policy:
        gps_cfg.warm_start_policy = True
    if args.eval_every is not None:
        gps_cfg.eval_every = args.eval_every
    if args.eval_len is not None:
        gps_cfg.eval_ep_len = args.eval_len
    if args.distill_buffer_cap is not None:
        gps_cfg.distill_buffer_cap = args.distill_buffer_cap
    if args.ema_decay is not None:
        gps_cfg.ema_decay = args.ema_decay
    if args.ema_hard_sync:
        gps_cfg.ema_hard_sync = True
    if args.reset_optim_per_iter:
        gps_cfg.reset_optim_per_iter = True
    if args.prev_iter_kl_coef is not None:
        gps_cfg.prev_iter_kl_coef = args.prev_iter_kl_coef
    if args.dagger_relabel:
        gps_cfg.dagger_relabel = True
    if args.grad_clip_norm is not None:
        gps_cfg.grad_clip_norm = args.grad_clip_norm
    if args.clip_eps is not None:
        gps_cfg.clip_eps = args.clip_eps
    if args.adaptive_policy_trust:
        gps_cfg.adaptive_policy_trust = True
    if args.policy_trust_min is not None:
        gps_cfg.policy_trust_min = args.policy_trust_min
    if args.policy_trust_max is not None:
        gps_cfg.policy_trust_max = args.policy_trust_max
    if args.policy_trust_bad_cost_per_step is not None:
        gps_cfg.policy_trust_bad_cost_per_step = args.policy_trust_bad_cost_per_step
    if args.policy_trust_eval_mppi_eps is not None:
        gps_cfg.policy_trust_eval_mppi_eps = args.policy_trust_eval_mppi_eps

    device = pick_device(args.device)
    print(f"policy device: {device}")
    policy_class = "Deterministic" if args.deterministic else "Gaussian"
    if gps_cfg.kl_target > 0.0 and not args.deterministic:
        alpha_desc = (
            f"alpha=KL-adaptive(target={gps_cfg.kl_target}, "
            f"range=[{gps_cfg.kl_alpha_min}, {gps_cfg.kl_alpha_max}], "
            f"warmup={gps_cfg.alpha_warmup_iters} iters via "
            f"{gps_cfg.alpha_schedule})"
        )
    elif gps_cfg.alpha_schedule != "constant" and gps_cfg.alpha_warmup_iters > 0:
        alpha_desc = (
            f"alpha={gps_cfg.alpha_start}→{gps_cfg.policy_augmented_alpha} "
            f"({gps_cfg.alpha_schedule} ramp over "
            f"{gps_cfg.alpha_warmup_iters} iters)"
        )
    else:
        alpha_desc = f"alpha={gps_cfg.policy_augmented_alpha}"
    trust_desc = ""
    if gps_cfg.adaptive_policy_trust:
        trust_desc = (
            f", trust=adaptive["
            f"{gps_cfg.policy_trust_min}, {gps_cfg.policy_trust_max}], "
            f"j_bad={gps_cfg.policy_trust_bad_cost_per_step}, "
            f"mppi_eps={gps_cfg.policy_trust_eval_mppi_eps}"
        )
    elif gps_cfg.policy_trust_max != 1.0:
        trust_desc = f", trust=const({gps_cfg.policy_trust_max})"
    print(
        f"GPS: policy={policy_class}, "
        f"{alpha_desc}, "
        f"prior={gps_cfg.policy_prior_type}"
        f"{trust_desc}"
    )

    # ---- Construct trainer (needed early so we can load --init-ckpt and
    #      record the actual policy class in config.json) ----
    # Warp path uses BatchedMPPI under the hood and parallelizes the C-step
    # over conditions; CPU path is the standard MPPIGPS.
    Trainer = WarpMPPIGPS if args.warp else MPPIGPS
    gps = Trainer(
        env, mppi_cfg, policy_cfg, gps_cfg,
        device=device,
        deterministic=args.deterministic,
    )

    if args.init_ckpt is not None:
        init_ckpt = Path(args.init_ckpt)
        if not init_ckpt.exists():
            raise FileNotFoundError(f"--init-ckpt not found: {init_ckpt}")
        blob = load_checkpoint(init_ckpt, map_location=device)
        # Auto-handles GaussianPolicy → DeterministicPolicy by slicing the
        # mu head off the (mu | log_sigma) concatenated head. Other class
        # transitions raise a clear error.
        report = load_state_dict_into(gps.policy, blob)
        print(f"loaded initial policy weights from {init_ckpt}: {report['msg']}")

    # ---- Run dir + config dump ----
    run_dir = make_run_dir(args.exp_dir, args.env, args.exp_name)
    print(f"run dir: {run_dir}")

    start_time = datetime.now().isoformat(timespec="seconds")
    write_config(run_dir, {
        "name": args.exp_name,
        "env": args.env,
        "device": str(device),
        "policy_class": type(gps.policy).__name__,
        "start_time": start_time,
        "end_time": None,
        "git_sha": git_sha(),
        "cli_args": vars(args),
        "configs": {
            "gps": gps_cfg,
            "policy": policy_cfg,
            "mppi": mppi_cfg,
        },
        "best_iter": None,
        "best_cost": None,
    })

    # ---- Training ----
    history = gps.train(run_dir=run_dir)

    # ---- Save final checkpoint (wrapped) + copy to final.pt ----
    # EMA-swap window so final.pt on disk matches the smoothed policy that
    # best.pt was selected from. No-op when ema_decay == 0.
    final_path = run_dir / "final.pt"
    with gps.policy.ema_swapped_in():
        save_checkpoint(
            final_path, gps.policy,
            iteration=gps_cfg.num_iterations - 1,
            mppi_cost=history.iteration_costs[-1] if history.iteration_costs else None,
            eval_cost=history.iteration_eval_costs[-1] if history.iteration_eval_costs else None,
        )
    print(f"\nsaved final policy checkpoint to {final_path}")
    if history.best_iter >= 0:
        print(
            f"best iter: {history.best_iter} "
            f"(cost={history.best_cost:.2f}) → {run_dir / 'best.pt'}"
        )

    # ---- Save learning curves as JSON ----
    # ema_drift / prev_iter_kl exist on some GPSHistory variants but not all;
    # getattr lets a single curves.json schema cover all three trainer classes.
    curves_path = run_dir / "curves.json"
    curves_path.write_text(json.dumps({
        "mppi_costs": history.iteration_costs,
        "eval_costs": history.iteration_eval_costs,
        "distill_loss": history.distill_losses,
        "ema_drift": getattr(history, "iteration_ema_drift", []),
        "prev_iter_kl": getattr(history, "iteration_prev_iter_kl", []),
    }, indent=2))
    print(f"saved learning curves to {curves_path}")

    # ---- Plot learning curves ----
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    iters = np.arange(len(history.iteration_costs))
    ax.plot(iters, history.iteration_costs, label="MPPI C-step", alpha=0.6)
    eval_arr = np.array(history.iteration_eval_costs, dtype=float)
    eval_mask = ~np.isnan(eval_arr)
    if eval_mask.any():
        ax.plot(iters[eval_mask], eval_arr[eval_mask],
                marker="o", linewidth=2, label="policy eval (selects best)")
    ax.set_xlabel("GPS iteration")
    ax.set_ylabel("mean episode cost")
    ax.set_title("Cost")
    ax.legend()
    plt.tight_layout()
    plot_path = run_dir / "curves.png"
    plt.savefig(plot_path, dpi=120)
    print(f"saved learning curve plot to {plot_path}")

    # ---- Evaluation: GPS policy vs MPPI baseline ----
    # EMA-swap for the headline eval too — matches what best.pt / final.pt
    # contain, and therefore the numbers the user will see when they reload
    # the checkpoint later. No-op when ema_decay == 0.
    gps.policy.eval()
    print("\n--- Evaluating GPS policy ---")
    with gps.policy.ema_swapped_in():
        gps_stats = evaluate_policy(
            gps.policy, env,
            n_episodes=args.n_eval,
            episode_len=args.eval_len,
            seed=args.seed,
            render=True,
        )

    print("--- Evaluating MPPI baseline ---")
    mppi = MPPI(env, mppi_cfg)
    mppi_stats = evaluate_mppi(
        env, mppi,
        n_episodes=args.n_eval,
        episode_len=args.eval_len,
        seed=args.seed,
    )

    print()
    print(f"GPS policy:    {gps_stats['mean_cost']:8.2f} +/- {gps_stats['std_cost']:.2f}")
    print(f"MPPI baseline: {mppi_stats['mean_cost']:8.2f} +/- {mppi_stats['std_cost']:.2f}")
    print(f"gap (GPS - MPPI): {gps_stats['mean_cost'] - mppi_stats['mean_cost']:+8.2f}")
    print()
    print("per-episode (GPS vs MPPI):")
    for ep, (g, m) in enumerate(zip(gps_stats["per_ep"], mppi_stats["per_ep"])):
        print(f"  ep {ep:2d}  GPS={g:8.2f}  MPPI={m:8.2f}  gap={g - m:+8.2f}")

    # ---- Save video of trained policy ----
    video_path = None
    if gps_stats["frames"]:
        video_path = run_dir / f"{args.env}.mp4"
        mediapy.write_video(str(video_path), gps_stats["frames"], fps=30)
        print(f"\nsaved rollout video to {video_path}")

    # ---- Finalize config.json ----
    update_config(run_dir, {
        "end_time": datetime.now().isoformat(timespec="seconds"),
        "best_iter": history.best_iter if history.best_iter >= 0 else None,
        "best_cost": history.best_cost if history.best_iter >= 0 else None,
        "eval": {
            "gps_mean_cost": gps_stats["mean_cost"],
            "gps_std_cost": gps_stats["std_cost"],
            "mppi_mean_cost": mppi_stats["mean_cost"],
            "mppi_std_cost": mppi_stats["std_cost"],
        },
    })

    print(f"\nrun dir: {run_dir}")
    env.close()


if __name__ == "__main__":
    main()
