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
                   help="Per-iter α schedule. 'constant' uses --alpha verbatim; "
                        "other shapes ramp from --alpha-start to --alpha over "
                        "--alpha-warmup-iters iters and hold.")
    p.add_argument("--alpha-warmup-iters", type=int, default=None,
                   help="Ramp duration; 0 disables. Ignored when --alpha-schedule "
                        "is 'constant'.")
    p.add_argument("--alpha-start", type=float, default=0,
                   help="Starting α for the ramp (default 0.0).")
    p.add_argument("--kl-target", type=float, default=None,
                   help="KL-adaptive α (Gaussian only). When > 0, α is auto-"
                        "adjusted each iter to drive E_state[KL(N(μ_p,σ_p²) ‖ π_θ)] "
                        "toward this target. Schedule still drives the warmup window. "
                        "KL is summed over action dims — scale ~act_dim × 0.05 "
                        "(~0.1 acrobot, ~1.5 adroit_relocate). 0 disables.")
    p.add_argument("--kl-alpha-min", type=float, default=None,
                   help="Lower bound on KL-adaptive α (default 0.001).")
    p.add_argument("--kl-alpha-max", type=float, default=None,
                   help="Upper bound on KL-adaptive α (default 0.5). α ≫ 0.1 "
                        "typically crushes MPPI exploration regardless of "
                        "the constraint.")
    p.add_argument("--kl-step-rate", type=float, default=None,
                   help="Multiplicative dual-α update rate (default 1.5).")
    p.add_argument("--kl-sigma-floor-frac", type=float, default=None,
                   help="Floor on σ_p in the KL estimator as a fraction of "
                        "MPPI's proposal std (default 0.5). Prevents "
                        "log(σ_θ/σ_p) blow-up when MPPI softmin concentrates. "
                        "0 disables.")
    p.add_argument("--policy-prior", default=None,
                   choices=["nll", "mean_distance"],
                   help="Policy prior shape. Unset = auto (nll for Gaussian, "
                        "mean_distance for --deterministic). 'nll' is "
                        "Gaussian-only; 'mean_distance' works for both.")
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
    p.add_argument("--reset-optim-per-iter", action="store_true", default=True,
                   help="Recreate the policy's Adam optimizer at end of each GPS iteration, "
                        "wiping m/v moments. Recommended with --prev-iter-kl-coef "
                        "(accumulated momentum can blow through the trust region on the "
                        "first few steps).")
    p.add_argument("--prev-iter-kl-coef", type=float, default=0,
                   help="Trust-region KL penalty to the previous iteration's "
                        "policy (Gaussian only). Adds `coef * E_obs[KL(π_θ || "
                        "π_prev)]` to the S-step loss. 0 disables.")
    p.add_argument("--dagger-relabel", action="store_true",
                   help="DAgger-style decoupled relabel: per C-step timestep, "
                        "MPPI runs WITH the prior (executor) and WITHOUT it "
                        "(dry_run, the label). No-op when alpha == 0. Under "
                        "open_loop_steps > 1 the label call forces a full "
                        "rollout every step.")
    p.add_argument("--deterministic", action="store_true",
                   help="Use a DeterministicPolicy student (MSE distill, "
                        "mean_distance prior).")
    p.add_argument("--grad-clip-norm", type=float, default=None,
                   help="L2 grad-norm clip in the deterministic policy's "
                        "MSE step. 0 = disabled. Default 1.0. Ignored "
                        "without --deterministic.")
    p.add_argument("--clip-eps", type=float, default=0.3,
                   help="Action-space trust region for the deterministic "
                        "S-step. Per batch, the MPPI label is clamped to "
                        "[π_old.action(o) ± clip_eps] before MSE. 0 disables. "
                        "Prefer --grad-clip-norm unless you specifically want "
                        "action-space TR semantics.")
    # ---- Policy architecture knobs (PolicyConfig fields) ----
    p.add_argument("--disable-tanh", action="store_true",
                   help="Disable tanh squash on the policy head. Use for "
                        "pre-tanh checkpoints (unbounded head + act_np clamp).")
    p.add_argument("--featurize", default=None,
                   choices=["running_norm", "hand_crafted"],
                   help="Input pre-processor. 'running_norm' (default) is the "
                        "learned RunningNormalizer; 'hand_crafted' uses the "
                        "env-aware transform in src.policy.featurize_obs "
                        "(4-D acrobot, 6-D point_mass).")
    p.add_argument("--no-dropout", action="store_true",
                   help="Disable Dropout between hidden layers (default: on).")
    p.add_argument("--no-layernorm", action="store_true",
                   help="Disable LayerNorm between hidden layers (default: on).")
    p.add_argument("--dropout-p", type=float, default=None,
                   help="Override dropout probability. Default 0.2 det, 0.1 gaussian.")
    # ---- Policy-trust α dampener (port of upstream `gps_train.py`) ----
    p.add_argument("--adaptive-policy-trust", action="store_true",
                   help="Scale α by a trust scalar that ramps with the "
                        "policy↔raw-MPPI cost gap. Off (default) → trust ≡ "
                        "--policy-trust-max (1.0 → no scaling). Composes "
                        "multiplicatively with --kl-target and --alpha-schedule.")
    p.add_argument("--policy-trust-min", type=float, default=None,
                   help="Trust lower bound (default 0.0 — bad policy → prior off).")
    p.add_argument("--policy-trust-max", type=float, default=None,
                   help="Trust upper bound (default 1.0 — good policy → prior "
                        "passes through).")
    p.add_argument("--policy-trust-bad-cost-per-step", type=float, default=None,
                   help="Per-step cost above which the policy is treated as "
                        "'as bad as no policy'. Env-specific; default 0.25.")
    p.add_argument("--policy-trust-eval-mppi-eps", type=int, default=None,
                   help="Number of raw-MPPI baseline episodes per eval iter "
                        "(feeds the trust update). 0 disables. Default 1.")
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

    # Load MPPI cfg + resolve `num_conditions` BEFORE env construction —
    # the warp env's `nworld = N × K` is pinned at build time.
    mppi_cfg = MPPIConfig.load(args.env)
    policy_cfg = PolicyConfig.for_env(args.env)
    # Each CLI override fires only when the user set the flag; per-env
    # defaults from `for_env(...)` are otherwise preserved.
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
        # Warp path: N conditions × K samples = N*K mjw worlds.
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
    final_path = run_dir / "final.pt"
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
    # prev_iter_kl exists on some GPSHistory variants but not all;
    # getattr keeps the schema uniform across trainer classes.
    curves_path = run_dir / "curves.json"
    curves_path.write_text(json.dumps({
        "mppi_costs": history.iteration_costs,
        "eval_costs": history.iteration_eval_costs,
        "distill_loss": history.distill_losses,
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
    gps.policy.eval()
    print("\n--- Evaluating GPS policy ---")
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
        # Use the env's wall-clock dt so the saved video plays real-time —
        # matches eval_checkpoint and the per-env runners.
        dt = env.model.opt.timestep * env._frame_skip
        fps = int(round(1.0 / dt))
        mediapy.write_video(str(video_path), gps_stats["frames"], fps=fps)
        print(f"\nsaved rollout video to {video_path} (fps={fps})")

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
