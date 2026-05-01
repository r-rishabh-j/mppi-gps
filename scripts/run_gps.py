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
    python -m scripts.run_gps --env acrobot --disable-kl --distill-loss nll \
        --alpha 0.1 --gps-iters 20 --exp-name prior_only
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
from src.gps.mppi_gps_clip import MPPIGPS
from src.mppi.mppi import MPPI
from src.utils.config import GPSConfig, MPPIConfig, PolicyConfig
from src.utils.device import pick_device
from src.utils.evaluation import evaluate_mppi, evaluate_policy
from src.utils.seeding import seed_everything
from src.utils.experiment import (
    copy_as,
    git_sha,
    load_checkpoint,
    make_run_dir,
    save_checkpoint,
    update_config,
    write_config,
)


_ENVS = ["acrobot", "half_cheetah", "point_mass", "hopper"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="acrobot", choices=_ENVS)
    p.add_argument("--gps-iters", type=int, default=None,
                   help="Number of GPS iterations (default: from GPSConfig)")
    p.add_argument("--num-conditions", type=int, default=None,
                   help="Number of initial conditions to train across")
    p.add_argument("--episode-length", type=int, default=None,
                   help="Steps per episode during GPS training")
    p.add_argument("--alpha", type=float, default=None,
                   help="Policy-augmented cost weight (0 = no augmentation)")
    p.add_argument("--disable-kl", action="store_true", default=True,
                   help="Drop KL/BADMM: run policy-prior-only GPS (BC on MPPI data)")
    p.add_argument("--distill-loss", default=None, choices=["nll", "mse"],
                   help="Distillation loss (default from GPSConfig: nll)")
    p.add_argument("--nu", type=float, default=None,
                   help="Constant nu for the policy prior when --disable-kl is set")
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
                        "(Gaussian distill loss only). Adds "
                        "`coef * E_obs[KL(π_θ || π_prev)]` to the S-step loss. 0/unset "
                        "disables. Typical small values (e.g. 0.01-0.1). Ignored for "
                        "--distill-loss mse.")
    p.add_argument("--dagger-relabel", action="store_true",
                   help="DAgger-style decoupled relabel: per C-step timestep, run MPPI once "
                        "WITH the policy prior (executor, steers the env + feeds KL) and a "
                        "second time WITHOUT the prior (side-effect-free dry_run, its action "
                        "is the training label). Removes the self-reinforcing loop where the "
                        "executed MPPI action is already tilted toward the current policy. "
                        "No-op when α·ν == 0. Note: under open_loop_steps > 1 the label call "
                        "forces a full rollout every step (cached chunk actions are prior-"
                        "biased, so can't serve as plain labels) — wallclock per step becomes "
                        "~1 rollout instead of 1/N.")
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

    env = make_env(args.env)

    mppi_cfg = MPPIConfig.load(args.env)
    policy_cfg = PolicyConfig()
    gps_cfg = GPSConfig()

    if args.gps_iters is not None:
        gps_cfg.num_iterations = args.gps_iters
    if args.num_conditions is not None:
        gps_cfg.num_conditions = args.num_conditions
    if args.episode_length is not None:
        gps_cfg.episode_length = args.episode_length
    if args.alpha is not None:
        gps_cfg.policy_augmented_alpha = args.alpha
    if args.disable_kl:
        gps_cfg.disable_kl_constraint = True
    if args.distill_loss is not None:
        gps_cfg.distill_loss = args.distill_loss
    if args.nu is not None:
        gps_cfg.badmm_init_nu = args.nu
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

    device = pick_device(args.device)
    print(f"policy device: {device}")
    if gps_cfg.disable_kl_constraint:
        print(f"KL constraint DISABLED — policy-prior-only GPS "
              f"(alpha={gps_cfg.policy_augmented_alpha}, nu={gps_cfg.badmm_init_nu}, "
              f"distill_loss={gps_cfg.distill_loss})")

    # ---- Construct trainer (needed early so we can load --init-ckpt and
    #      record the actual policy class in config.json) ----
    gps = MPPIGPS(env, mppi_cfg, policy_cfg, gps_cfg, device=device)

    if args.init_ckpt is not None:
        init_ckpt = Path(args.init_ckpt)
        if not init_ckpt.exists():
            raise FileNotFoundError(f"--init-ckpt not found: {init_ckpt}")
        blob = load_checkpoint(init_ckpt, map_location=device)
        try:
            gps.policy.load_state_dict(blob["state_dict"])
        except RuntimeError as exc:
            raise RuntimeError(
                f"failed to load --init-ckpt {init_ckpt} into "
                f"{type(gps.policy).__name__} — shapes / policy class mismatch?"
            ) from exc
        print(f"loaded initial policy weights from {init_ckpt}"
              + (f"  (policy_class={blob['policy_class']})" if 'policy_class' in blob else ""))

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
    curves_path = run_dir / "curves.json"
    curves_path.write_text(json.dumps({
        "mppi_costs": history.iteration_costs,
        "eval_costs": history.iteration_eval_costs,
        "kl": history.iteration_kl,
        "nu": history.iteration_nu,
        "distill_loss": history.distill_losses,
        "ema_drift": history.iteration_ema_drift,
        "prev_iter_kl": history.iteration_prev_iter_kl,
    }, indent=2))
    print(f"saved learning curves to {curves_path}")

    # ---- Plot learning curves ----
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    iters = np.arange(len(history.iteration_costs))
    axes[0].plot(iters, history.iteration_costs, label="MPPI C-step", alpha=0.6)
    eval_arr = np.array(history.iteration_eval_costs, dtype=float)
    eval_mask = ~np.isnan(eval_arr)
    if eval_mask.any():
        axes[0].plot(iters[eval_mask], eval_arr[eval_mask],
                     marker="o", linewidth=2, label="policy eval (selects best)")
    axes[0].set_xlabel("GPS iteration")
    axes[0].set_ylabel("mean episode cost")
    axes[0].set_title("Cost")
    axes[0].legend()

    axes[1].plot(history.iteration_kl)
    axes[1].axhline(gps_cfg.kl_target, color="r", linestyle="--", label="target")
    axes[1].set_xlabel("GPS iteration")
    axes[1].set_ylabel("KL divergence")
    axes[1].set_title("KL")
    axes[1].legend()

    axes[2].plot(history.iteration_nu)
    axes[2].set_xlabel("GPS iteration")
    axes[2].set_ylabel("nu (BADMM dual)")
    axes[2].set_title("Dual variable")

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
