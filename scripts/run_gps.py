"""Train a reactive policy via MPPI-GPS and evaluate against baselines.

This is the main entry point for GPS training.  It:
  1. Creates an environment and loads tuned MPPI hyperparameters.
  2. Instantiates the MPPIGPS trainer and runs the full training loop.
  3. Saves the trained policy checkpoint and learning curves (JSON + PNG).
  4. Evaluates the GPS policy vs the MPPI baseline on identical initial
     conditions, printing per-episode cost comparisons.
  5. Renders a video of the trained policy's first evaluation episode.

Usage examples:
    # Train on acrobot with defaults (50 iterations, 5 conditions)
    python scripts/run_gps.py --env acrobot

    # Train on hopper with custom settings
    python scripts/run_gps.py --env hopper --gps-iters 30 --alpha 0.05

    # Quick test run
    python scripts/run_gps.py --env acrobot --gps-iters 3 --episode-length 100
"""

import argparse
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import mediapy
from pathlib import Path

from src.gps.mppi_gps import MPPIGPS
from src.utils.config import MPPIConfig, PolicyConfig, GPSConfig
from src.utils.evaluation import evaluate_policy, evaluate_mppi
from src.mppi.mppi import MPPI


# Maps env name → (module path, class name) for dynamic import.
# Add new environments here to make them available via --env.
ENV_REGISTRY = {
    "acrobot": ("src.envs.acrobot", "Acrobot"),
    "half_cheetah": ("src.envs.half_cheetah", "HalfCheetah"),
    "point_mass": ("src.envs.point_mass", "PointMass"),
    "hopper": ("src.envs.hopper", "Hopper"),
}


def make_env(name: str):
    """Dynamically import and instantiate an environment by name."""
    mod_path, cls_name = ENV_REGISTRY[name]
    import importlib
    mod = importlib.import_module(mod_path)
    return getattr(mod, cls_name)()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="acrobot", choices=list(ENV_REGISTRY.keys()))
    p.add_argument("--gps-iters", type=int, default=None,
                   help="Number of GPS iterations (default: from GPSConfig)")
    p.add_argument("--num-conditions", type=int, default=None,
                   help="Number of initial conditions to train across")
    p.add_argument("--episode-length", type=int, default=None,
                   help="Steps per episode during GPS training")
    p.add_argument("--alpha", type=float, default=None,
                   help="Policy-augmented cost weight (0 = no augmentation)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ckpt-dir", type=str, default="checkpoints")
    p.add_argument("--n-eval", type=int, default=10,
                   help="Number of evaluation episodes")
    p.add_argument("--eval-len", type=int, default=500,
                   help="Max steps per evaluation episode")
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = make_env(args.env)

    # Load tuned MPPI hyperparameters from configs/<env>_best.json
    mppi_cfg = MPPIConfig.load(args.env)
    policy_cfg = PolicyConfig()
    gps_cfg = GPSConfig()

    # Apply any command-line overrides to GPS config
    if args.gps_iters is not None:
        gps_cfg.num_iterations = args.gps_iters
    if args.num_conditions is not None:
        gps_cfg.num_conditions = args.num_conditions
    if args.episode_length is not None:
        gps_cfg.episode_length = args.episode_length
    if args.alpha is not None:
        gps_cfg.policy_augmented_alpha = args.alpha

    # ---- Training ----
    gps = MPPIGPS(env, mppi_cfg, policy_cfg, gps_cfg)
    history = gps.train()

    # ---- Save checkpoint ----
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"gps_{args.env}.pt"
    torch.save(gps.policy.state_dict(), ckpt_path)
    print(f"\nsaved policy checkpoint to {ckpt_path}")

    # ---- Save learning curves as JSON ----
    curves_path = ckpt_dir / f"gps_{args.env}_curves.json"
    curves_path.write_text(json.dumps({
        "costs": history.iteration_costs,
        "kl": history.iteration_kl,
        "nu": history.iteration_nu,
        "distill_loss": history.distill_losses,
    }, indent=2))
    print(f"saved learning curves to {curves_path}")

    # ---- Plot learning curves ----
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(history.iteration_costs)
    axes[0].set_xlabel("GPS iteration")
    axes[0].set_ylabel("mean episode cost")
    axes[0].set_title("Cost")

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
    plot_path = ckpt_dir / f"gps_{args.env}_curves.png"
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

    # Print summary comparison
    print()
    print(f"GPS policy:    {gps_stats['mean_cost']:8.2f} +/- {gps_stats['std_cost']:.2f}")
    print(f"MPPI baseline: {mppi_stats['mean_cost']:8.2f} +/- {mppi_stats['std_cost']:.2f}")
    print(f"gap (GPS - MPPI): {gps_stats['mean_cost'] - mppi_stats['mean_cost']:+8.2f}")
    print()
    print("per-episode (GPS vs MPPI):")
    for ep, (g, m) in enumerate(zip(gps_stats["per_ep"], mppi_stats["per_ep"])):
        print(f"  ep {ep:2d}  GPS={g:8.2f}  MPPI={m:8.2f}  gap={g - m:+8.2f}")

    # ---- Save video of trained policy ----
    if gps_stats["frames"]:
        video_path = ckpt_dir / f"gps_{args.env}.mp4"
        mediapy.write_video(str(video_path), gps_stats["frames"], fps=30)
        print(f"\nsaved rollout video to {video_path}")

    env.close()


if __name__ == "__main__":
    main()
