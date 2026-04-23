"""DAgger with MPPI as the expert — distills MPPI into a GaussianPolicy.

Example:
    python -m scripts.run_dagger --env acrobot --dagger-iters 10 \
        --rollouts-per-iter 20 --episode-len 200 --device auto

    # Hopper (terminating env — use --auto-reset so early falls don't
    # waste the per-rollout step budget):
    python -m scripts.run_dagger --env hopper --deterministic --auto-reset \
        --warmup-rollouts 30 --warmup-epochs 50 \
        --dagger-iters 30 --rollouts-per-iter 10 --episode-len 500 \
        --beta-schedule linear --distill-epochs 6 --buffer-cap 50000 --device cuda

MPPI runs on CPU (MuJoCo). Policy training uses the device selected by
`--device` (auto → cuda → mps → cpu).
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from src.envs import make_env
from src.mppi.mppi import MPPI
from src.policy.gaussian_policy import GaussianPolicy
from src.policy.deterministic_policy import DeterministicPolicy
from src.utils.config import DAggerConfig, MPPIConfig, PolicyConfig
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
from src.gps.dagger import DAggerTrainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="acrobot")
    p.add_argument("--dagger-iters", type=int, default=20)
    p.add_argument("--rollouts-per-iter", type=int, default=20)
    p.add_argument("--episode-len", type=int, default=500)
    p.add_argument("--beta-schedule", default="linear", choices=["linear", "constant_zero"])
    p.add_argument("--distill-epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--buffer-cap", type=int, default=400_000)
    p.add_argument("--n-eval-eps", type=int, default=10)
    p.add_argument("--eval-ep-len", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    p.add_argument("--deterministic", action="store_true", default=True,
                   help="use DeterministicPolicy (direct action regression) instead of GaussianPolicy")
    p.add_argument("--init-ckpt", default=None,
                   help="path to a policy checkpoint to load before warmup / DAgger training")
    p.add_argument("--seed-from", default=None,
                   help="path to existing BC h5 (e.g. data/acrobot_bc.h5) to warm-start the buffer")
    p.add_argument("--warmup-rollouts", type=int, default=30,
                   help="Pre-DAgger: collect this many pure-MPPI rollouts and BC-train the policy on them")
    p.add_argument("--warmup-epochs", type=int, default=50,
                   help="Epochs of BC pre-training on the warmup rollouts (ignored if --warmup-rollouts=0)")
    p.add_argument("--warmup-cache", default=None,
                   help="optional h5 path for warmup rollouts. If the file exists it's "
                        "loaded (skipping collection); otherwise rollouts are collected "
                        "and saved here for reuse. Disabled by default. Compatible with "
                        "collect_bc_demos output — point this at an existing "
                        "data/<env>_bc.h5 to skip warmup collection entirely.")
    p.add_argument("--auto-reset", action="store_true",
                   help="On episode termination during rollout/relabel, auto-reset and keep "
                        "collecting until episode_len steps are taken for that slot. Recommended "
                        "for terminating envs like hopper where policy-driven episodes fall early.")
    p.add_argument("--exp-name", default="run",
                   help="Human-readable experiment name (used in the run dir name).")
    p.add_argument("--exp-dir", default="checkpoints/dagger",
                   help="Parent dir under which a <timestamp>_<env>_<name>/ run dir is created.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cfg = DAggerConfig(
        dagger_iters=args.dagger_iters,
        rollouts_per_iter=args.rollouts_per_iter,
        episode_len=args.episode_len,
        beta_schedule=args.beta_schedule,
        buffer_cap=args.buffer_cap,
        distill_epochs=args.distill_epochs,
        batch_size=args.batch_size,
        n_eval_eps=args.n_eval_eps,
        eval_ep_len=args.eval_ep_len,
        seed=args.seed,
        auto_reset=args.auto_reset,
    )

    device = pick_device(args.device)
    print(f"policy device: {device}")

    seed_everything(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    env = make_env(args.env)
    mppi_cfg = MPPIConfig.load(args.env)
    mppi = MPPI(env, cfg=mppi_cfg)

    obs_dim = env.obs_dim
    act_dim = env.action_dim
    policy_cfg = PolicyConfig()
    bounds = env.action_bounds
    if args.deterministic:
        policy = DeterministicPolicy(obs_dim, act_dim, policy_cfg,
                                     device=device, action_bounds=bounds)
        print("policy: DeterministicPolicy (MSE regression)")
    else:
        policy = GaussianPolicy(obs_dim, act_dim, policy_cfg,
                                device=device, action_bounds=bounds)
        print("policy: GaussianPolicy (mu/sigma head, MSE on mu)")

    if args.init_ckpt is not None:
        init_ckpt = Path(args.init_ckpt)
        if not init_ckpt.exists():
            raise FileNotFoundError(f"--init-ckpt not found: {init_ckpt}")
        blob = load_checkpoint(init_ckpt, map_location=device)
        try:
            policy.load_state_dict(blob["state_dict"])
        except RuntimeError as exc:
            raise RuntimeError(
                f"failed to load --init-ckpt {init_ckpt}; make sure the checkpoint "
                f"matches --deterministic={args.deterministic}"
            ) from exc
        print(f"loaded initial policy weights from {init_ckpt}"
              + (f"  (policy_class={blob['policy_class']})" if 'policy_class' in blob else ""))

    trainer = DAggerTrainer(env, mppi, policy, cfg, rng=rng)
    if args.seed_from is not None:
        seed_path = Path(args.seed_from)
        if seed_path.exists():
            print(f"seeding buffer from {seed_path}")
            trainer.seed_from_h5(seed_path)
            print(f"  buffer size after seeding: {trainer.buffer_size():,}")
        else:
            print(f"warning: --seed-from {seed_path} does not exist; skipping")

    run_dir = make_run_dir(args.exp_dir, args.env, args.exp_name)
    print(f"run dir: {run_dir}")

    start_time = datetime.now().isoformat(timespec="seconds")
    write_config(run_dir, {
        "name": args.exp_name,
        "env": args.env,
        "device": str(device),
        "policy_class": type(policy).__name__,
        "start_time": start_time,
        "end_time": None,
        "git_sha": git_sha(),
        "cli_args": vars(args),
        "configs": {
            "dagger": cfg,
            "policy": policy_cfg,
            "mppi": mppi_cfg,
        },
        "best_iter": None,
        "best_cost": None,
    })

    if args.init_ckpt is not None and args.warmup_rollouts > 0:
        print("\nskipping warmup because --init-ckpt was provided")
    elif args.warmup_rollouts > 0:
        warmup_cache = Path(args.warmup_cache) if args.warmup_cache else None
        if warmup_cache is not None and warmup_cache.exists():
            print(f"\nwarmup: reusing cached rollouts at {warmup_cache} + "
                  f"{args.warmup_epochs} epochs of BC pre-training...")
        else:
            print(f"\nwarmup: collecting {args.warmup_rollouts} pure-MPPI rollouts + "
                  f"{args.warmup_epochs} epochs of BC pre-training"
                  + (f"  (will cache to {warmup_cache})" if warmup_cache else "")
                  + "...")
        losses = trainer.warmup(
            args.warmup_rollouts, args.warmup_epochs,
            cache_path=warmup_cache,
        )
        if losses:
            print(f"  warmup train_mse: {losses[0]:.5f} → {losses[-1]:.5f}  "
                  f"(buf={trainer.buffer_size():,})")

    # # --- MPPI baseline (once, shared across iterations) ---
    # print("\nevaluating MPPI baseline...")
    # mppi_eval = MPPI(env, cfg=mppi_cfg)
    # mppi_stats = evaluate_mppi(env, mppi_eval,
    #                             n_episodes=cfg.n_eval_eps,
    #                             episode_len=cfg.eval_ep_len,
    #                             seed=cfg.seed)
    # print(f"MPPI baseline: {mppi_stats['mean_cost']:.2f} ± {mppi_stats['std_cost']:.2f}")

    log_lines: list[str] = [
        f"# DAgger on {args.env}, device={device}",
        # f"# MPPI baseline: mean={mppi_stats['mean_cost']:.4f} std={mppi_stats['std_cost']:.4f}",
        "iter,beta,new_samples,buffer_size,train_mse,val_mse,policy_mean_cost,policy_std_cost",
    ]

    best_iter: int | None = None
    best_cost = float("inf")
    last_ckpt_path: Path | None = None

    for k in range(cfg.dagger_iters):
        print(f"\n=== DAgger iter {k}/{cfg.dagger_iters - 1} ===")
        info = trainer.step(k)
        print(f"  beta={info['beta']:.2f}  new={info['new_samples']:,}  "
              f"buf={info['buffer_size']:,}  train_mse={info['train_mse']:.5f}  "
              f"val_mse={info['val_mse']:.5f}")

        policy.eval()
        eval_stats = evaluate_policy(policy, env,
                                     n_episodes=cfg.n_eval_eps,
                                     episode_len=cfg.eval_ep_len,
                                     seed=cfg.seed)
        policy.train()
        print(f"  policy cost: {eval_stats['mean_cost']:.2f} ± {eval_stats['std_cost']:.2f}  ")
            #   f"(gap vs MPPI: {eval_stats['mean_cost'] - mppi_stats['mean_cost']:+.2f})")

        log_lines.append(
            f"{k},{info['beta']:.4f},{info['new_samples']},{info['buffer_size']},"
            f"{info['train_mse']:.6f},{info['val_mse']:.6f},"
            f"{eval_stats['mean_cost']:.4f},{eval_stats['std_cost']:.4f}"
        )

        ckpt_path = run_dir / f"iter_{k:02d}.pt"
        save_checkpoint(
            ckpt_path, policy,
            round=k,
            train_mse=info["train_mse"],
            val_mse=info["val_mse"],
            eval_mean_cost=eval_stats["mean_cost"],
            eval_std_cost=eval_stats["std_cost"],
        )
        last_ckpt_path = ckpt_path

        if eval_stats["mean_cost"] < best_cost:
            best_cost = eval_stats["mean_cost"]
            best_iter = k
            copy_as(ckpt_path, run_dir / "best.pt")

        (run_dir / "dagger_log.csv").write_text("\n".join(log_lines))

    if last_ckpt_path is not None:
        copy_as(last_ckpt_path, run_dir / "final.pt")

    update_config(run_dir, {
        "end_time": datetime.now().isoformat(timespec="seconds"),
        "best_iter": best_iter,
        "best_cost": best_cost if best_iter is not None else None,
        # "mppi_baseline": {
        #     "mean_cost": mppi_stats["mean_cost"],
        #     "std_cost": mppi_stats["std_cost"],
        # },
    })

    print(f"\nrun dir: {run_dir}")
    print(f"best iter: {best_iter}  best cost: {best_cost:.2f}")
    env.close()


if __name__ == "__main__":
    main()
