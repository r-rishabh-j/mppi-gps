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
import csv
import os
from contextlib import nullcontext as _nullctx
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
    load_state_dict_into,
    make_run_dir,
    save_checkpoint,
    update_config,
    write_config,
)
from src.gps.dagger import DAggerTrainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="hopper")
    p.add_argument("--warp", action="store_true",
                   help="Use mujoco_warp GPU batch rollout for MPPI's "
                        "relabel step. Runs --rollouts-per-iter parallel "
                        "envs simultaneously; one BatchedMPPI call per "
                        "timestep instead of N sequential calls. Only envs "
                        "with a Warp variant in the registry "
                        "(adroit_relocate, hopper). Requires "
                        "`uv pip install warp-lang mujoco-warp` and an "
                        "NVIDIA GPU with CUDA. nworld is pinned to "
                        "rollouts_per_iter * MPPIConfig.K at env "
                        "construction.")
    p.add_argument("--dagger-iters", type=int, default=20)
    p.add_argument("--rollouts-per-iter", type=int, default=10)
    p.add_argument("--episode-len", type=int, default=600)
    p.add_argument("--beta-schedule", default="constant_zero", choices=["linear", "constant_zero"])
    p.add_argument("--distill-epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--buffer-cap", type=int, default=400_000)
    p.add_argument("--n-eval-eps", type=int, default=10)
    p.add_argument("--eval-ep-len", type=int, default=800)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    p.add_argument("--deterministic", action="store_true", default=False,
                   help="use DeterministicPolicy (direct action regression) instead of GaussianPolicy")
    p.add_argument("--init-ckpt", default=None,
                   help="path to a policy checkpoint to load before warmup / DAgger training")
    p.add_argument("--seed-from", default=None,
                   help="path to existing BC h5 (e.g. data/acrobot_bc.h5) to warm-start the buffer")
    p.add_argument("--warmup-rollouts", type=int, default=30,
                   help="Pre-DAgger: collect this many pure-MPPI rollouts and BC-train the policy on them")
    p.add_argument("--warmup-epochs", type=int, default=100,
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
    p.add_argument("--ema-decay", type=float, default=None,
                   help="Exponential moving average decay over the policy's trainable "
                        "parameters (e.g. 0.999). 0 or unset disables EMA. Eval and best.pt "
                        "selection use the EMA snapshot when enabled, so the checkpoint on "
                        "disk matches the reported eval cost.")
    p.add_argument("--ema-hard-sync", action="store_true", default=True,
                   help="At end of each DAgger round, copy EMA shadow into the live policy "
                        "(θ ← EMA) so subsequent rollouts and finetunes start from the "
                        "smoothed weights. No effect unless --ema-decay > 0. Pair with "
                        "--reset-optim-per-iter for Adam consistency.")
    p.add_argument("--reset-optim-per-iter", action="store_true", default=True,
                   help="Recreate the policy's Adam optimizer at end of each DAgger round, "
                        "wiping m/v moments. Recommended with --ema-hard-sync (stale Adam "
                        "state after a hard-sync) and whenever the buffer shifts distribution "
                        "enough that stale momentum becomes a liability.")
    p.add_argument("--grad-clip-norm", type=float, default=None,
                   help="L2 gradient-norm clip applied inside the deterministic "
                        "policy's mse_step during DAgger warmup + finetune. Bounds "
                        "per-update parameter movement; loss-agnostic, no biased "
                        "estimator. 0 = disabled. Default 1.0 (see DAggerConfig). "
                        "Only takes effect with --deterministic; Gaussian dagger "
                        "ignores it.")
    p.add_argument("--clip-ratio", type=float, default=None,
                   help="PPO-style probability ratio clip for the Gaussian dagger "
                        "finetune (mirrors mppi_gps_clip's Gaussian branch). "
                        "Snapshots the policy at the start of each round; the "
                        "per-batch surrogate caps how much each (obs, expert action) "
                        "pair can boost its log-likelihood per step (saturates at "
                        "ratio=1+eps). 0/unset = disabled (default = plain MSE-on-"
                        "mean). Typical 0.1–0.3; standard PPO uses 0.2. NOT applied "
                        "in --warmup. Only takes effect WITHOUT --deterministic.")
    p.add_argument("--loss", default=None, choices=["mse", "nll"],
                   dest="loss_type",
                   help="Distillation loss for the Gaussian DAgger finetune. "
                        "'mse' (default) — MSE on the policy mean; log_sigma "
                        "is not supervised. 'nll' — full diagonal-Gaussian "
                        "negative log-likelihood; both mu AND log_sigma are "
                        "trained, so σ shrinks/widens to reflect expert action "
                        "variance. Recommended when the policy's σ matters "
                        "downstream (GPS warm-start, KL-adaptive α). Ignored "
                        "with --deterministic (always MSE) and silently "
                        "overridden when --clip-ratio > 0 (PPO clip is its "
                        "own NLL variant).")
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
    if args.ema_decay is not None:
        cfg.ema_decay = args.ema_decay
    if args.ema_hard_sync:
        cfg.ema_hard_sync = True
    if args.reset_optim_per_iter:
        cfg.reset_optim_per_iter = True
    if args.grad_clip_norm is not None:
        cfg.grad_clip_norm = args.grad_clip_norm
    if args.clip_ratio is not None:
        cfg.clip_ratio = args.clip_ratio
    if args.loss_type is not None:
        cfg.loss_type = args.loss_type

    device = pick_device(args.device)
    print(f"policy device: {device}")

    seed_everything(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    # Construct env + MPPI controller. Warp path uses BatchedMPPI with
    # nworld = rollouts_per_iter * cfg.K so all parallel rollouts share
    # one CUDA-graph launch per timestep.
    mppi_cfg = MPPIConfig.load(args.env)
    if args.warp:
        env = make_env(
            args.env,
            use_warp=True,
            nworld=cfg.rollouts_per_iter * mppi_cfg.K,
        )
        from src.mppi.batched_mppi import BatchedMPPI
        mppi = BatchedMPPI(env, mppi_cfg, num_conditions=cfg.rollouts_per_iter)
    else:
        env = make_env(args.env)
        mppi = MPPI(env, cfg=mppi_cfg)

    obs_dim = env.obs_dim
    act_dim = env.action_dim
    policy_cfg = PolicyConfig.for_env(args.env)
    bounds = env.action_bounds
    if args.deterministic:
        policy = DeterministicPolicy(obs_dim, act_dim, policy_cfg,
                                     device=device, action_bounds=bounds)
        print("policy: DeterministicPolicy (MSE regression)")
    else:
        policy = GaussianPolicy(obs_dim, act_dim, policy_cfg,
                                device=device, action_bounds=bounds)
        print("policy: GaussianPolicy (mu/sigma head)")

    if args.init_ckpt is not None:
        init_ckpt = Path(args.init_ckpt)
        if not init_ckpt.exists():
            raise FileNotFoundError(f"--init-ckpt not found: {init_ckpt}")
        blob = load_checkpoint(init_ckpt, map_location=device)
        # Auto-handles GaussianPolicy → DeterministicPolicy head conversion
        # so a Gaussian BC pretrain can warm-start a deterministic DAgger run.
        report = load_state_dict_into(policy, blob)
        print(f"loaded initial policy weights from {init_ckpt}: {report['msg']}")

    if args.warp:
        from src.gps.dagger_warp import WarpDAggerTrainer
        trainer = WarpDAggerTrainer(env, mppi, policy, cfg, rng=rng)
    else:
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

    # Per-iter CSV log, opened once, header written once, one row per iter
    # with explicit flush + fsync so a hard kill (SIGKILL/OOM/reboot) during
    # iter N+1 still leaves rows 0..N readable on disk. Same crash-safe
    # pattern as `gps_log.csv`. Uniform schema (NaN for inactive cols)
    # makes pd.read_csv across runs straightforward.
    csv_columns = [
        "iter", "beta", "new_samples", "buffer_size",
        "train_mse", "val_mse",
        "policy_mean_cost", "policy_std_cost",
        "best_iter", "best_cost",
    ]
    csv_path = run_dir / "dagger_log.csv"
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_columns)
    csv_writer.writeheader()
    csv_file.flush()

    best_iter: int | None = None
    best_cost = float("inf")
    last_ckpt_path: Path | None = None

    for k in range(cfg.dagger_iters):
        print(f"\n=== DAgger iter {k}/{cfg.dagger_iters - 1} ===")
        info = trainer.step(k)
        print(f"  beta={info['beta']:.2f}  new={info['new_samples']:,}  "
              f"buf={info['buffer_size']:,}  train_mse={info['train_mse']:.5f}  "
              f"val_mse={info['val_mse']:.5f}")

        # EMA-swap window: eval and per-iter save happen with the smoothed
        # weights when EMA is enabled, so the reported cost and the state_dict
        # on disk match. No-op context when EMA is off (bit-for-bit identical).
        with policy.ema_swapped_in() if hasattr(policy, "ema_swapped_in") else _nullctx():
            policy.eval()
            eval_stats = evaluate_policy(policy, env,
                                         n_episodes=cfg.n_eval_eps,
                                         episode_len=cfg.eval_ep_len,
                                         seed=cfg.seed)
            policy.train()
            print(f"  policy cost: {eval_stats['mean_cost']:.2f} ± {eval_stats['std_cost']:.2f}  ")
                #   f"(gap vs MPPI: {eval_stats['mean_cost'] - mppi_stats['mean_cost']:+.2f})")

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

        # Crash-safe per-iter row. flush() + fsync() so a hard kill during
        # iter k+1 still leaves iter k readable on disk.
        csv_writer.writerow({
            "iter": k,
            "beta": info["beta"],
            "new_samples": info["new_samples"],
            "buffer_size": info["buffer_size"],
            "train_mse": info["train_mse"],
            "val_mse": info["val_mse"],
            "policy_mean_cost": eval_stats["mean_cost"],
            "policy_std_cost": eval_stats["std_cost"],
            "best_iter": best_iter if best_iter is not None else float("nan"),
            "best_cost": (
                best_cost if best_cost != float("inf") else float("nan")
            ),
        })
        csv_file.flush()
        os.fsync(csv_file.fileno())

    csv_file.close()

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
