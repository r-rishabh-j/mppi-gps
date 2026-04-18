"""DAgger with MPPI as the expert — distills MPPI into a GaussianPolicy.

Example:
    python -m scripts.run_dagger --env acrobot --dagger-iters 10 \
        --rollouts-per-iter 20 --episode-len 200 --device auto

    python -m scripts.run_dagger --env acrobot --deterministic \
        --init-ckpt checkpoints/dagger/dagger_acrobot_iter09.pt \
        --dagger-iters 5 --rollouts-per-iter 20 --episode-len 200 --device auto

MPPI runs on CPU (MuJoCo). Policy training uses the device selected by
`--device` (auto → cuda → mps → cpu).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from src.envs.acrobot import Acrobot
from src.mppi.mppi import MPPI
from src.policy.gaussian_policy import GaussianPolicy
from src.policy.deterministic_policy import DeterministicPolicy
from src.utils.config import DAggerConfig, MPPIConfig, PolicyConfig
from src.utils.device import pick_device
from src.utils.evaluation import evaluate_mppi, evaluate_policy
from src.gps.dagger import DAggerTrainer


def make_env(name: str):
    if name == "acrobot":
        return Acrobot()
    raise ValueError(f"unsupported env for DAgger: {name}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="acrobot")
    p.add_argument("--dagger-iters", type=int, default=10)
    p.add_argument("--rollouts-per-iter", type=int, default=20)
    p.add_argument("--episode-len", type=int, default=200)
    p.add_argument("--beta-schedule", default="linear", choices=["linear", "constant_zero"])
    p.add_argument("--distill-epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--buffer-cap", type=int, default=200_000)
    p.add_argument("--n-eval-eps", type=int, default=10)
    p.add_argument("--eval-ep-len", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    p.add_argument("--deterministic", action="store_true",
                   help="use DeterministicPolicy (direct action regression) instead of GaussianPolicy")
    p.add_argument("--init-ckpt", default=None,
                   help="path to a policy checkpoint to load before warmup / DAgger training")
    p.add_argument("--seed-from", default=None,
                   help="path to existing BC h5 (e.g. data/acrobot_bc.h5) to warm-start the buffer")
    p.add_argument("--warmup-rollouts", type=int, default=0,
                   help="Pre-DAgger: collect this many pure-MPPI rollouts and BC-train the policy on them")
    p.add_argument("--warmup-epochs", type=int, default=20,
                   help="Epochs of BC pre-training on the warmup rollouts (ignored if --warmup-rollouts=0)")
    p.add_argument("--ckpt-dir", default="checkpoints/dagger")
    p.add_argument("--results-dir", default="results/dagger_acrobot")
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
    )

    device = pick_device(args.device)
    print(f"policy device: {device}")

    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    env = make_env(args.env)
    mppi_cfg = MPPIConfig.load(args.env)
    mppi = MPPI(env, cfg=mppi_cfg)

    obs_dim = env.obs_dim
    act_dim = env.action_dim
    policy_cfg = PolicyConfig()
    bounds = env.action_bounds if policy_cfg.squash_tanh else None
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
        try:
            policy.load_state_dict(torch.load(init_ckpt, map_location=device))
        except RuntimeError as exc:
            raise RuntimeError(
                f"failed to load --init-ckpt {init_ckpt}; make sure the checkpoint "
                f"matches --deterministic={args.deterministic}"
            ) from exc
        print(f"loaded initial policy weights from {init_ckpt}")

    trainer = DAggerTrainer(env, mppi, policy, cfg, rng=rng)
    if args.seed_from is not None:
        seed_path = Path(args.seed_from)
        if seed_path.exists():
            print(f"seeding buffer from {seed_path}")
            trainer.seed_from_h5(seed_path)
            print(f"  buffer size after seeding: {trainer.buffer_size():,}")
        else:
            print(f"warning: --seed-from {seed_path} does not exist; skipping")

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.warmup_rollouts > 0:
        print(f"\nwarmup: collecting {args.warmup_rollouts} pure-MPPI rollouts + "
              f"{args.warmup_epochs} epochs of BC pre-training...")
        losses = trainer.warmup(args.warmup_rollouts, args.warmup_epochs)
        if losses:
            print(f"  warmup train_mse: {losses[0]:.5f} → {losses[-1]:.5f}  "
                  f"(buf={trainer.buffer_size():,})")

    # --- MPPI baseline (once, shared across iterations) ---
    print("\nevaluating MPPI baseline...")
    mppi_eval = MPPI(env, cfg=mppi_cfg)
    mppi_stats = evaluate_mppi(env, mppi_eval,
                                n_episodes=cfg.n_eval_eps,
                                episode_len=cfg.eval_ep_len,
                                seed=cfg.seed)
    print(f"MPPI baseline: {mppi_stats['mean_cost']:.2f} ± {mppi_stats['std_cost']:.2f}")

    log_lines: list[str] = [
        f"# DAgger on {args.env}, device={device}",
        f"# MPPI baseline: mean={mppi_stats['mean_cost']:.4f} std={mppi_stats['std_cost']:.4f}",
        "iter,beta,new_samples,buffer_size,train_mse,val_mse,policy_mean_cost,policy_std_cost",
    ]

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
        print(f"  policy cost: {eval_stats['mean_cost']:.2f} ± {eval_stats['std_cost']:.2f}  "
              f"(gap vs MPPI: {eval_stats['mean_cost'] - mppi_stats['mean_cost']:+.2f})")

        log_lines.append(
            f"{k},{info['beta']:.4f},{info['new_samples']},{info['buffer_size']},"
            f"{info['train_mse']:.6f},{info['val_mse']:.6f},"
            f"{eval_stats['mean_cost']:.4f},{eval_stats['std_cost']:.4f}"
        )

        ckpt_path = ckpt_dir / f"dagger_{args.env}_iter{k:02d}.pt"
        torch.save(policy.state_dict(), ckpt_path)

    (results_dir / "dagger_log.csv").write_text("\n".join(log_lines))
    print(f"\nlog written to {results_dir / 'dagger_log.csv'}")

    env.close()


if __name__ == "__main__":
    main()
