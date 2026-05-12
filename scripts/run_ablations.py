"""GPS ablations (alpha, K, num_conditions, policy-vs-MPPI wallclock).

Saves to ``results/ablations/<env>_ablations.json``.
"""

import argparse
import json
import time
import numpy as np
import torch
from pathlib import Path

from src.gps.mppi_gps import MPPIGPS
from src.utils.config import MPPIConfig, PolicyConfig, GPSConfig
from src.utils.evaluation import evaluate_policy, evaluate_mppi
from src.mppi.mppi import MPPI
from src.envs import make_env as _make_env
from src.utils.seeding import seed_everything


def make_env(name: str):
    return _make_env(name)


def run_gps_trial(env_name, gps_overrides: dict, seed: int, eval_len: int = 500, n_eval: int = 5):
    """One GPS train + eval cycle. Returns cost / loss / time stats."""
    seed_everything(seed)

    env = make_env(env_name)
    mppi_cfg = MPPIConfig.load(env_name)
    policy_cfg = PolicyConfig()
    gps_cfg = GPSConfig()
    for k, v in gps_overrides.items():
        setattr(gps_cfg, k, v)

    t0 = time.time()
    gps = MPPIGPS(env, mppi_cfg, policy_cfg, gps_cfg)
    history = gps.train()
    train_time = time.time() - t0

    gps.policy.eval()
    gps_stats = evaluate_policy(gps.policy, env, n_eval, eval_len, seed)

    mppi = MPPI(env, mppi_cfg)
    mppi_stats = evaluate_mppi(env, mppi, n_eval, eval_len, seed)

    env.close()

    return {
        "gps_mean_cost": gps_stats["mean_cost"],
        "gps_std_cost": gps_stats["std_cost"],
        "mppi_mean_cost": mppi_stats["mean_cost"],
        "final_loss": history.distill_losses[-1] if history.distill_losses else None,
        "train_time_s": train_time,
        "costs_curve": history.iteration_costs,
    }


def wallclock_comparison(env_name: str, n_steps: int = 100, seed: int = 0):
    """Time policy forward vs MPPI planning. Returns ms-per-step + speedup."""
    env = make_env(env_name)
    mppi_cfg = MPPIConfig.load(env_name)

    # JIT'd MLP matching the GPS policy architecture.
    policy = torch.jit.script(
        torch.nn.Sequential(
            torch.nn.Linear(env.obs_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 2 * env.action_dim),
        )
    )
    obs = torch.randn(1, env.obs_dim)
    # Warmup: JIT compilation + cache warming
    for _ in range(10):
        policy(obs)
    t0 = time.time()
    for _ in range(n_steps):
        policy(obs)
    policy_time = (time.time() - t0) / n_steps

    # -- MPPI planning timing --
    mppi = MPPI(env, mppi_cfg)
    env.reset()
    state = env.get_state()
    # Warmup: first few calls are slower due to thread pool init
    for _ in range(3):
        mppi.plan_step(state)
    mppi.reset()
    t0 = time.time()
    for _ in range(n_steps):
        action, _ = mppi.plan_step(state)
    mppi_time = (time.time() - t0) / n_steps

    env.close()
    return {
        "policy_ms": policy_time * 1000,
        "mppi_ms": mppi_time * 1000,
        "speedup": mppi_time / max(policy_time, 1e-9),
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="acrobot")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=str, default="results/ablations")
    p.add_argument("--gps-iters", type=int, default=20,
                   help="GPS iterations per ablation trial (keep low for speed)")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_results = {}

    # Reduced episode length + iterations for faster ablation sweeps
    base_overrides = {"num_iterations": args.gps_iters, "episode_length": 300}

    # ========== 1. Alpha ablation ==========
    # alpha=0 means no policy augmentation (MPPI plans independently of policy)
    # Higher alpha biases MPPI toward policy-representable trajectories
    print("=== Alpha ablation ===")
    alpha_results = {}
    for alpha in [0.0, 0.01, 0.1, 0.5]:
        print(f"\nalpha={alpha}")
        overrides = {**base_overrides, "policy_augmented_alpha": alpha}
        r = run_gps_trial(args.env, overrides, args.seed)
        alpha_results[str(alpha)] = r
        print(f"  GPS cost={r['gps_mean_cost']:.2f}, MPPI cost={r['mppi_mean_cost']:.2f}")
    all_results["alpha"] = alpha_results

    # ========== 2. Sample count (K) ablation ==========
    # More MPPI samples → better trajectory distribution → better distillation?
    print("\n=== Sample count (K) ablation ===")
    k_results = {}
    for K in [128, 256, 512]:
        print(f"\nK={K}")
        overrides = {**base_overrides}
        seed_everything(args.seed)
        env = make_env(args.env)
        mppi_cfg = MPPIConfig.load(args.env)
        mppi_cfg.K = K  # override sample count
        gps_cfg = GPSConfig(**overrides)
        gps = MPPIGPS(env, mppi_cfg, PolicyConfig(), gps_cfg)
        gps.mppi = MPPI(env, mppi_cfg)  # rebuild controller with new K
        history = gps.train()
        gps.policy.eval()
        gps_stats = evaluate_policy(gps.policy, env, 5, 500, args.seed)
        env.close()
        k_results[str(K)] = {
            "gps_mean_cost": gps_stats["mean_cost"],
            "costs_curve": history.iteration_costs,
        }
        print(f"  GPS cost={gps_stats['mean_cost']:.2f}")
    all_results["K"] = k_results

    # ========== 3. Number of conditions ablation ==========
    # More conditions → more diverse training data → better generalisation?
    print("\n=== Num conditions ablation ===")
    nc_results = {}
    for nc in [3, 5, 10]:
        print(f"\nnum_conditions={nc}")
        overrides = {**base_overrides, "num_conditions": nc}
        r = run_gps_trial(args.env, overrides, args.seed)
        nc_results[str(nc)] = r
        print(f"  GPS cost={r['gps_mean_cost']:.2f}")
    all_results["num_conditions"] = nc_results

    # ========== 4. Wall-clock comparison ==========
    print("\n=== Wall-clock comparison ===")
    wc = wallclock_comparison(args.env, n_steps=100, seed=args.seed)
    all_results["wallclock"] = wc
    print(f"  policy: {wc['policy_ms']:.3f} ms, MPPI: {wc['mppi_ms']:.3f} ms, speedup: {wc['speedup']:.1f}x")

    # ========== Save all results ==========
    results_path = out_dir / f"{args.env}_ablations.json"
    results_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nsaved all ablation results to {results_path}")


if __name__ == "__main__":
    main()
