"""Train SAC/PPO baselines via stable-baselines3 for comparison with GPS.

The purpose of this script is to answer: how does a policy learned from
scratch by model-free RL compare to one distilled from MPPI via GPS?

GPS has the advantage of a model-based "teacher" (MPPI with MuJoCo) but
is limited by the quality of that teacher and the distillation process.
SAC/PPO learn directly from environment interaction without a model.

Usage:
    # Train SAC on Hopper-v5 for 500k steps
    python scripts/run_sb3_baseline.py --env Hopper-v5 --algo SAC

    # Train PPO on HalfCheetah-v5 for 1M steps
    python scripts/run_sb3_baseline.py --env HalfCheetah-v5 --algo PPO --total-timesteps 1000000

Results (mean reward/cost) are saved to checkpoints/sb3/<algo>_<env>_results.json.
Cost = -reward so that the comparison with MPPI/GPS (which minimise cost) is direct.
"""

import argparse
import json
import numpy as np
import gymnasium as gym
from pathlib import Path
from stable_baselines3 import SAC, PPO


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="Hopper-v5",
                   help="Gymnasium env id (e.g. Hopper-v5, HalfCheetah-v5)")
    p.add_argument("--algo", default="SAC", choices=["SAC", "PPO"])
    p.add_argument("--total-timesteps", type=int, default=500_000)
    p.add_argument("--n-eval", type=int, default=10)
    p.add_argument("--eval-len", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-dir", type=str, default="checkpoints/sb3")
    return p.parse_args()


def evaluate(model, env_id: str, n_episodes: int, episode_len: int, seed: int) -> dict:
    """Evaluate a trained SB3 model on the given environment.

    Uses deterministic actions (no exploration) and reports both reward
    (native Gymnasium convention) and cost (-reward, for MPPI/GPS comparison).
    """
    env = gym.make(env_id)
    returns = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        ep_reward = 0.0
        for _ in range(episode_len):
            # deterministic=True → use mean action, no exploration noise
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            if terminated or truncated:
                break
        returns.append(ep_reward)
    env.close()
    arr = np.array(returns)
    return {
        "mean_reward": float(arr.mean()),
        "std_reward": float(arr.std()),
        "per_ep": arr.tolist(),
        # Negate reward to get cost (MPPI/GPS minimise cost)
        "mean_cost": float(-arr.mean()),
        "std_cost": float(arr.std()),
    }


def main():
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    env = gym.make(args.env)
    AlgoCls = SAC if args.algo == "SAC" else PPO
    model = AlgoCls("MlpPolicy", env, seed=args.seed, verbose=1)

    print(f"Training {args.algo} on {args.env} for {args.total_timesteps} steps...")
    model.learn(total_timesteps=args.total_timesteps)

    # Save the trained model (can be reloaded with AlgoCls.load())
    tag = f"{args.algo.lower()}_{args.env}"
    model_path = save_dir / tag
    model.save(str(model_path))
    print(f"saved model to {model_path}")

    # Evaluate and save results
    stats = evaluate(model, args.env, args.n_eval, args.eval_len, args.seed)
    print(f"\n{args.algo} on {args.env}:")
    print(f"  mean reward: {stats['mean_reward']:.2f} +/- {stats['std_reward']:.2f}")
    print(f"  mean cost:   {stats['mean_cost']:.2f} (for MPPI/GPS comparison)")

    results_path = save_dir / f"{tag}_results.json"
    results_path.write_text(json.dumps(stats, indent=2))
    print(f"saved results to {results_path}")

    env.close()


if __name__ == "__main__":
    main()
