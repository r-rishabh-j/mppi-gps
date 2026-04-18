"""Evaluate a saved policy checkpoint on an environment.

Example:
    python -m scripts.eval_checkpoint --env acrobot \
        --ckpt checkpoints/dagger/dagger_acrobot_iter09.pt \
        --n-eval 10 --render
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import mediapy

from src.envs import make_env
from src.policy.gaussian_policy import GaussianPolicy
from src.policy.deterministic_policy import DeterministicPolicy
from src.utils.config import PolicyConfig
from src.utils.device import pick_device
from src.utils.evaluation import evaluate_policy


_ENVS = ["acrobot", "half_cheetah", "point_mass", "hopper"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="acrobot", choices=_ENVS)
    p.add_argument("--ckpt", required=True, help="path to .pt policy state_dict")
    p.add_argument("--n-eval", type=int, default=10)
    p.add_argument("--eval-len", type=int, default=3000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto", help="auto | cpu | mps | cuda")
    p.add_argument("--render", action="store_true", help="save mp4 of episode 0")
    p.add_argument("--deterministic", action="store_true",
                   help="load weights into DeterministicPolicy instead of GaussianPolicy")
    p.add_argument("--video-out", default=None,
                   help="video path (default: <ckpt>_eval.mp4)")
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = pick_device(args.device)
    env = make_env(args.env)

    policy_cfg = PolicyConfig()
    bounds = env.action_bounds if policy_cfg.squash_tanh else None
    PolicyCls = DeterministicPolicy if args.deterministic else GaussianPolicy
    policy = PolicyCls(env.obs_dim, env.action_dim, policy_cfg,
                       device=device, action_bounds=bounds)
    policy.load_state_dict(torch.load(args.ckpt, map_location=device))
    policy.eval()

    print(f"env={args.env}  device={device}  policy={PolicyCls.__name__}  ckpt={args.ckpt}")
    stats = evaluate_policy(
        policy, env,
        n_episodes=args.n_eval,
        episode_len=args.eval_len,
        seed=args.seed,
        render=args.render,
    )
    print(f"mean cost: {stats['mean_cost']:.2f} +/- {stats['std_cost']:.2f}")
    for ep, c in enumerate(stats["per_ep"]):
        print(f"  ep {ep:2d}  cost={c:8.2f}")

    if args.render and stats["frames"]:
        ckpt_path = Path(args.ckpt)
        video_path = Path(args.video_out) if args.video_out else \
            ckpt_path.with_name(f"{ckpt_path.stem}_eval.mp4")
        mediapy.write_video(str(video_path), stats["frames"], fps=30)
        print(f"saved rollout video to {video_path}")

    env.close()


if __name__ == "__main__":
    main()