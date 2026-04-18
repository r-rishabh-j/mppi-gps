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
from src.utils.experiment import load_checkpoint
import json


_ENVS = ["acrobot", "half_cheetah", "point_mass", "hopper"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default=None, choices=_ENVS,
                   help="If omitted and --ckpt points to a run dir, env is read from config.json.")
    p.add_argument("--ckpt", required=True,
                   help="Path to a .pt checkpoint OR a run dir (in which case best.pt is used).")
    p.add_argument("--n-eval", type=int, default=10)
    p.add_argument("--eval-len", type=int, default=3000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto", help="auto | cpu | mps | cuda")
    p.add_argument("--render", action="store_true", help="save mp4 of episode 0")
    p.add_argument("--deterministic", action="store_true",
                   help="Load into DeterministicPolicy. Auto-detected when the checkpoint "
                        "or sibling config.json records policy_class.")
    p.add_argument("--video-out", default=None,
                   help="video path (default: <ckpt>_eval.mp4)")
    return p.parse_args()


def _resolve_ckpt_and_config(ckpt_arg: str) -> tuple[Path, dict | None]:
    """If ckpt_arg is a run dir, return (run_dir/best.pt, parsed config.json).
    If it's a file, return (path, sibling config.json if present else None)."""
    p = Path(ckpt_arg)
    if p.is_dir():
        best = p / "best.pt"
        if not best.exists():
            raise FileNotFoundError(f"{best} not found; pass a specific .pt file instead.")
        cfg_path = p / "config.json"
        cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else None
        return best, cfg
    cfg_path = p.parent / "config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else None
    return p, cfg


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = pick_device(args.device)
    ckpt_path, run_cfg = _resolve_ckpt_and_config(args.ckpt)
    blob = load_checkpoint(ckpt_path, map_location=device)

    env_name = args.env or (run_cfg["env"] if run_cfg else None)
    if env_name is None:
        raise ValueError("could not determine env; pass --env explicitly.")
    env = make_env(env_name)

    policy_class = blob.get("policy_class") or (run_cfg or {}).get("policy_class")
    use_deterministic = args.deterministic or (policy_class == "DeterministicPolicy")

    policy_cfg = PolicyConfig()
    bounds = env.action_bounds
    PolicyCls = DeterministicPolicy if use_deterministic else GaussianPolicy
    policy = PolicyCls(env.obs_dim, env.action_dim, policy_cfg,
                       device=device, action_bounds=bounds)
    policy.load_state_dict(blob["state_dict"])
    policy.eval()

    print(f"env={env_name}  device={device}  policy={PolicyCls.__name__}  ckpt={ckpt_path}")
    if run_cfg:
        print(f"  from run: name={run_cfg.get('name')}  start_time={run_cfg.get('start_time')}  "
              f"best_iter={run_cfg.get('best_iter')}  best_cost={run_cfg.get('best_cost')}")
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
        video_path = Path(args.video_out) if args.video_out else \
            ckpt_path.with_name(f"{ckpt_path.stem}_eval.mp4")
        mediapy.write_video(str(video_path), stats["frames"], fps=30)
        print(f"saved rollout video to {video_path}")

    env.close()


if __name__ == "__main__":
    main()