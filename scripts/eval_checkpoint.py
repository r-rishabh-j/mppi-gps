"""Evaluate a saved policy checkpoint."""
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


_ENVS = [
    "acrobot",
    "adroit_pen",
    "adroit_relocate",
    "point_mass",
    "hopper",
    "ur5_push",
]

# Per-env camera names (must match <camera name="..."> in the MuJoCo XML).
# Tracking cameras follow the body for locomotion envs; fixed cameras
# give a stable third-person view for the pendulum-style envs. Adroit
# envs use vil_camera (the canonical view from the VIL paper) so the
# rendered eval video frames the workspace correctly.
_CAMERA = {
    "hopper": "track",
    "acrobot": "fixed",
    "point_mass": "fixed",
    "adroit_pen": "vil_camera",
    "adroit_relocate": "vil_camera",
    "ur5_push": "birdseye_tilted_cam",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default=None, choices=_ENVS,
                   help="If omitted and --ckpt points to a run dir, env is read from config.json.")
    p.add_argument("--ckpt", required=True,
                   help="Path to a .pt checkpoint OR a run dir (in which case best.pt is used).")
    p.add_argument("--n-eval", type=int, default=1)
    p.add_argument("--eval-len", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto", help="auto | cpu | mps | cuda")
    p.add_argument("--render", action="store_true", help="save mp4 of episode 0", default=True)
    p.add_argument("--deterministic", action="store_true",
                   help="Load into DeterministicPolicy. Auto-detected from "
                        "checkpoint or sibling config.json's policy_class.")
    p.add_argument("--disable-tanh", action="store_true",
                   help="Force tanh squash OFF, overriding the run's "
                        "config.json. Mostly a safety override; the legacy "
                        "fallback already assumes tanh=False.")
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

    # Use the per-env config the trainers use (`PolicyConfig.for_env`).
    # An adroit checkpoint trained with hidden_dims=(512, 512, 512) would
    # otherwise mis-load into the (256, 256) default.
    policy_cfg = PolicyConfig.for_env(env_name)

    # Replay arch fields from config.json. Required so the policy is
    # constructed with the SAME architecture it was trained under —
    # otherwise state_dict shape mismatches (featurize="hand_crafted"
    # changes input dim) or silent behaviour changes (tanh is parameter-
    # free → squashes outputs the training never expected). Legacy
    # fallback assumes pre-flip arch (no tanh, learned-norm, dropout+ln).
    run_policy_cfg = (
        ((run_cfg or {}).get("configs") or {}).get("policy")
        or (run_cfg or {}).get("policy")
        or {}
    )
    LEGACY = {
        "tanh_squash": False,
        "featurize": "running_norm",
        "use_dropout": True,
        "use_layernorm": True,
        "dropout_p": None,
    }
    for key, legacy_value in LEGACY.items():
        setattr(policy_cfg, key, run_policy_cfg.get(key, legacy_value))
    if args.disable_tanh:
        policy_cfg.tanh_squash = False

    bounds = env.action_bounds
    PolicyCls = DeterministicPolicy if use_deterministic else GaussianPolicy
    policy = PolicyCls(env.obs_dim, env.action_dim, policy_cfg,
                       device=device, action_bounds=bounds)
    policy.load_state_dict(blob["state_dict"])
    # Health check: NaN/Inf weights silently produce NaN actions (clamp
    # doesn't sanitize NaN), tripping MuJoCo's "huge value in CTRL" at
    # t=0 and freezing the hand. Catch it here with a clear error.
    bad = sorted(
        k for k, v in policy.state_dict().items() if not torch.isfinite(v).all()
    )
    if bad:
        raise ValueError(
            f"checkpoint {ckpt_path} has non-finite weights in {len(bad)} "
            f"tensors (showing first 5: {bad[:5]}). The run was poisoned "
            f"by a NaN gradient — unrecoverable. Delete the run dir and "
            f"retrain (NaN guards in mppi/policy prevent recurrence)."
        )
    policy.eval()

    print(
        f"env={env_name}  device={device}  policy={PolicyCls.__name__}  "
        f"tanh={policy_cfg.tanh_squash}  featurize={policy_cfg.featurize}  "
        f"ckpt={ckpt_path}"
    )
    if run_cfg:
        print(f"  from run: name={run_cfg.get('name')}  start_time={run_cfg.get('start_time')}  "
              f"best_iter={run_cfg.get('best_iter')}  best_cost={run_cfg.get('best_cost')}")
    stats = evaluate_policy(
        policy, env,
        n_episodes=args.n_eval,
        episode_len=args.eval_len,
        seed=args.seed,
        render=args.render,
        camera=_CAMERA.get(env_name),
    )
    print(f"mean cost: {stats['mean_cost']:.2f} +/- {stats['std_cost']:.2f}")
    for ep, c in enumerate(stats["per_ep"]):
        print(f"  ep {ep:2d}  cost={c:8.2f}")

    if args.render and stats["frames"]:
        video_path = Path(args.video_out) if args.video_out else \
            ckpt_path.with_name(f"{ckpt_path.stem}_eval.mp4")
        # Match the sim's wall-clock speed: dt per env step = timestep * frame_skip
        dt = env.model.opt.timestep * env._frame_skip
        fps = int(round(1.0 / dt))
        mediapy.write_video(str(video_path), stats["frames"], fps=fps)
        print(f"saved rollout video to {video_path} (fps={fps})")

    env.close()


if __name__ == "__main__":
    main()