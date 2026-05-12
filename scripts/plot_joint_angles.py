"""Side-by-side qpos trajectories: MPPI vs trained policy from the same seed."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np
import torch

from src.envs import make_env
from src.mppi.mppi import MPPI
from src.policy.gaussian_policy import GaussianPolicy
from src.policy.deterministic_policy import DeterministicPolicy
from src.utils.config import MPPIConfig, PolicyConfig
from src.utils.device import pick_device
from src.utils.experiment import load_checkpoint


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True,
                   help="Run dir with config.json + iter_*.pt + best.pt.")
    p.add_argument("--ckpt", default="best.pt",
                   help="Checkpoint filename inside --run-dir. Default: best.pt.")
    p.add_argument("--steps", type=int, default=200,
                   help="Number of env steps to roll out for each method.")
    p.add_argument("--seed", type=int, default=42,
                   help="np.random seed used before BOTH resets so the two "
                        "rollouts start from the exact same initial state.")
    p.add_argument("--joints", default=None,
                   help="Comma-separated qpos indices to plot (e.g. '3,4,5'). "
                        "Default: all nq joints.")
    p.add_argument("--mppi-config", default=None,
                   help="Override MPPI config name (e.g. 'hopper'). "
                        "Default: env name from config.json.")
    p.add_argument("--device", default="auto", help="auto | cpu | mps | cuda")
    p.add_argument("--out", default=None,
                   help="Output PNG path. Default: <run-dir>/joint_angles.png.")
    p.add_argument("--y-range", default=None,
                   help="Y-axis range as 'lo,hi' (e.g. '-2,2' to match GPS paper). "
                        "Default: auto.")
    return p.parse_args()


def _build_policy(env, env_name: str, blob: dict, run_cfg: dict | None,
                  device) -> torch.nn.Module:
    """Resolve policy class from the checkpoint and load weights."""
    policy_class = blob.get("policy_class") or (run_cfg or {}).get("policy_class")
    use_det = policy_class == "DeterministicPolicy"
    PolicyCls = DeterministicPolicy if use_det else GaussianPolicy
    policy_cfg = PolicyConfig.for_env(env_name)
    bounds = env.action_bounds
    policy = PolicyCls(env.obs_dim, env.action_dim, policy_cfg,
                       device=device, action_bounds=bounds)
    policy.load_state_dict(blob["state_dict"])
    policy.eval()
    return policy


def _joint_names(model) -> list[str]:
    """One name per qpos slot; expanded for free / ball joints."""
    names: list[str] = []
    for j in range(model.njnt):
        adr = model.jnt_qposadr[j]
        next_adr = model.jnt_qposadr[j + 1] if j + 1 < model.njnt else model.nq
        width = next_adr - adr
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or f"j{j}"
        if width == 1:
            names.append(name)
        else:
            names.extend(f"{name}[{k}]" for k in range(width))
    # Pad/truncate to nq just in case.
    if len(names) < model.nq:
        names.extend(f"q[{i}]" for i in range(len(names), model.nq))
    return names[: model.nq]


def _rollout_mppi(env, controller: MPPI, steps: int) -> np.ndarray:
    """Return (steps, nq) array of qpos over the rollout."""
    qpos_log = np.zeros((steps, env.model.nq), dtype=np.float64)
    controller.reset()
    for t in range(steps):
        state = env.get_state()
        action, _ = controller.plan_step(state)
        env.step(action)
        qpos_log[t] = env.data.qpos.copy()
    return qpos_log


def _rollout_policy(env, policy, steps: int) -> np.ndarray:
    """Return (steps, nq) array of qpos over the rollout (greedy actions)."""
    qpos_log = np.zeros((steps, env.model.nq), dtype=np.float64)
    for t in range(steps):
        obs = env._get_obs()
        action = policy.act_np(obs)
        env.step(action)
        qpos_log[t] = env.data.qpos.copy()
    return qpos_log


def main() -> None:
    args = parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.is_dir():
        raise NotADirectoryError(run_dir)
    cfg_path = run_dir / "config.json"
    run_cfg = json.loads(cfg_path.read_text())
    env_name = run_cfg["env"]

    device = pick_device(args.device)
    env = make_env(env_name)
    nq = env.model.nq

    if args.joints is None:
        joint_idx = list(range(nq))
    else:
        joint_idx = [int(s) for s in args.joints.split(",")]
        if any(i < 0 or i >= nq for i in joint_idx):
            raise ValueError(f"--joints out of range; env has nq={nq}")

    names = _joint_names(env.model)
    print(f"[joint-plot] env={env_name}  nq={nq}  steps={args.steps}  seed={args.seed}")
    print(f"[joint-plot] plotting joints {joint_idx} = "
          f"{[names[i] for i in joint_idx]}")

    # ------- MPPI rollout -------
    try:
        mppi_cfg = MPPIConfig.load(args.mppi_config or env_name)
    except FileNotFoundError:
        print(f"[joint-plot] no configs/{env_name}_best.json — falling back to "
              f"in-config MPPI hyperparams.")
        mppi_cfg = MPPIConfig(**run_cfg["configs"]["mppi"])
    controller = MPPI(env, mppi_cfg)
    print(f"[joint-plot] MPPI rollout — K={mppi_cfg.K}, H={mppi_cfg.H}, "
          f"open_loop={mppi_cfg.open_loop_steps}")
    np.random.seed(args.seed)
    env.reset()
    qpos_mppi = _rollout_mppi(env, controller, args.steps)

    # ------- Policy rollout -------
    ckpt_path = run_dir / args.ckpt
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    blob = load_checkpoint(ckpt_path, map_location=device)
    policy = _build_policy(env, env_name, blob, run_cfg, device)
    print(f"[joint-plot] policy rollout — ckpt={ckpt_path.name}")
    np.random.seed(args.seed)   # SAME init as the MPPI rollout
    env.reset()
    qpos_policy = _rollout_policy(env, policy, args.steps)

    # Sanity: confirm the two rollouts started from the same qpos.
    # (Both saw the same seed before reset and the same env, so this
    # should be exactly equal.)
    if not np.allclose(qpos_mppi[0], qpos_policy[0], atol=1e-6):
        # Note: index 0 is post-FIRST-step, so a small divergence is
        # already possible even with identical inits since actions
        # differ. Print the pre-step state instead via env.reset diag.
        print("[joint-plot] note: qpos[0] differs across rollouts — that's "
              "expected (it's the state AFTER the first action).")

    # ------- Plot -------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    t = np.arange(args.steps)
    cmap = plt.colormaps.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(len(joint_idx))]

    for ax, qpos, title in [
        (axes[0], qpos_mppi, "Model-Predictive Control"),
        (axes[1], qpos_policy, "Neural Network Policy"),
    ]:
        for c, j in zip(colors, joint_idx):
            ax.plot(t, qpos[:, j], color=c, linewidth=1.0,
                    label=names[j])
        ax.set_title(title)
        ax.set_xlabel("env step")
        ax.grid(True, alpha=0.3)
        ax.axhline(0.0, color="black", linewidth=0.5, alpha=0.4)

    axes[0].set_ylabel("joint angle")
    if args.y_range is not None:
        lo, hi = (float(s) for s in args.y_range.split(","))
        axes[0].set_ylim(lo, hi)

    # Single legend on the right side, outside the second subplot.
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
                   fontsize=8, frameon=False)
    fig.tight_layout()

    out_path = Path(args.out) if args.out else (run_dir / "joint_angles.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[joint-plot] → {out_path}")

    env.close()


if __name__ == "__main__":
    main()
