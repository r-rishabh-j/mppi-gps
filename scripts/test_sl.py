"""Pure supervised BC on cached MPPI rollouts.

Fits a `GaussianPolicy` (MSE on mean, default) or `DeterministicPolicy`
(direct action regression, via `--deterministic`) on (obs, action) pairs
loaded from an h5 produced by `collect_bc_demos.py`. Baseline for DAgger /
GPS — no policy-in-the-loop, no expert relabeling.

Creates a run dir `experiments/bc/<timestamp>_<env>_<name>/` containing:
    config.json        CLI args + dataclass configs + metadata
    iter_<k:03d>.pt    per-epoch wrapped checkpoints (kept only every
                       --ckpt-every epochs; always kept for the best epoch)
    best.pt / final.pt copies of best-by-val-mse / last epoch
    bc_log.csv         per-epoch train/val MSE (+ eval cost at eval epochs)
    loss.png           train/val curve
    <env>.mp4          rollout of the best policy with env-appropriate camera

Example:
    # 1. collect demos (cached — rerunning is a no-op unless --force)
    python -m scripts.collect_bc_demos --env acrobot

    # 2. train BC
    python -m scripts.test_sl --env acrobot --device auto
    python -m scripts.test_sl --env hopper --deterministic --device auto

    # warm-start from a prior checkpoint and continue training
    python -m scripts.test_sl --env acrobot --device auto \\
        --init-ckpt experiments/bc/<run>/best.pt --num-epochs 30
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import mediapy
import numpy as np
import torch
from tqdm.auto import tqdm

from src.envs import make_env
from src.mppi.mppi import MPPI
from src.policy.deterministic_policy import DeterministicPolicy
from src.policy.gaussian_policy import GaussianPolicy
from src.utils.config import MPPIConfig, PolicyConfig
from src.utils.device import pick_device
from src.utils.evaluation import evaluate_mppi, evaluate_policy
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


_ENVS = [
    "acrobot",
    "adroit_pen",
    "adroit_relocate",
    "point_mass",
    "hopper",
    "ur5_push",
]

# Camera per env — must match <camera name="..."> in the MuJoCo XML.
_CAMERA = {
    "hopper": "track",
    "acrobot": "fixed",
    "point_mass": "fixed",
    "adroit_pen": "vil_camera",
    "adroit_relocate": "vil_camera",
    "ur5_push": "birdseye_tilted_cam",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="acrobot", choices=_ENVS)
    p.add_argument("--demos", default=None,
                   help="path to h5 demos (default: data/<env>_bc.h5). "
                        "If the file is missing, you'll be told to run collect_bc_demos.")
    p.add_argument("--device", default="auto", help="auto | cpu | mps | cuda")
    p.add_argument("--deterministic", action="store_true",
                   help="use DeterministicPolicy (direct action regression) instead of "
                        "GaussianPolicy (MSE on mean, log_sigma head unused).")
    p.add_argument("--init-ckpt", default=None,
                   help="warm-start: load a wrapped or raw state_dict into the policy "
                        "before training. Must match --deterministic.")
    p.add_argument("--num-epochs", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=20000)
    p.add_argument("--val-frac", type=float, default=0,
                   help="fraction of trajectories (not transitions) held out for val")
    p.add_argument("--eval-every", type=int, default=5000,
                   help="run env eval every N epochs (0 disables mid-training eval)")
    p.add_argument("--ckpt-every", type=int, default=1000,
                   help="save iter_<epoch>.pt every N epochs (best.pt/final.pt always saved)")
    p.add_argument("--n-eval-eps", type=int, default=10)
    p.add_argument("--eval-ep-len", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--disable-tanh", action="store_true",
                   help="Disable the default tanh squash on the policy head "
                        "(see PolicyConfig.tanh_squash). Use when reproducing "
                        "pre-tanh experiments.")
    p.add_argument("--exp-name", default="run",
                   help="human-readable experiment name (goes in the run dir name)")
    p.add_argument("--exp-dir", default="checkpoints/bc",
                   help="parent dir under which <timestamp>_<env>_<name>/ is created")
    return p.parse_args()


def load_demos(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Returns ``(states, actions, sensordata)``.

    Shapes: ``states (M, T, obs_dim)``, ``actions (M, T, act_dim)``,
    ``sensordata (M, T, nsensor)`` or ``None`` for legacy datasets that
    pre-date sensordata capture.

    Trajectory-preserving so we can split train/val by trajectory before
    flattening.
    """
    with h5py.File(path, "r") as f:
        states = f["states"][:].astype(np.float32)
        actions = f["actions"][:].astype(np.float32)
        sensordata = (
            f["sensordata"][:].astype(np.float32) if "sensordata" in f else None
        )
    return states, actions, sensordata


def split_and_flatten(
    states: np.ndarray,
    actions: np.ndarray,
    val_frac: float,
    rng: np.random.Generator,
    sensordata: np.ndarray | None = None,
) -> tuple[
    tuple[np.ndarray, np.ndarray, np.ndarray | None],
    tuple[np.ndarray, np.ndarray, np.ndarray | None],
    int,
    int,
]:
    """Split-by-trajectory then flatten to (N, *) per side.

    If ``sensordata`` is provided, it's split + flattened on the same
    trajectory permutation so each row stays aligned with its (state,
    action). Returned as ``None`` per side when not provided.
    """
    M = states.shape[0]
    perm = rng.permutation(M)
    n_val = max(1, int(M * val_frac))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    def flatten(idx):
        s = states[idx].reshape(-1, states.shape[-1])
        a = actions[idx].reshape(-1, actions.shape[-1])
        sd = (
            sensordata[idx].reshape(-1, sensordata.shape[-1])
            if sensordata is not None
            else None
        )
        return s, a, sd

    return flatten(train_idx), flatten(val_idx), len(train_idx), len(val_idx)


@torch.no_grad()
def eval_mse(policy, obs: np.ndarray, actions: np.ndarray, batch: int = 16384) -> float:
    """Validation MSE in the policy's training-loss space.

    When the action-norm toggle is on, ``mse_step`` normalizes labels and
    computes MSE in the network's [-1, 1]-ish space; we mirror that here
    so val_mse stays interpretable as a fixed reference metric (the loss
    sense, ``best.pt`` is selected on the same objective being optimized).
    With the toggle off (`policy._has_act_norm = False`) this is byte-
    identical to physical-space MSE — the rescale is a no-op.
    """
    device = policy.device
    total, n = 0.0, 0
    for s in range(0, len(obs), batch):
        o = torch.as_tensor(obs[s:s + batch], dtype=torch.float32, device=device)
        a = torch.as_tensor(actions[s:s + batch], dtype=torch.float32, device=device)
        pred = policy.action(o)                 # physical
        diff = pred - a                         # physical residual
        if policy._has_act_norm:
            diff = diff / policy._act_scale     # → normalized space
        total += float((diff ** 2).sum().item())
        n += a.numel()
    return total / max(n, 1)


def main() -> None:
    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    # -------- Demos --------
    demo_path = Path(args.demos) if args.demos else Path(f"data/{args.env}_bc.h5")
    if not demo_path.exists():
        raise FileNotFoundError(
            f"{demo_path} not found. Collect first:\n"
            f"    python -m scripts.collect_bc_demos --env {args.env}"
        )
    states, actions, sensordata = load_demos(demo_path)
    M, T, obs_dim = states.shape
    act_dim = actions.shape[-1]
    sd_msg = (
        f", sensordata={sensordata.shape}" if sensordata is not None else ""
    )
    print(f"loaded {M} trajectories of length {T} from {demo_path}  "
          f"(obs_dim={obs_dim}, act_dim={act_dim}{sd_msg})")

    (tr_s, tr_a, tr_sd), (va_s, va_a, va_sd), n_tr, n_va = split_and_flatten(
        states, actions, args.val_frac, rng, sensordata=sensordata,
    )
    # ``tr_sd`` / ``va_sd`` are loaded and split alongside (states, actions)
    # so callers downstream of this can use them, but the BC training path
    # below trains on (obs, action) only — sensordata is deliberately kept
    # available for future obs-recompute hooks rather than wired into the
    # current loss.
    _ = (tr_sd, va_sd)
    print(f"train trajs: {n_tr}  val trajs: {n_va}")
    print(f"train samples: {len(tr_s):,}   val samples: {len(va_s):,}")

    # -------- Env + policy --------
    device = pick_device(args.device)
    env = make_env(args.env)
    mppi_cfg = MPPIConfig.load(args.env)

    if env.obs_dim != obs_dim or env.action_dim != act_dim:
        raise ValueError(
            f"demo/env shape mismatch: demos have obs_dim={obs_dim}, act_dim={act_dim}; "
            f"env has obs_dim={env.obs_dim}, act_dim={env.action_dim}. "
            f"Regenerate the demos after an env change: "
            f"python -m scripts.collect_bc_demos --env {args.env} --force"
        )

    policy_cfg = PolicyConfig.for_env(args.env)
    if args.disable_tanh:
        policy_cfg.tanh_squash = False
    PolicyCls = DeterministicPolicy if args.deterministic else GaussianPolicy
    policy = PolicyCls(obs_dim, act_dim, policy_cfg,
                       device=device, action_bounds=env.action_bounds)
    print(f"policy: {PolicyCls.__name__}  device: {device}  tanh={policy_cfg.tanh_squash}")

    if args.init_ckpt is not None:
        init_ckpt = Path(args.init_ckpt)
        if not init_ckpt.exists():
            raise FileNotFoundError(f"--init-ckpt not found: {init_ckpt}")
        blob = load_checkpoint(init_ckpt, map_location=device)
        # Auto-handles GaussianPolicy → DeterministicPolicy head conversion
        # so a Gaussian BC pretrain can warm-start a --deterministic re-fit.
        report = load_state_dict_into(policy, blob)
        print(f"loaded initial policy weights from {init_ckpt}: {report['msg']}")

    # -------- Run dir --------
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
        "demo_path": str(demo_path),
        "dataset": {
            "M": int(M), "T": int(T),
            "obs_dim": int(obs_dim), "act_dim": int(act_dim),
            "n_train_trajs": int(n_tr), "n_val_trajs": int(n_va),
            "n_train_samples": int(len(tr_s)), "n_val_samples": int(len(va_s)),
        },
        "configs": {
            "policy": policy_cfg,
            "mppi": mppi_cfg,
        },
        "best_epoch": None,
        "best_val_mse": None,
    })

    # -------- Training loop --------
    N = len(tr_s)
    idx = np.arange(N)
    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val = float("inf")
    best_epoch = -1
    last_ckpt_path: Path | None = None

    log_lines = [
        f"# BC on {args.env}, device={device}, policy={type(policy).__name__}",
        f"# demos={demo_path}  n_train={len(tr_s)}  n_val={len(va_s)}",
        "epoch,train_loss,val_mse,eval_mean_cost,eval_std_cost",
    ]

    # Gaussian: full diagonal-Gaussian NLL via train_weighted with uniform
    # weights. Both mu AND log_sigma heads are trained (σ shrinks/widens
    # to reflect expert action variance). Deterministic: MSE-on-mean —
    # there's no σ to fit on a single-head policy. The Gaussian → NLL
    # default applies project-wide; a checkpoint trained under MSE-on-
    # mean will load and re-train fine, just won't have meaningful σ
    # until the new training updates it.
    is_gaussian = isinstance(policy, GaussianPolicy)
    epoch_bar = tqdm(range(args.num_epochs), desc="BC", unit="ep")
    for epoch in epoch_bar:
        policy.train()
        rng.shuffle(idx)
        running, nb = 0.0, 0
        for start in range(0, N, args.batch_size):
            b = idx[start:start + args.batch_size]
            if is_gaussian:
                # Uniform weights → plain NLL (same path GPS uses for
                # distill_loss="nll" Gaussian S-step).
                w = np.ones(len(b), dtype=np.float32)
                running += policy.train_weighted(tr_s[b], tr_a[b], w)
            else:
                running += policy.mse_step(tr_s[b], tr_a[b])
            nb += 1
        # `train_loss` reflects whatever the policy class trained under
        # — NLL for Gaussian, MSE-on-mean for Deterministic. `val_mse` is
        # always MSE-on-mean (held-out, computed via `eval_mse`) so the
        # validation curve stays comparable across runs and policy classes.
        train_loss = running / max(nb, 1)
        val_mse = eval_mse(policy, va_s, va_a)
        train_losses.append(train_loss)
        val_losses.append(val_mse)

        # optional mid-training env eval — expensive on long episodes
        eval_mean = eval_std = float("nan")
        do_eval = args.eval_every > 0 and (
            (epoch + 1) % args.eval_every == 0 or epoch == args.num_epochs - 1
        )
        if do_eval:
            policy.eval()
            stats = evaluate_policy(policy, env,
                                    n_episodes=args.n_eval_eps,
                                    episode_len=args.eval_ep_len,
                                    seed=args.seed)
            eval_mean = stats["mean_cost"]
            eval_std = stats["std_cost"]

        is_best = val_mse < best_val
        if is_best:
            best_val = val_mse
            best_epoch = epoch

        # Checkpoints: always save best; save per-epoch only every --ckpt-every.
        # Per-epoch saves are wrapped so downstream eval_checkpoint auto-detects
        # the policy class. Best + final are plain copies of an iter ckpt.
        keep_this = (args.ckpt_every > 0 and (epoch + 1) % args.ckpt_every == 0) \
            or is_best or epoch == args.num_epochs - 1
        if keep_this:
            ckpt_path = run_dir / f"iter_{epoch:03d}.pt"
            save_checkpoint(
                ckpt_path, policy,
                round=epoch,
                train_loss=train_loss,
                val_mse=val_mse,
                eval_mean_cost=eval_mean,
                eval_std_cost=eval_std,
            )
            last_ckpt_path = ckpt_path
            if is_best:
                copy_as(ckpt_path, run_dir / "best.pt")

        if do_eval:
            log_lines.append(
                f"{epoch},{train_loss:.6f},{val_mse:.6f},"
                f"{eval_mean:.4f},{eval_std:.4f}"
            )
        else:
            log_lines.append(f"{epoch},{train_loss:.6f},{val_mse:.6f},,")
        (run_dir / "bc_log.csv").write_text("\n".join(log_lines))

        postfix = {"train": f"{train_loss:.4f}", "val": f"{val_mse:.4f}",
                   "best": f"{best_val:.4f}@{best_epoch}"}
        if do_eval:
            postfix["cost"] = f"{eval_mean:.1f}"
        epoch_bar.set_postfix(postfix)

    if last_ckpt_path is not None:
        copy_as(last_ckpt_path, run_dir / "final.pt")

    # -------- Loss curve --------
    plt.figure()
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.xlabel("epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "loss.png", dpi=120)
    plt.close()
    print(f"saved loss curve to {run_dir / 'loss.png'}")

    # -------- Reload best weights before final env eval --------
    best_blob = load_checkpoint(run_dir / "best.pt", map_location=device)
    policy.load_state_dict(best_blob["state_dict"])
    policy.eval()
    print(f"reloaded best checkpoint (epoch={best_epoch}, val_mse={best_val:.5f})")

    # -------- Final env eval: BC vs MPPI --------
    print("\n--- Evaluating BC policy ---")
    bc_stats = evaluate_policy(
        policy, env,
        n_episodes=args.n_eval_eps,
        episode_len=args.eval_ep_len,
        seed=args.seed,
        render=True,
        camera=_CAMERA.get(args.env),
    )

    print("--- Evaluating MPPI baseline ---")
    mppi = MPPI(env, cfg=mppi_cfg)
    mppi_stats = evaluate_mppi(
        env, mppi,
        n_episodes=args.n_eval_eps,
        episode_len=args.eval_ep_len,
        seed=args.seed,
    )

    print()
    print(f"BC policy:     {bc_stats['mean_cost']:8.2f} ± {bc_stats['std_cost']:.2f}")
    print(f"MPPI baseline: {mppi_stats['mean_cost']:8.2f} ± {mppi_stats['std_cost']:.2f}")
    print(f"gap (BC - MPPI): {bc_stats['mean_cost'] - mppi_stats['mean_cost']:+8.2f}")
    print()
    print("per-episode (BC vs MPPI):")
    for ep, (b, m) in enumerate(zip(bc_stats["per_ep"], mppi_stats["per_ep"])):
        print(f"  ep {ep:2d}  BC={b:8.2f}  MPPI={m:8.2f}  gap={b - m:+8.2f}")

    if bc_stats["frames"]:
        dt = env.model.opt.timestep * env._frame_skip
        fps = int(round(1.0 / dt))
        video_path = run_dir / f"{args.env}.mp4"
        mediapy.write_video(str(video_path), bc_stats["frames"], fps=fps)
        print(f"\nsaved rollout video to {video_path} (fps={fps})")

    # -------- Finalize config.json --------
    update_config(run_dir, {
        "end_time": datetime.now().isoformat(timespec="seconds"),
        "best_epoch": best_epoch,
        "best_val_mse": best_val,
        "eval": {
            "bc_mean_cost": bc_stats["mean_cost"],
            "bc_std_cost": bc_stats["std_cost"],
            "mppi_mean_cost": mppi_stats["mean_cost"],
            "mppi_std_cost": mppi_stats["std_cost"],
        },
    })

    print(f"\nrun dir: {run_dir}")
    env.close()


if __name__ == "__main__":
    main()
