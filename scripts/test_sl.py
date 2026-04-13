"""Pure SL behavior cloning with MSE loss on MPPI executed trajectories.

Loads (M, T) executed (state, action) pairs from collect_bc_demos.py and fits
the policy mean by MSE. No weights, no NLL — just regression.
"""

import numpy as np
import h5py
import torch
import mujoco
import mediapy
import matplotlib.pyplot as plt
from dataclasses import dataclass
from pathlib import Path

from src.policy.gaussian_policy import GaussianPolicy
from src.utils.config import PolicyConfig, MPPIConfig
from src.utils.evaluation import evaluate_policy, evaluate_mppi
from src.envs.acrobot import Acrobot
from src.mppi.mppi import MPPI


@dataclass
class BCConfig:
    demo_path:   Path  = Path("data/acrobot_bc.h5")
    ckpt_path:   Path  = Path("checkpoints/bc_acrobot.pt")
    loss_plot:   Path  = Path("bc_loss.png")
    video_path:  Path  = Path("policy_sl.mp4")

    obs_dim:     int   = 4
    act_dim:     int   = 1

    batch_size:  int   = 4096
    num_epochs:  int   = 50
    val_frac:    float = 0.2       # fraction of trajectories held out
    n_eval_eps:  int   = 10
    eval_ep_len: int   = 500

    seed:        int   = 0


# ---------- data ----------
def load_demos(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Returns (M, T, obs_dim), (M, T, act_dim) — preserves trajectory structure
    so we can split train/val by trajectory before flattening."""
    with h5py.File(path, "r") as f:
        states  = f["states"][:].astype(np.float32)
        actions = f["actions"][:].astype(np.float32)
    return states, actions


def split_and_flatten(states: np.ndarray,
                      actions: np.ndarray,
                      val_frac: float,
                      rng: np.random.Generator):
    """Split trajectories (not transitions) into train/val, then flatten each
    half to (N, obs_dim) / (N, act_dim)."""
    M = states.shape[0]
    perm = rng.permutation(M)
    n_val = max(1, int(M * val_frac))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    def flatten(idx):
        s = states[idx].reshape(-1, states.shape[-1])
        a = actions[idx].reshape(-1, actions.shape[-1])
        return s, a

    tr_s, tr_a = flatten(train_idx)
    va_s, va_a = flatten(val_idx)
    return (tr_s, tr_a), (va_s, va_a), len(train_idx), len(val_idx)


# ---------- training ----------
def train_step_mse(policy: GaussianPolicy,
                   obs: np.ndarray,
                   actions: np.ndarray) -> float:
    """One Adam step on the MSE between policy mean and target action."""
    obs_t = torch.as_tensor(obs, dtype=torch.float32)
    act_t = torch.as_tensor(actions, dtype=torch.float32)

    mu, _ = policy.forward(obs_t)
    loss = ((mu - act_t) ** 2).mean()

    policy.optimizer.zero_grad()
    loss.backward()
    policy.optimizer.step()
    return loss.item()


@torch.no_grad()
def eval_mse(policy: GaussianPolicy,
             obs: np.ndarray,
             actions: np.ndarray,
             batch: int = 16384) -> float:
    """Mean MSE over a dataset, computed in chunks to bound memory."""
    total, n = 0.0, 0
    for s in range(0, len(obs), batch):
        o = torch.as_tensor(obs[s:s + batch], dtype=torch.float32)
        a = torch.as_tensor(actions[s:s + batch], dtype=torch.float32)
        mu, _ = policy.forward(o)
        total += ((mu - a) ** 2).sum().item()
        n     += a.numel()
    return total / max(n, 1)


# ---------- main ----------
def main(cfg: BCConfig = BCConfig()) -> None:
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)

    states, actions = load_demos(cfg.demo_path)
    M, T, _ = states.shape
    print(f"loaded {M} trajectories of length {T}")

    (tr_s, tr_a), (va_s, va_a), n_tr, n_va = split_and_flatten(
        states, actions, cfg.val_frac, rng,
    )
    print(f"train trajs: {n_tr}  val trajs: {n_va}")
    print(f"train samples: {len(tr_s):,}   val samples: {len(va_s):,}")

    policy = GaussianPolicy(cfg.obs_dim, cfg.act_dim, PolicyConfig())

    cfg.ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    train_losses: list[float] = []
    val_losses:   list[float] = []
    best_val = float("inf")

    N = len(tr_s)
    idx = np.arange(N)

    for epoch in range(cfg.num_epochs):
        rng.shuffle(idx)
        running, n_batches = 0.0, 0
        for start in range(0, N, cfg.batch_size):
            b = idx[start:start + cfg.batch_size]
            running += train_step_mse(policy, tr_s[b], tr_a[b])
            n_batches += 1

        train_losses.append(running / n_batches)
        val_losses.append(eval_mse(policy, va_s, va_a))

        if val_losses[-1] < best_val:
            best_val = val_losses[-1]
            torch.save(policy.state_dict(), cfg.ckpt_path)
            tag = "  ↳ new best"
        else:
            tag = ""

        print(f"epoch {epoch:3d}  "
              f"train_mse={train_losses[-1]:.5f}  "
              f"val_mse={val_losses[-1]:.5f}{tag}")

    # loss curve
    plt.figure()
    plt.plot(train_losses, label="train")
    plt.plot(val_losses,   label="val")
    plt.xlabel("epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(cfg.loss_plot, dpi=120)
    print(f"saved loss curve to {cfg.loss_plot}")

    # reload best-val checkpoint before env eval
    policy.load_state_dict(torch.load(cfg.ckpt_path))
    policy.eval()
    print(f"reloaded best-val checkpoint (val_mse={best_val:.5f})")

    # multi-seed env eval
    env = Acrobot()
    stats = evaluate_policy(
        policy, env,
        n_episodes=cfg.n_eval_eps,
        episode_len=cfg.eval_ep_len,
        seed=cfg.seed,
        render=True,
    )

    # MPPI baseline on the exact same initial conditions
    mppi = MPPI(env, cfg=MPPIConfig.load("acrobot"))
    mppi_stats = evaluate_mppi(
        env, mppi,
        n_episodes=cfg.n_eval_eps,
        episode_len=cfg.eval_ep_len,
        seed=cfg.seed,
    )

    print()
    print(f"BC  policy:    {stats['mean_cost']:8.2f} ± {stats['std_cost']:.2f}")
    print(f"MPPI baseline: {mppi_stats['mean_cost']:8.2f} ± {mppi_stats['std_cost']:.2f}")
    print(f"gap (BC - MPPI): {stats['mean_cost'] - mppi_stats['mean_cost']:+8.2f}")
    print()
    print("per-episode (BC vs MPPI):")
    for ep, (b, m) in enumerate(zip(stats["per_ep"], mppi_stats["per_ep"])):
        print(f"  ep {ep:2d}  BC={b:8.2f}  MPPI={m:8.2f}  gap={b - m:+8.2f}")

    if stats["frames"]:
        mediapy.write_video(str(cfg.video_path), stats["frames"], fps=30)
        print(f"saved rollout video to {cfg.video_path}")

    env.close()


if __name__ == "__main__":
    main()
