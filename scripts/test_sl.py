"""Weighted BC on MPPI rollouts — basic distillation sanity check."""

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
from src.envs.acrobot import Acrobot
from src.mppi.mppi import MPPI


@dataclass
class BCConfig:
    demo_path:   Path  = Path("data/acrobot_demos.h5")
    ckpt_path:   Path  = Path("checkpoints/bc_acrobot.pt")
    loss_plot:   Path  = Path("bc_loss.png")
    video_path:  Path  = Path("policy_sl.mp4")

    obs_dim:     int   = 4
    act_dim:     int   = 1

    batch_size:  int   = 4096
    num_epochs:  int   = 50
    val_frac:    float = 0.2       # fraction of conditions held out
    H_keep:      int   = 5         # horizon steps per rollout to keep

    n_eval_eps:  int   = 10
    eval_ep_len: int   = 500

    seed:        int   = 0

# data
def split_conditions(path: Path,
                     val_frac: float,
                     rng: np.random.Generator) -> tuple[list[str], list[str]]:
    with h5py.File(path, "r") as f:
        keys = sorted(f.keys())
    keys = list(keys)
    rng.shuffle(keys)
    n_val = max(1, int(len(keys) * val_frac))
    return keys[n_val:], keys[:n_val]    # train, val


def load_demos(path: Path,
               condition_keys: list[str],
               H_keep: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flatten (condition, step, K, H_keep) rollouts to (N, obs_dim), (N, act_dim), (N,).

    Weights are MPPI per-step softmax weights (sum to 1 per planning step),
    broadcast across the H_keep kept horizon timesteps of each rollout.
    """
    all_obs, all_act, all_w = [], [], []
    with h5py.File(path, "r") as f:
        for cond_key in condition_keys:
            cond = f[cond_key]
            for step_key in sorted(cond.keys()):
                step = cond[step_key]
                obs_kh = step["obs"][:, :H_keep, :]       # (K, H_keep, obs_dim)
                act_kh = step["actions"][:, :H_keep, :]   # (K, H_keep, act_dim)
                w_k    = step["weights"][:]               # (K,)  sums to 1

                K, H, _ = obs_kh.shape
                all_obs.append(obs_kh.reshape(K * H, -1))
                all_act.append(act_kh.reshape(K * H, -1))
                all_w.append(np.repeat(w_k, H))           # (K * H,)

    return (np.concatenate(all_obs).astype(np.float32),
            np.concatenate(all_act).astype(np.float32),
            np.concatenate(all_w).astype(np.float32))


# eval helpers 
@torch.no_grad()
def eval_nll(policy: GaussianPolicy,
             obs: np.ndarray,
             actions: np.ndarray,
             weights: np.ndarray,
             batch: int = 16384) -> float:
    """Weighted mean NLL over a dataset, computed in chunks to bound memory."""
    total, wsum = 0.0, 0.0
    for s in range(0, len(obs), batch):
        o = torch.as_tensor(obs[s:s + batch])
        a = torch.as_tensor(actions[s:s + batch])
        w = torch.as_tensor(weights[s:s + batch])
        lp = policy.log_prob(o, a)
        total += -(w * lp).sum().item()
        wsum  += w.sum().item()
    return total / max(wsum, 1e-8)


def evaluate_policy(policy: GaussianPolicy,
                    env: Acrobot,
                    n_episodes: int,
                    episode_len: int,
                    seed: int,
                    render: bool = False) -> dict:
    """Roll out n_episodes deterministic (mean-action) episodes.

    Returns mean/std total cost per episode plus optional frames from ep 0.
    """
    returns: list[float] = []
    frames:  list[np.ndarray] = []
    renderer = mujoco.Renderer(env.model, height=480, width=640) if render else None

    for ep in range(n_episodes):
        # Acrobot.reset() uses np.random internally — re-seed per episode for
        # reproducibility across runs.
        np.random.seed(seed + ep)
        env.reset()

        ep_cost = 0.0
        for t in range(episode_len):
            obs_t = torch.as_tensor(env._get_obs(), dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                mu, _ = policy.forward(obs_t)
            action = mu.squeeze(0).numpy()
            _, cost, done, _ = env.step(action)
            ep_cost += cost

            if renderer is not None and ep == 0:
                renderer.update_scene(env.data)
                frames.append(renderer.render().copy())

            if done:
                break
        returns.append(ep_cost)

    arr = np.array(returns)
    return {
        "mean_cost": float(arr.mean()),
        "std_cost":  float(arr.std()),
        "frames":    frames,
    }


def evaluate_mppi(env: Acrobot,
                  controller: MPPI,
                  n_episodes: int,
                  episode_len: int,
                  seed: int) -> dict:
    """Same eval protocol as evaluate_policy, but stepping with MPPI.

    Uses the identical seed schedule (seed + ep) so episode i starts from the
    exact same initial condition as episode i of evaluate_policy — gives a
    per-condition apples-to-apples comparison.
    """
    returns: list[float] = []
    for ep in range(n_episodes):
        np.random.seed(seed + ep)
        env.reset()
        controller.reset()                 # clear MPPI's warm-start buffer

        ep_cost = 0.0
        for t in range(episode_len):
            state = env.get_state()
            action, _ = controller.plan_step(state)
            _, cost, done, _ = env.step(action)
            ep_cost += cost
            if done:
                break
        returns.append(ep_cost)

    arr = np.array(returns)
    return {"mean_cost": float(arr.mean()), "std_cost": float(arr.std())}


# main training loop
def main(cfg: BCConfig = BCConfig()) -> None:
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)

    train_keys, val_keys = split_conditions(cfg.demo_path, cfg.val_frac, rng)
    print(f"train conditions: {len(train_keys)}  val conditions: {len(val_keys)}")

    tr_obs, tr_act, tr_w = load_demos(cfg.demo_path, train_keys, cfg.H_keep)
    va_obs, va_act, va_w = load_demos(cfg.demo_path, val_keys,   cfg.H_keep)
    print(f"train samples: {len(tr_obs):,}   val samples: {len(va_obs):,}")

    policy = GaussianPolicy(cfg.obs_dim, cfg.act_dim, PolicyConfig())
    
    train_losses: list[float] = []
    val_losses:   list[float] = []
     
    cfg.ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")

    N = len(tr_obs)
    idx = np.arange(N)

    for epoch in range(cfg.num_epochs):
        rng.shuffle(idx)
        running, n_batches = 0.0, 0
        for start in range(0, N, cfg.batch_size):
            b = idx[start:start + cfg.batch_size]
            loss = policy.train_weighted(tr_obs[b], tr_act[b], tr_w[b])
            running += loss
            n_batches += 1

        train_losses.append(running / n_batches)
        val_losses.append(eval_nll(policy, va_obs, va_act, va_w))

        if val_losses[-1] < best_val:
            best_val = val_losses[-1]
            torch.save(policy.state_dict(), cfg.ckpt_path)
            print(f"  ↳ new best val={best_val:.4f}, checkpoint saved")

    # loss curve
    plt.figure()
    plt.plot(train_losses, label="train")
    plt.plot(val_losses,   label="val")
    plt.xlabel("epoch")
    plt.ylabel("weighted NLL")
    plt.legend()
    plt.tight_layout()
    plt.savefig(cfg.loss_plot, dpi=120)
    print(f"saved loss curve to {cfg.loss_plot}")

    # reload best-val checkpoint before env eval — otherwise we'd be evaluating
    # the in-memory final-epoch policy, which is the most overfit one.
    policy.load_state_dict(torch.load(cfg.ckpt_path))
    policy.eval()
    print(f"reloaded best-val checkpoint (val={best_val:.4f})")

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

    print(f"BC  policy:    {stats['mean_cost']:.2f} ± {stats['std_cost']:.2f}")
    print(f"MPPI baseline: {mppi_stats['mean_cost']:.2f} ± {mppi_stats['std_cost']:.2f}")
    print(f"gap (BC - MPPI): {stats['mean_cost'] - mppi_stats['mean_cost']:+.2f}")

    if stats["frames"]:
        mediapy.write_video(str(cfg.video_path), stats["frames"], fps=30)
        print(f"saved rollout video to {cfg.video_path}")

    env.close()


if __name__ == "__main__":
    main()
