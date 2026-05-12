"""Shared eval helpers (policy + MPPI on identical seeds for comparison)."""

import numpy as np
import torch
import mujoco

from src.policy.gaussian_policy import GaussianPolicy
from src.envs.base import BaseEnv
from src.mppi.mppi import MPPI


def evaluate_policy(
    policy: GaussianPolicy,
    env: BaseEnv,
    n_episodes: int,
    episode_len: int,
    seed: int,
    render: bool = False,
    camera: str | int | None = None,
) -> dict:
    """Mean-mode policy rollout, ``n_episodes`` × ``episode_len``.

    Episode i uses ``seed + i`` (matches ``evaluate_mppi`` for fair compare).
    Returns dict: mean_cost, std_cost, per_ep, frames (ep 0 only when render).
    """
    returns: list[float] = []
    frames: list[np.ndarray] = []
    renderer = None
    if render and hasattr(env, 'model'):
        renderer = mujoco.Renderer(env.model, height=480, width=640)

    for ep in range(n_episodes):
        np.random.seed(seed + ep)
        env.reset()

        ep_cost = 0.0
        for t in range(episode_len):
            # act_np (not policy.action) clips to env ctrlrange — raw means
            # can otherwise trip MuJoCo's "huge CTRL" watchdog early on.
            obs = env._get_obs()
            action = policy.act_np(obs)
            if np.isnan(action).any():
                obs_nan = bool(np.isnan(obs).any())
                bad_params = [
                    k for k, v in policy.state_dict().items()
                    if not torch.isfinite(v).all()
                ]
                norm_bad = False
                if getattr(policy, "normalizer", None) is not None:
                    n = policy.normalizer
                    norm_bad = (
                        not torch.isfinite(n.mean).all()
                        or not torch.isfinite(n.var).all()
                    )
                print(f"[NaN action] ep={ep} t={t}  "
                      f"obs_has_nan={obs_nan}  "
                      f"normalizer_bad={norm_bad}  "
                      f"params_with_nan={bad_params[:3]}{'...' if len(bad_params) > 3 else ''}  "
                      f"(total {len(bad_params)} bad params)")
                if t == 0 and ep == 0:
                    raise RuntimeError(
                        "NaN action on first eval step. See diagnostic above."
                    )
            _, cost, done, _ = env.step(action)
            ep_cost += cost

            if renderer is not None and ep == 0:
                if camera is not None:
                    renderer.update_scene(env.data, camera=camera)
                else:
                    renderer.update_scene(env.data)
                frames.append(renderer.render().copy())

            # NB: we intentionally don't break on done — see evaluate_mppi.
        returns.append(ep_cost)

    arr = np.array(returns)
    return {
        "mean_cost": float(arr.mean()),
        "std_cost": float(arr.std()),
        "per_ep": arr.tolist(),
        "frames": frames,
    }


def evaluate_mppi(
    env: BaseEnv,
    controller: MPPI,
    n_episodes: int,
    episode_len: int,
    seed: int,
) -> dict:
    """MPPI baseline rollout (same seed schedule as ``evaluate_policy``)."""
    returns: list[float] = []
    for ep in range(n_episodes):
        np.random.seed(seed + ep)
        env.reset()
        controller.reset()

        ep_cost = 0.0
        for t in range(episode_len):
            state = env.get_state()
            action, _ = controller.plan_step(state)
            _, cost, done, _ = env.step(action)
            ep_cost += cost
            # We don't break on done — evaluate_policy doesn't either, and
            # asymmetric truncation would make the policy↔MPPI gap unfair.
        returns.append(ep_cost)

    arr = np.array(returns)
    return {
        "mean_cost": float(arr.mean()),
        "std_cost": float(arr.std()),
        "per_ep": arr.tolist(),
    }
