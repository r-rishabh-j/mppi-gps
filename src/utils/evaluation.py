"""Shared evaluation helpers for policy and MPPI controllers.

Both GPS training (run_gps.py) and behaviour cloning (test_sl.py) need to
evaluate a trained policy and compare it against the MPPI baseline on
identical initial conditions.  This module provides those two functions
so that evaluation is consistent across all scripts.
"""

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
    """Roll out a trained policy for n_episodes and collect cost statistics.

    Each episode uses a deterministic seed (seed + ep) so that the initial
    conditions are reproducible and directly comparable with evaluate_mppi().

    The policy is used in mean-mode (no sampling noise): we take the mean
    of the Gaussian output as the action.

    Args:
        policy:      Trained GaussianPolicy network (should be in eval mode).
        env:         Environment instance.
        n_episodes:  Number of evaluation episodes.
        episode_len: Maximum steps per episode.
        seed:        Base seed — episode i uses seed + i.
        render:      If True, capture frames from episode 0 for video export.
    Returns:
        Dict with keys: mean_cost, std_cost, per_ep (list), frames (list).
    """
    returns: list[float] = []
    frames: list[np.ndarray] = []
    # Only create a renderer if the env has a MuJoCo model (not all BaseEnv do)
    renderer = None
    if render and hasattr(env, 'model'):
        renderer = mujoco.Renderer(env.model, height=480, width=640)

    for ep in range(n_episodes):
        # Seed before each episode so env.reset() produces a deterministic
        # initial condition that matches evaluate_mppi's episode i.
        np.random.seed(seed + ep)
        env.reset()

        ep_cost = 0.0
        for t in range(episode_len):
            # IMPORTANT: act_np (NOT policy.action) — act_np clips to the
            # env's actuator ctrlrange before returning. policy.action returns
            # the raw network mean which can sit well outside [low, high]
            # early in training. Writing those raw values into data.ctrl
            # used to trip MuJoCo's "huge value in CTRL" stability watchdog
            # on every eval step (in run_gps's per-iter eval especially)
            # and produced misleading eval costs that didn't reflect what
            # MPPI/DAgger see at execution time (both of which DO clip).
            action = policy.act_np(env._get_obs())
            _, cost, done, _ = env.step(action)
            ep_cost += cost

            # Capture video frames from the first episode only
            if renderer is not None and ep == 0:
                if camera is not None:
                    renderer.update_scene(env.data, camera=camera)
                else:
                    renderer.update_scene(env.data)
                frames.append(renderer.render().copy())

            # if done:
            #     break
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
    """Roll out MPPI for n_episodes and collect cost statistics.

    Uses the same seed schedule as evaluate_policy (seed + ep) so that
    episode i starts from the *exact* same initial condition in both
    evaluations, making the per-episode cost gap a fair comparison.

    Args:
        env:         Environment instance.
        controller:  MPPI controller (will be reset each episode).
        n_episodes:  Number of evaluation episodes.
        episode_len: Maximum steps per episode.
        seed:        Base seed — episode i uses seed + i.
    Returns:
        Dict with keys: mean_cost, std_cost, per_ep (list).
    """
    returns: list[float] = []
    for ep in range(n_episodes):
        np.random.seed(seed + ep)
        env.reset()
        controller.reset()  # clear MPPI's warm-start nominal U

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
    return {
        "mean_cost": float(arr.mean()),
        "std_cost": float(arr.std()),
        "per_ep": arr.tolist(),
    }
