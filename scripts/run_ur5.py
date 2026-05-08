"""Standalone MPPI runner for the UR5 push task.

Mirrors the structure of ``run_hopper.py`` / ``run_adroit_pen.py``: reset
the env, replan with MPPI each step, render to either an interactive
viewer (``--live``) or a recorded MP4 (default).

Optional policy prior — pass ``--policy-ckpt`` and ``--alpha > 0`` to
visualise how a learned-policy-augmented MPPI rollout differs from
vanilla MPPI (auto-detects Gaussian vs Deterministic from the
checkpoint).
"""

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

from src.envs.ur5_push import UR5Push
from src.mppi.mppi import MPPI
from src.utils.config import MPPIConfig
from src.utils.policy_prior_loader import (
    add_policy_prior_args, resolve_policy_prior,
)
from src.utils.seeding import add_seed_arg, seed_everything

T = 1000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true",
                        help="interactive viewer instead of recording")
    add_seed_arg(parser, default=0)
    add_policy_prior_args(parser)
    args = parser.parse_args()
    seed_everything(args.seed)

    env = UR5Push()
    cfg = MPPIConfig.load("ur5_push")
    controller = MPPI(env, cfg)
    prior_fn = resolve_policy_prior(args, env)

    dt = env.model.opt.timestep * env._frame_skip

    viewer = None
    renderer = None
    frames = []

    if args.live:
        viewer = mujoco.viewer.launch_passive(env.model, env.data)
    else:
        renderer = mujoco.Renderer(env.model, height=480, width=640)

    env.reset()
    controller.reset()
    state = env.get_state()
    total_cost = 0.0

    for t in range(T):
        action, info = controller.plan_step(state, prior=prior_fn)
        _, cost, done, step_info = env.step(action)
        total_cost += cost
        state = env.get_state()

        if viewer is not None:
            viewer.sync()
            time.sleep(dt)
        elif renderer is not None:
            renderer.update_scene(env.data, camera="birdseye_tilted_cam")
            frames.append(renderer.render().copy())

        if t % 20 == 0:
            print(
                f"step={t:4d}  cost_min={info['cost_min']:.2f}  "
                f"ee→tape={step_info['ee_to_tape']:.3f}  "
                f"tape→target={step_info['tape_to_target']:.3f}  "
                f"success={step_info['success']}"
            )

    print(f"total_cost={total_cost:.2f}  "
          f"final tape→target={step_info['tape_to_target']:.3f}")

    if viewer is not None:
        viewer.close()
    elif frames:
        import mediapy
        mediapy.write_video("ur5_push_mppi.mp4", frames, fps=int(1 / dt))
        print("saved video: ur5_push_mppi.mp4")

    env.close()


if __name__ == "__main__":
    main()
