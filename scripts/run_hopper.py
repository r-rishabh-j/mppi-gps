import argparse
import mujoco
import mujoco.viewer
import numpy as np
import time
import torch

from src.envs.hopper import Hopper
from src.mppi.mppi import MPPI
from src.utils.config import MPPIConfig
from src.utils.policy_prior_loader import (
    add_policy_prior_args, resolve_policy_prior,
)
from src.utils.seeding import add_seed_arg, seed_everything

T = 1000

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="interactive viewer instead of recording")
    parser.add_argument("--cost-mode", default="v2", choices=["v1", "v2"],
                        help="hopper running cost: original v1 or dm_control-style v2")
    parser.add_argument("--warp", action="store_true",
                        help="Use mujoco_warp GPU batch rollout for MPPI. "
                             "Requires `uv pip install warp-lang mujoco-warp` "
                             "and an NVIDIA GPU with CUDA. nworld is pinned "
                             "to MPPIConfig.K — single-condition standalone, "
                             "no batched-MPPI in this script.")
    add_seed_arg(parser, default=42)
    add_policy_prior_args(parser)
    args = parser.parse_args()
    seed_everything(args.seed)

    # Load MPPI cfg before constructing env: the warp env needs `nworld=cfg.K`
    # and `nworld` is fixed for the env's lifetime.
    cfg = MPPIConfig.load("hopper")
    if args.warp:
        from src.envs.hopper_warp import HopperWarp
        env = HopperWarp(nworld=cfg.K, cost_mode=args.cost_mode)
    else:
        env = Hopper(cost_mode=args.cost_mode)
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
        _, cost, done, _ = env.step(action)
        total_cost += cost
        state = env.get_state()

        if done:
            print('done')

        if viewer is not None:
            viewer.sync()
            time.sleep(dt)
        elif renderer is not None:
            renderer.update_scene(env.data, camera="track")
            frames.append(renderer.render().copy())

        if t % 100 == 0:
            print(f"step={t:4d}  cost_min={info['cost_min']:.2f}")

    print(f"total_cost={total_cost:.2f}")

    if viewer is not None:
        viewer.close()
    elif frames:
        import mediapy
        mediapy.write_video("hopper_mppi.mp4", frames, fps=int(1 / dt))
        print("saved video")

    env.close()


if __name__ == "__main__":
    main()
