import argparse
import mujoco
import mujoco.viewer
import numpy as np
import time
import torch

from src.envs.hopper import Hopper
from src.mppi.mppi import MPPI
from src.utils.config import MPPIConfig

np.random.seed(42)
torch.manual_seed(42)

T = 2000

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="interactive viewer instead of recording")
    parser.add_argument("--rollout-backend", default="cpu", choices=["cpu", "warp"])
    args = parser.parse_args()

    env = Hopper(backend=args.rollout_backend)
    cfg = MPPIConfig.load("hopper")
    controller = MPPI(env, cfg)

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
        action, info = controller.plan_step(state)
        _, cost, done, _ = env.step(action)
        total_cost += cost
        state = env.get_state()

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
