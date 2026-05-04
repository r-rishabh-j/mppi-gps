import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

from src.envs.adroit_pen import AdroitPen
from src.mppi.mppi import MPPI
from src.utils.config import MPPIConfig
from src.utils.seeding import add_seed_arg, seed_everything

T = 200


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true",
                        help="interactive viewer instead of recording")
    add_seed_arg(parser, default=400)
    args = parser.parse_args()
    seed_everything(args.seed)

    env = AdroitPen()
    cfg = MPPIConfig.load("adroit_pen")
    controller = MPPI(env, cfg)

    dt = env.model.opt.timestep * env._frame_skip

    viewer = None
    renderer = None
    frames = []

    if args.live:
        viewer = mujoco.viewer.launch_passive(env.model, env.data)
    else:
        renderer = mujoco.Renderer(env.model, height=480, width=640)


    for _ in range(10):
        env.reset()
        controller.reset()
        state = env.get_state()
        total_cost = 0.0
        done_counter = 0
        for t in range(T):
            action, info = controller.plan_step(state)
            _, cost, done, step_info = env.step(action)
            total_cost += cost
            state = env.get_state()

            if viewer is not None:
                viewer.sync()
                time.sleep(dt)
            elif renderer is not None:
                renderer.update_scene(env.data, camera="vil_camera")
                frames.append(renderer.render().copy())

            if t % 20 == 0:
                print(f"step={t:4d}  cost_min={info['cost_min']:.2f}  "
                    f"success={step_info.get('success', False)}")

            if done:
                done_counter += 1
                if done_counter >= 2:
                    print(f"  -> success at step {t}")
                    break

    print(f"total_cost={total_cost:.2f}")

    if viewer is not None:
        viewer.close()
    elif frames:
        import mediapy
        mediapy.write_video("adroit_pen_mppi.mp4", frames, fps=int(1 / dt))
        print("saved video")

    env.close()


if __name__ == "__main__":
    main()
