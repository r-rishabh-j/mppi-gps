import argparse
import mujoco
import numpy as np

from src.envs.point_mass import PointMass
from src.mppi.mppi import MPPI
from src.utils.config import MPPIConfig
from src.utils.seeding import add_seed_arg, seed_everything


def main():
    parser = argparse.ArgumentParser()
    add_seed_arg(parser, default=0)
    args = parser.parse_args()
    seed_everything(args.seed)

    env = PointMass()
    cfg = MPPIConfig(
        K=256,
        H=64,
        lam=0.01,
        noise_sigma=0.5,
        adaptive_lam=False,
    )
    controller = MPPI(env, cfg)

    env.reset()
    state = env.get_state()

    renderer = mujoco.Renderer(env.model, height=480, width=640)
    frames = []

    for vid in range(5):
        env.reset()
        for t in range(500):
            action, info = controller.plan_step(state)
            _, cost, _, _ = env.step(action)
            state = env.get_state()

            renderer.update_scene(env.data)
            frames.append(renderer.render().copy())

            if t % 20 == 0:
                pos = env.data.qpos.copy()
                vel = env.data.qvel.copy()
                print(
                    f"step={t:4d}  cost_min={info['cost_min']:.3f}  "
                    f"pos=({pos[0]:.3f}, {pos[1]:.3f})  "
                    f"vel=({vel[0]:.3f}, {vel[1]:.3f})"
                )

        final_pos = env.data.qpos.copy()
        final_vel = env.data.qvel.copy()
        print(
            f"final_pos=({final_pos[0]:.4f}, {final_pos[1]:.4f})  "
            f"final_vel=({final_vel[0]:.4f}, {final_vel[1]:.4f})"
        )

    import mediapy
    mediapy.write_video("point_mass_mppi.mp4", frames, fps=30)
    env.close()


if __name__ == "__main__":
    main()
