import argparse
import numpy as np
from src.envs.half_cheetah import HalfCheetah
from src.mppi.mppi import MPPI
from src.utils.config import MPPIConfig
from src.utils.seeding import add_seed_arg, seed_everything
import gymnasium as gym
import mujoco


def main():
    parser = argparse.ArgumentParser()
    add_seed_arg(parser, default=0)
    args = parser.parse_args()
    seed_everything(args.seed)

    env = HalfCheetah()
    cfg = MPPIConfig(K=1024, H=20, lam=0.025187910703501223, noise_sigma=0.35180969970545684, adaptive_lam=False)
    controller = MPPI(env, cfg)

    env.reset()
    state = env.get_state()

    total_steps = 1000
    costs = []

    renderer = mujoco.Renderer(env.model, height = 480, width = 640)
    frames = []

    for t in range(total_steps):
        action, info = controller.plan_step(state)
        
        obs, cost, done, _ = env.step(action)
        state = env.get_state()
        costs.append(info["cost_mean"])

        renderer.update_scene(env.data, camera="track")
        frames.append(renderer.render().copy())

        weights = np.sum(-controller._last_weights * np.log2(controller._last_weights))
        if t % 50 == 0:
            print(
                f"step={t:4d}  "
                f"cost_mean={info['cost_mean']}  "
                f"cost_min={info['cost_min']}  "
                f"weights = {weights}   "
                f"x_pos={env.data.qpos[0]}"
        )
    
    import mediapy
    mediapy.write_video("cheetah_mppi.mp4", frames, fps=30)

    env.close()


if __name__ == "__main__":
    main()


