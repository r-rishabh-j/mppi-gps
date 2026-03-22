import numpy as np 
from src.envs.gym_wrapper import GymEnv
from src.mppi.mppi import MPPI
from src.utils.config import MPPIConfig
import gymnasium as gym 
import mujoco

def main():
    env = GymEnv("HalfCheetah-v5")
    cfg = MPPIConfig(K=256, H=50, lam=1.0, noise_sigma=0.5)
    controller = MPPI(env, cfg)

    # renderer = mujoco.Renderer(env.model, height = 480, width = 640)
    # frames = []

    env.reset()
    state = env.get_state()

    total_steps = 1000 
    costs = []

    for t in range(total_steps):
        action, info = controller.plan_step(state)
        obs, cost, done, _ = env.step(action)
        state = env.get_state()
        costs.append(info["cost_mean"])
        
        # renderer.update_scene(env.data)
        # frames.append(renderer.render().copy())

        if t % 50 == 0:
            print(
                f"step={t:4d}  "
                f"cost_mean={info['cost_mean']:8.2f}  "
                f"cost_min={info['cost_min']:8.2f}  "
                f"n_eff={info['n_eff']:6.1f}  "
                f"lam={info['lam']:.3f}  "
                f"x_pos={env.data.qpos[0]:6.2f}"
        )

    env.close()

    # import mediapy
    # mediapy.write_video("cheetah_mppi.mp4", frames, fps=30)

if __name__ == "__main__":
    main()


