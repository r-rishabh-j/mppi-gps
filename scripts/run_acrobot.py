import numpy as np 
from src.envs.acrobot import Acrobot
from src.mppi.mppi import MPPI
from src.utils.config import MPPIConfig
import mujoco

def main():
    env = Acrobot()
    cfg = MPPIConfig(K=2048, H=100, lam=1.0, noise_sigma=0.5, adaptive_lam=False)
    controller = MPPI(env, cfg)

    env.reset()
    state = env.get_state()

    renderer = mujoco.Renderer(env.model, height=480, width=640)
    frames = []

    for t in range(2000):
        action, info = controller.plan_step(state)
        obs, cost, done, _ = env.step(action)
        state = env.get_state()

        renderer.update_scene(env.data)
        frames.append(renderer.render().copy())

        if t % 100 == 0:
            print(f"step={t:4d}  cost_min={info['cost_min']:.2f}  "
                f"shoulder={env.data.qpos[0]:.2f}  elbow={env.data.qpos[1]:.2f}")

    import mediapy
    mediapy.write_video("acrobot_mppi.mp4", frames, fps=30)
    env.close()

if __name__ == "__main__":
    main()