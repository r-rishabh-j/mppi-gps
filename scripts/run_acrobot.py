import mujoco
import numpy as np

from src.envs.acrobot import Acrobot
from src.mppi.mppi import MPPI
from src.utils.config import MPPIConfig

def main():
    env = Acrobot()
    cfg = MPPIConfig.load("acrobot")
    controller = MPPI(env, cfg)

    renderer = mujoco.Renderer(env.model, height=480, width=640)
    render = True
    frames = []

    for vid in range(2):
        env.reset()
        controller.reset()
        print(env.data.qpos)
        state = env.get_state()
        total_cost = 0.0 
        for t in range(2000):
            action, info = controller.plan_step(state)
            obs, cost, done, _ = env.step(action)
            total_cost += cost

            state = env.get_state()

            if render:
                renderer.update_scene(env.data)
                frames.append(renderer.render().copy())

            if t % 100 == 0:
                tip_z = env.data.site("tip").xpos[2]
                print(
                    f"step={t:4d}  cost_min={info['cost_min']:.2f}  "
                    # f"n_eff={info['n_eff']:.1f}  lam={info['lam']:.3f}  "
                    f"shoulder={env.data.qpos[0]:.2f}  elbow={env.data.qpos[1]:.2f}  "
                    f"tip_z={tip_z:.2f}"
                    f"total cost={total_cost}"
                )
        print(total_cost)

    if render and frames:
        import mediapy
        mediapy.write_video("acrobot_mppi.mp4", frames, fps=30 * 5)
        print(f"saved video")
    env.close()


if __name__ == "__main__":
    main()
