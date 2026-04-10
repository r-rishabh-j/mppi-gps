"""Visualize MPPI rollout trajectories in the MuJoCo renderer.

At each planning step, the K sampled rollouts are drawn as line segments
in the 3D scene so you can see where the controller is exploring.
"""

import cv2
import mujoco
import numpy as np

from src.envs.acrobot import Acrobot
from src.mppi.mppi import MPPI
from src.utils.config import MPPIConfig


def cost_to_rgba(cost, cost_min, cost_max, alpha=0.4):
    """Map a cost value to green (low) -> red (high)."""
    if cost_max - cost_min < 1e-8:
        t = 0.5
    else:
        t = (cost - cost_min) / (cost_max - cost_min)
    t = np.clip(t, 0.0, 1.0)
    # green -> red
    r = t
    g = 1.0 - t
    return np.array([r, g, 0.0, alpha], dtype=np.float32)


def add_rollout_lines(scene, tip_positions, costs,
                      max_rollouts=30, step_skip=4):
    """Draw rollout trajectories as line segments in the MuJoCo scene.

    Lines are colored green (low cost) to red (high cost).

    Args:
        scene: MjvScene to add geoms to
        tip_positions: (K, H, 3) array of 3D positions along each rollout
        costs: (K,) total cost per rollout
        max_rollouts: max number of rollouts to draw (subsampled randomly)
        step_skip: draw every Nth horizon step to reduce geom count
    """
    K, H, _ = tip_positions.shape

    # subsample rollouts if needed
    if K > max_rollouts:
        indices = np.random.choice(K, max_rollouts, replace=False)
    else:
        indices = np.arange(K)

    cost_min = costs[indices].min()
    cost_max = costs[indices].max()

    for k in indices:
        rgba = cost_to_rgba(costs[k], cost_min, cost_max)
        for h in range(0, H - step_skip, step_skip):
            if scene.ngeom >= scene.maxgeom:
                return
            p0 = tip_positions[k, h].reshape(3, 1).astype(np.float64)
            p1 = tip_positions[k, h + step_skip].reshape(3, 1).astype(np.float64)
            mujoco.mjv_connector(
                scene.geoms[scene.ngeom],
                mujoco.mjtGeom.mjGEOM_LINE,
                0.002,
                p0, p1,
            )
            scene.geoms[scene.ngeom].rgba = rgba
            scene.ngeom += 1


def main():
    env = Acrobot()
    cfg = MPPIConfig.load("acrobot")
    controller = MPPI(env, cfg)

    max_rollouts = 30
    step_skip = 4

    renderer = mujoco.Renderer(env.model, height=480, width=640)
    frames = []

    env.reset()
    controller.reset()
    state = env.get_state()

    n_steps = 1000
    for t in range(n_steps):
        action, info = controller.plan_step(state)

        # get rollout data
        rollout_data = controller.get_rollout_data()
        sensordata = rollout_data['sensordata']  # (K, H, nsensor)
        costs = rollout_data['costs']             # (K,)
        tip_positions = sensordata[:, :, :3]      # (K, H, 3)

        # render scene with cost-colored rollout overlay
        renderer.update_scene(env.data)
        add_rollout_lines(
            renderer.scene, tip_positions, costs,
            max_rollouts=max_rollouts,
            step_skip=step_skip,
        )
        frame = renderer.render().copy()

        # text overlay with cost info
        tip_z = env.data.site("tip").xpos[2]
        lines = [
            f"t={t:4d}",
            f"cost_min={info['cost_min']:.2f}  cost_mean={info['cost_mean']:.2f}",
            f"tip_z={tip_z:.2f}",
        ]
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (10, 25 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        frames.append(frame)

        # step env
        obs, cost, done, _ = env.step(action)
        state = env.get_state()

        if t % 100 == 0:
            print(
                f"step={t:4d}  cost_min={info['cost_min']:.2f}  "
                f"shoulder={env.data.qpos[0]:.2f}  elbow={env.data.qpos[1]:.2f}  "
                f"tip_z={tip_z:.2f}"
            )

    if frames:
        import mediapy
        mediapy.write_video("acrobot_rollouts.mp4", frames, fps=5 * 30)
        print("saved acrobot_rollouts.mp4")
    env.close()


if __name__ == "__main__":
    main()
