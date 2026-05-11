import argparse
import mujoco
import numpy as np

from src.envs.point_mass import PointMass
from src.mppi.mppi import MPPI
from src.utils.config import MPPIConfig
from src.utils.policy_prior_loader import (
    add_policy_prior_args, resolve_policy_prior,
)
from src.utils.seeding import add_seed_arg, seed_everything


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--episodes", type=int, default=5,
        help="number of episodes to render (each with a fresh random goal)",
    )
    parser.add_argument(
        "--steps", type=int, default=800,
        help="steps per episode (env dt=0.02 → 300 steps = 6s)",
    )
    parser.add_argument(
        "--no-video", action="store_true",
        help="skip writing point_mass_mppi.mp4 — useful for quick smoke tests",
    )
    add_seed_arg(parser, default=0)
    add_policy_prior_args(parser)
    args = parser.parse_args()
    seed_everything(args.seed)

    env = PointMass()
    cfg = MPPIConfig.load("point_mass")
    controller = MPPI(env, cfg)
    prior_fn = resolve_policy_prior(args, env)

    renderer = mujoco.Renderer(env.model, height=480, width=640) if not args.no_video else None
    frames = []

    successes = 0
    for vid in range(args.episodes):
        env.reset()                    # new random goal each episode
        controller.reset()
        state = env.get_state()
        gx, gy = env._goal
        print(f"--- episode {vid}  goal=({gx:+.3f}, {gy:+.3f}) ---")

        for t in range(args.steps):
            action, info = controller.plan_step(state, prior=prior_fn)
            _, cost, _, _ = env.step(action)
            state = env.get_state()

            if renderer is not None:
                renderer.update_scene(env.data)
                frames.append(renderer.render().copy())

            if t % 50 == 0:
                pos = env.data.qpos.copy()
                vel = env.data.qvel.copy()
                print(
                    f"  step={t:4d}  cost_min={info['cost_min']:.3f}  "
                    f"pos=({pos[0]:+.3f}, {pos[1]:+.3f})  "
                    f"|v|={np.linalg.norm(vel):.3f}"
                )

        metrics = env.task_metrics()
        successes += int(metrics["success"])
        print(
            f"  final  dist={metrics['goal_dist']:.4f}  |v|={metrics['qvel_norm']:.4f}  "
            f"success={metrics['success']}"
        )

    print(f"\nsuccess rate: {successes}/{args.episodes}")

    if frames:
        import mediapy
        # fps = 1 / (env.dt) so the mp4 plays at sim wall-clock speed. With
        # `fps=30` (the legacy hardcoded value) and the point_mass running
        # at 50 Hz (timestep=0.02s, frame_skip=1), the video used to play at
        # 30/50 = 0.6× real-time — matching `eval_checkpoint` and the other
        # runners (`run_hopper`, `run_ur5`, `run_adroit_pen`) avoids that.
        dt = env.model.opt.timestep * env._frame_skip
        fps = int(round(1.0 / dt))
        mediapy.write_video("point_mass_mppi.mp4", frames, fps=fps)
        print(f"saved video → point_mass_mppi.mp4 (fps={fps})")
    env.close()


if __name__ == "__main__":
    main()
