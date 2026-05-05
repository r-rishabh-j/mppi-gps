import argparse
import os
import time

import mujoco
import mujoco.viewer
import numpy as np

from src.envs.adroit_relocate import AdroitRelocate
from src.mppi.mppi import MPPI
from src.utils.config import MPPIConfig
from src.utils.seeding import add_seed_arg, seed_everything

T = 100

# Cameras to tile in the recorded video (2x2 grid). Order = top-left,
# top-right, bottom-left, bottom-right.
_CAMERAS = ("vil_camera", "cam_iso", "cam_side", "cam_top")
# Per-camera tile size. Final frame is (2 * _TILE_H, 2 * _TILE_W, 3).
_TILE_H, _TILE_W = 240, 320

_OUTPUT_VIDEO = "adroit_relocate_mppi.mp4"


def resolve_camera_ids(model: mujoco.MjModel) -> list[int]:
    """Look up each camera by name; assert all resolve.

    Passing string names per-frame to ``update_scene`` was occasionally
    falling through silently (all four tiles ending up identical).
    Resolving once and passing int IDs makes that failure mode loud.
    """
    ids = []
    for name in _CAMERAS:
        cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        if cid < 0:
            raise RuntimeError(
                f"camera {name!r} not found in model — XML out of sync with "
                f"_CAMERAS in {__file__}"
            )
        ids.append(cid)
    return ids


def render_tiled(
    renderer: mujoco.Renderer,
    data: mujoco.MjData,
    camera_ids: list[int],
) -> np.ndarray:
    """Render every camera in _CAMERAS and stitch into a 2x2 grid."""
    imgs = []
    for cid in camera_ids:
        renderer.update_scene(data, camera=cid)
        imgs.append(renderer.render().copy())
    top = np.hstack([imgs[0], imgs[1]])
    bot = np.hstack([imgs[2], imgs[3]])
    return np.vstack([top, bot])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true",
                        help="interactive viewer instead of recording")
    add_seed_arg(parser, default=400)
    args = parser.parse_args()
    seed_everything(args.seed)

    env = AdroitRelocate()
    cfg = MPPIConfig.load("adroit_relocate")
    controller = MPPI(env, cfg)

    dt = env.model.opt.timestep * env._frame_skip

    viewer = None
    renderer = None
    camera_ids: list[int] = []
    frames = []

    if args.live:
        viewer = mujoco.viewer.launch_passive(env.model, env.data)
    else:
        # One renderer reused across cameras; size matches a single tile.
        renderer = mujoco.Renderer(env.model, height=_TILE_H, width=_TILE_W)
        camera_ids = resolve_camera_ids(env.model)
        print(f"recording cameras {list(zip(_CAMERAS, camera_ids))} -> "
              f"frame shape ({2*_TILE_H}, {2*_TILE_W}, 3)")

    for ep in range(1):
        env.reset()
        controller.reset()
        state = env.get_state()
        total_cost = 0.0
        for t in range(T):
            action, info = controller.plan_step(state)
            _, cost, _, step_info = env.step(action)
            total_cost += cost
            state = env.get_state()

            if viewer is not None:
                viewer.sync()
                # time.sleep(dt)
            elif renderer is not None:
                frames.append(render_tiled(renderer, env.data, camera_ids))

            if t % 20 == 0:
                print(f"ep={ep} step={t:4d}  cost_min={info['cost_min']:.2f}  "
                      f"goal_dist={step_info.get('goal_distance', float('nan')):.3f}  "
                      f"success={step_info.get('success', False)}")

        print(f"ep={ep} total_cost={total_cost:.2f}")

    if viewer is not None:
        viewer.close()
    elif frames:
        import mediapy
        out_path = os.path.abspath(_OUTPUT_VIDEO)
        # Overwrite any stale file rather than letting the OS / player
        # cache trick the user into thinking nothing changed.
        if os.path.exists(out_path):
            os.remove(out_path)
        print(f"writing {len(frames)} frames of shape {frames[0].shape} -> {out_path}")
        mediapy.write_video(out_path, frames, fps=int(1 / dt)//4)
        print(f"saved video: {out_path}")

    env.close()


if __name__ == "__main__":
    main()
