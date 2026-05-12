"""Diagnostic: print ball/palm/fingertip positions on CPU vs warp.

Run as ``uv run python -m scripts.check_warp_sensors``.
"""
from __future__ import annotations

import numpy as np
import mujoco

from src.envs.adroit_relocate import AdroitRelocate
from src.envs.adroit_relocate_warp import AdroitRelocateWarp


def main() -> None:
    K = 16
    H = 4

    np.random.seed(0)
    cpu_env = AdroitRelocate()
    warp_env = AdroitRelocateWarp(nworld=K)

    # Reset both to identical initial state.
    cpu_env.reset()
    warp_env.reset()
    np.copyto(warp_env.data.qpos, cpu_env.data.qpos)
    np.copyto(warp_env.data.qvel, cpu_env.data.qvel)
    np.copyto(warp_env.data.mocap_pos, cpu_env.data.mocap_pos)
    np.copyto(warp_env.data.mocap_quat, cpu_env.data.mocap_quat)

    # Forward kinematics on CPU so we can read the *initial* xpos.
    mujoco.mj_forward(cpu_env.model, cpu_env.data)

    # Find the Object body's index — its xpos is what framepos(obj) reports.
    obj_body_id = mujoco.mj_name2id(
        cpu_env.model, mujoco.mjtObj.mjOBJ_BODY, "Object",
    )
    palm_site_id = cpu_env._palm_site_id

    print(f"Object body id: {obj_body_id}")
    print(f"Palm site id  : {palm_site_id}")
    print()
    print(f"=== INITIAL state (before any rollout step) ===")
    print(f"CPU  data.qpos[free joint slice 0:7]: {cpu_env.data.qpos[:7]}")
    print(f"CPU  data.xpos[Object]              : {cpu_env.data.xpos[obj_body_id]}")
    print(f"CPU  data.site_xpos[palm]           : {cpu_env.data.site_xpos[palm_site_id]}")
    print()

    state = cpu_env.get_state()
    U = np.zeros((K, H, cpu_env.action_dim), dtype=np.float64)

    states_cpu, costs_cpu, sensors_cpu = cpu_env.batch_rollout(state, U)
    states_warp, costs_warp, sensors_warp = warp_env.batch_rollout(state, U)

    # Object position via the obj_pos framepos sensor (cheaper than xpos lookup).
    obj_slice = cpu_env._obj_pos_slice
    palm_slice = cpu_env._palm_pos_slice

    def show(label, sensors, states, h):
        s = sensors[0, h]
        st = states[0, h]
        # qpos slice in FULLPHYSICS layout: [time(1), qpos(nq), qvel(nv)]
        nq = cpu_env.model.nq
        qpos = st[1 : 1 + nq]
        # Object's free joint pos lives at the END of qpos (last 7 entries
        # in Adroit's relocate model; the free joint is the last joint).
        # Print the last 7 qpos entries to see the ball's free-joint state.
        print(f"  [{label}] obj_pos sensor       = {s[obj_slice]}")
        print(f"  [{label}] palm_pos sensor      = {s[palm_slice]}")
        print(f"  [{label}] qpos[-7:] (free jnt) = {qpos[-7:]}")
        print(f"  [{label}] cost(this rollout)   = {costs_cpu[0] if 'CPU' in label else costs_warp[0]:.4f}")

    print(f"=== AFTER 1 control step (h=0, frame_skip={cpu_env._frame_skip}) ===")
    show("CPU ", sensors_cpu, states_cpu, 0)
    print()
    show("WARP", sensors_warp, states_warp, 0)
    print()

    print(f"=== AFTER {H} control steps (h={H-1}) ===")
    show("CPU ", sensors_cpu, states_cpu, H - 1)
    print()
    show("WARP", sensors_warp, states_warp, H - 1)
    print()

    # Step-by-step ball z over rollout to see if mjw integrates the free joint.
    nq = cpu_env.model.nq
    print(f"=== Ball z over rollout (qpos[-5] of last 7 free joint = z) ===")
    print(f"step    cpu_z    warp_z")
    for h in range(H):
        cpu_z  = states_cpu [0, h, 1 + nq - 5]
        warp_z = states_warp[0, h, 1 + nq - 5]
        print(f"  {h}   {cpu_z:.4f}   {warp_z:.4f}")


if __name__ == "__main__":
    main()
