"""UR5 push task — vendored from `jaiselsingh1/ur5_research`.

Mirrors upstream's **MPPI** path (``mppi_ur5.py`` + ``ur5_env.py``):
6-D raw joint-velocity actions fed straight into ``mujoco.rollout`` for
the C-side parallel batch — no Python IK in the hot path. The
Cartesian-controller / delta-pose action space exists upstream too but
is only used in ``ur5_push_env.py.step()`` for PPO training, not MPPI.

To keep the EE pointing down without paying per-step IK, ``running_cost``
adds an **orientation penalty** computed from the ee_finger framequat
sensor. The cylinder z-axis in world coords has a closed-form expression
in the quaternion: ``cyl_axis_z = 1 − 2(qx² + qy²)``; perfectly vertical
gives ``cyl_axis_z = −1`` (and the dot with world ``+z`` is the negative
of that, which we want minimised). MPPI naturally finds trajectories
that keep the EE down because tilted rollouts are penalised.

State / observation layout
--------------------------
* ``model.nq = 13``  — 6 UR5 joint angles + 7 tape free-joint.
* ``model.nv = 12``  — 6 UR5 joint velocities + 6 tape (lin + ang).
* ``model.nu = 6``   — joint velocity actuators (ctrlrange ±3.15 / ±3.2).
* Policy obs (34-D):
  ``[qpos (13), qvel (12), ee−tape (3), tape−target (3), ee−target (3)]``

Cost
----
``running_cost``  = ``w_reach · ‖ee − tape‖²  +  w_orient · (1 + cyl_z)``
``terminal_cost`` = ``w_term  · ‖tape − target‖``

Where ``cyl_z = 1 − 2(qx² + qy²)`` is the world-z component of the
fingertip cylinder axis. Perfect downward: ``cyl_z = −1`` ⇒ orientation
cost = 0. Horizontal: ``cyl_z ≈ 0`` ⇒ orientation cost = 1. Upside-down:
``cyl_z = +1`` ⇒ orientation cost = 2. Linear in ``cyl_z`` is a smooth
convex penalty.
"""

from pathlib import Path

import mujoco
import numpy as np
from jaxtyping import Array, Float
from numpy import ndarray

from src.envs.mujoco_env import MuJoCoEnv

_XML = str(Path(__file__).resolve().parents[2] / "assets" / "ur5" / "scene.xml")

# Default initial UR5 joint configuration. Pose chosen so the fingertip
# axis points straight down (world -z) and the fingertip sits ~0.18m
# above the tape — i.e. EE is poised to descend onto the tape and push
# laterally. Verified via mj_forward + body_xmat: cyl axis ≈ (0, 0, -1).
_INIT_QPOS_UR5 = np.array(
    [0.0, -np.pi / 3, 2 * np.pi / 3, -np.pi / 3 - np.pi / 2, -np.pi / 2, 0.0],
    dtype=np.float64,
)

# Target sampling box on the table.
_TARGET_X_RANGE = (0.35, 0.65)
_TARGET_Y_RANGE = (-0.4, 0.4)
_TARGET_Z = -0.1175

# Cost weights.
_W_REACH = 10.0           # running, ‖ee − tape‖² (upstream's term)
_W_ORIENT = 100.0         # running, orientation penalty — large because
                          # the unitless penalty maxes at 2; MPPI needs a
                          # strong nudge to suppress wrist tilt during
                          # sample exploration. Bump if EE still tilts.
_W_TERMINAL = 50.0        # terminal, ‖tape − target‖ — bumped from upstream's 10
                          # because our default H=32 makes summed running ~3×
                          # larger than upstream's H=10, so terminal needs to
                          # remain influential after the running reach decays.

_SUCCESS_DIST = 0.05


class UR5Push(MuJoCoEnv):
    def __init__(self, frame_skip: int = 5, **kwargs) -> None:
        super().__init__(model_path=_XML, frame_skip=frame_skip, **kwargs)
        assert self.model.nu == 6, f"expected 6 actuators (UR5 joints), got {self.model.nu}"
        assert self.model.nmocap == 1, "scene.xml should expose target_marker as mocap"

        self._ee_pos_slice = self._sensor_slice("ee_pos")
        self._tape_pos_slice = self._sensor_slice("tape_pos")
        self._ee_quat_slice = self._sensor_slice("ee_quat")

        self._ee_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_finger_tip"
        )

        self._target_pos = np.array([0.65, 0.2, _TARGET_Z], dtype=np.float64)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _sensor_slice(self, name: str) -> slice:
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        adr = int(self.model.sensor_adr[sid])
        dim = int(self.model.sensor_dim[sid])
        return slice(adr, adr + dim)

    def _sample_target(self) -> np.ndarray:
        return np.array([
            np.random.uniform(*_TARGET_X_RANGE),
            np.random.uniform(*_TARGET_Y_RANGE),
            _TARGET_Z,
        ], dtype=np.float64)

    # ------------------------------------------------------------------
    # Reset / step (inherits parent's batch_rollout — fast C path)
    # ------------------------------------------------------------------

    def reset(self, state: np.ndarray | None = None) -> np.ndarray:
        if state is not None:
            return super().reset(state=state)

        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:6] = _INIT_QPOS_UR5
        self._target_pos = self._sample_target()
        self.data.mocap_pos[0] = self._target_pos
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        action = np.clip(action, self.action_bounds[0], self.action_bounds[1])
        obs, cost_arr, done, info = super().step(action)

        ee = self.data.site_xpos[self._ee_site_id]
        tape = self.data.body("tape_roll").xpos
        info["ee_to_tape"] = float(np.linalg.norm(ee - tape))
        info["tape_to_target"] = float(np.linalg.norm(tape - self._target_pos))
        info["success"] = bool(info["tape_to_target"] < _SUCCESS_DIST)
        return obs, cost_arr, done, info

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _get_obs(self) -> Float[ndarray, "34"]:
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        ee = self.data.site_xpos[self._ee_site_id]
        tape = self.data.body("tape_roll").xpos
        target = self._target_pos
        return np.concatenate([
            qpos, qvel, ee - tape, tape - target, ee - target,
        ]).astype(np.float64)

    def state_to_obs(
        self,
        states: np.ndarray,
        sensordata: np.ndarray | None = None,
    ) -> np.ndarray:
        qpos = self.state_qpos(states)
        qvel = self.state_qvel(states)
        target = np.broadcast_to(self._target_pos, qpos.shape[:-1] + (3,))

        if sensordata is not None:
            ee = sensordata[..., self._ee_pos_slice]
            tape = sensordata[..., self._tape_pos_slice]
        else:
            flat = qpos.reshape(-1, qpos.shape[-1])
            ee_out = np.zeros((flat.shape[0], 3), dtype=np.float64)
            tape_out = np.zeros_like(ee_out)
            for i, q in enumerate(flat):
                self.data.qpos[:] = q
                mujoco.mj_kinematics(self.model, self.data)
                ee_out[i] = self.data.site_xpos[self._ee_site_id]
                tape_out[i] = self.data.body("tape_roll").xpos
            ee = ee_out.reshape(qpos.shape[:-1] + (3,))
            tape = tape_out.reshape(qpos.shape[:-1] + (3,))

        return np.concatenate([
            qpos, qvel, ee - tape, tape - target, ee - target,
        ], axis=-1)

    @property
    def obs_dim(self) -> int:
        return self.model.nq + self.model.nv + 9   # 13 + 12 + 9 = 34

    # ------------------------------------------------------------------
    # Cost
    # ------------------------------------------------------------------

    def running_cost(
        self,
        states: Float[Array, "K H nstate"],
        actions: Float[Array, "K H nu"],
        sensordata: Float[Array, "K H nsensor"] | None = None,
    ) -> Float[Array, "K H"]:
        ee = sensordata[..., self._ee_pos_slice]
        tape = sensordata[..., self._tape_pos_slice]
        ee_to_tape = np.linalg.norm(ee - tape, axis=-1)

        # Orientation penalty. ee_quat is mujoco wxyz: [qw, qx, qy, qz].
        # The body's local +z axis in world coords is the third column of
        # the rotation matrix:
        #   R[:, 2] = [2(qx·qz + qy·qw), 2(qy·qz − qx·qw), 1 − 2(qx² + qy²)]
        # The body's local +z is rotated to local +x by the body's
        # `euler="0 1.57 0"` (quat composes with the joint-frame rotation
        # already baked into qx/qy/qz/qw via the kinematic chain), so the
        # CYLINDER axis in world is what `xmat @ [0,0,1]` returns — this
        # is exactly R[:, 2]. Verified empirically: at the init pose the
        # downward orientation gives cyl_z = −1.
        # Penalty = 1 + cyl_z ∈ [0, 2]: 0 when perfectly down, 2 when up.
        ee_quat = sensordata[..., self._ee_quat_slice]
        qx = ee_quat[..., 1]
        qy = ee_quat[..., 2]
        cyl_z = 1.0 - 2.0 * (qx ** 2 + qy ** 2)
        orient_penalty = 1.0 + cyl_z   # 0 = down, 2 = up

        return _W_REACH * ee_to_tape ** 2 + _W_ORIENT * orient_penalty

    def terminal_cost(
        self,
        states: Float[Array, "K nstate"],
        sensordata: Float[Array, "K nsensor"] | None = None,
    ) -> Float[Array, "K"]:
        tape = sensordata[..., self._tape_pos_slice]
        return _W_TERMINAL * np.linalg.norm(tape - self._target_pos, axis=-1)
