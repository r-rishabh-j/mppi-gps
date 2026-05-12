"""Adroit Relocate (30-DoF arm+hand picking a sphere to a 3-D target).

Layout: nq=nv=36 (6 arm + 24 hand + 6 object joints). nu=30. Object body
has identity orientation, so OBJT slides are world-aligned:
``xpos[Object] = body_pos[Object] + qpos[30:33]``.

Observations (toggle via ``USE_DAPG_OBS``):
- False → 72-D: qpos[:30] + qvel[:30] + obj_pos + obj_vel + target +
  (obj-target). Cheap from (qpos, qvel) alone.
- True (default) → 39-D D4RL/DAPG: qpos[:30] + (palm-obj) + (palm-target)
  + (obj-target); no qvel; needs palm position. ``state_to_obs`` falls
  back to mj_kinematics when no sensordata is supplied.

Cost (DAPG dense reward + touch-sensor shaping). With
``USE_GRASP_SHAPING=True``, anti-cheat against palm-balancing /
fingers-on-table / fling-for-luck via per-finger proximity gates and a
grasp gate on the lift / milestone rewards. See ``running_cost`` for the
exact formula.
"""

from pathlib import Path

import mujoco
import numpy as np
from jaxtyping import Array, Float

from src.envs.mujoco_env import MuJoCoEnv

_XML = str(Path(__file__).resolve().parents[2] / "assets" / "adroit" / "relocate.xml")

# Per-episode randomisation ranges (match upstream).
_OBJ_X_RANGE = (-0.15, 0.15)
_OBJ_Y_RANGE = (-0.15, 0.30)
_TARGET_X_RANGE = (-0.20, 0.20)
_TARGET_Y_RANGE = (-0.20, 0.20)
_TARGET_Z_RANGE = (0.15, 0.35)

# Object lift threshold (z) for the lift-bonus terms.
_LIFT_Z = 0.04
# Object → target goal distance thresholds for milestone bonuses.
_GOAL_LOOSE = 0.1
_GOAL_TIGHT = 0.05

# Touch-sensor based grasp shaping (anti-cheat). False → plain dense reward.
USE_GRASP_SHAPING = True

# Observation layout. True → 39-D D4RL/DAPG (qpos + 3 deltas, no qvel,
# needs palm pos via mj_kinematics in state_to_obs). False → 72-D
# (qpos+qvel + obj/target absolutes + delta), pure qpos/qvel.
USE_DAPG_OBS = True

# Grip reward: a fast-saturating "establishment" tanh + a slow-saturating
# "tightness" tanh so MPPI is still rewarded for firmer grips after the
# first contact saturates the basic term.
_GRIP_BONUS = 5.0
_GRIP_TOUCH_SCALE = 1
_TIGHT_GRIP_BONUS = 2.0
_TIGHT_GRIP_SCALE = 3.0

# Per-finger Gaussian proximity gate σ (m). Stops the "fingers flat on table"
# cheat: a fingertip's touch only counts when that finger is near the ball.
_GRIP_PROXIMITY_SCALE = 0.05
# Per-finger weights into grip_score (ff, mf, rf, lf, th).
_FINGER_GRIP_WEIGHTS = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

# Thumb-tip → ball Gaussian pull (independent of contact). Wider σ than
# the grip gate so MPPI gets a gradient pulling the thumb in from the
# open-hand pose, where the contact-gated grip term is ~0.
_THUMB_PROXIMITY_BONUS = 2.0
_THUMB_PROXIMITY_SCALE = 0.10

# Smoothness penalties on qvel². Arm is gently smoothed; fingers can
# still snap closed during grasping. Object qvel[30:] excluded.
_ARM_VEL_PENALTY = 0.05
_FINGER_VEL_PENALTY = 0.005

# Control penalty on Σ action[:6]² (arm only) — tie-breaker against MPPI
# slamming the arm into joint limits. Hand controls excluded.
_ARM_CTRL_PENALTY = 0.05


class AdroitRelocate(MuJoCoEnv):
    def __init__(self, frame_skip: int = 5, **kwargs) -> None:
        super().__init__(model_path=_XML, frame_skip=frame_skip, **kwargs)
        assert self.model.nq == 36 and self.model.nv == 36 and self.model.nu == 30

        self._obj_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "Object"
        )
        # Target authoritative position is body_pos[target_body]; the
        # site under it is purely visual.
        self._target_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "target_body"
        )

        self._obj_pos_slice = self._sensor_slice("object_pos")
        self._palm_pos_slice = self._sensor_slice("palm_pos")
        # palm_pos is a framepos of the S_grasp site; cached site id for
        # the DAPG-obs mj_kinematics fallback.
        self._palm_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "S_grasp"
        )
        # Virtual reach target offset above the palm (palm-side; see
        # S_reach in adroit_model.xml). The reach reward pulls THIS to
        # the ball, keeping the palm below it so fingers can wrap.
        self._reach_pos_slice = self._sensor_slice("reach_pos")
        self._fingertip_touch_slice = self._fingertip_slice()
        # 5 fingertip framepos sensors × 3 dims = 15 floats; finger order
        # (ff, mf, rf, lf, th) matches the touch slice.
        self._fingertip_pos_slice = self._fingertip_pos_slice_helper()

        self._obj_body_init_pos = self.model.body_pos[self._obj_body_id].copy()
        self._target_pos = self.model.body_pos[self._target_body_id].copy()

    def _sensor_slice(self, name: str) -> slice:
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        adr = int(self.model.sensor_adr[sid])
        dim = int(self.model.sensor_dim[sid])
        return slice(adr, adr + dim)

    def _fingertip_slice(self) -> slice:
        """5 fingertip touch sensors; contiguity asserted to catch XML drift."""
        return self._contiguous_sensor_slice(
            ("ST_Tch_fftip", "ST_Tch_mftip", "ST_Tch_rftip",
             "ST_Tch_lftip", "ST_Tch_thtip"),
            stride=1,
        )

    def _fingertip_pos_slice_helper(self) -> slice:
        """5 fingertip framepos sensors (15 floats); same finger order as touch."""
        return self._contiguous_sensor_slice(
            ("fftip_pos", "mftip_pos", "rftip_pos", "lftip_pos", "thtip_pos"),
            stride=3,
        )

    def _contiguous_sensor_slice(self, names: tuple[str, ...], stride: int) -> slice:
        adrs = [
            int(self.model.sensor_adr[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, n)
            ])
            for n in names
        ]
        expected = list(range(adrs[0], adrs[0] + stride * len(names), stride))
        assert adrs == expected, (
            f"expected contiguous sensors {names} stride={stride}, got {adrs}"
        )
        return slice(adrs[0], adrs[0] + stride * len(names))

    def reset(self, state: np.ndarray | None = None) -> np.ndarray:
        if state is not None:
            return super().reset(state=state)

        mujoco.mj_resetData(self.model, self.data)

        # Randomise object via QPOS (slide joints at 30:33), NOT
        # body_pos: mjw.put_model captures the model at construction, so
        # later body_pos mutations don't propagate to the GPU copy. qpos
        # IS dynamic state and is broadcast to warp worlds via
        # WarpRolloutMixin._seed_warp_state.
        self.data.qpos[30] = np.random.uniform(*_OBJ_X_RANGE)
        self.data.qpos[31] = np.random.uniform(*_OBJ_Y_RANGE)
        # qpos[32] stays 0; total ball z = body_pos.z (0.035 from XML).

        # Target is read from ``self._target_pos`` (Python attribute) by
        # every cost/obs callsite, so mutating CPU body_pos here is fine
        # — the warp side never needs target position.
        self.model.body_pos[self._target_body_id, 0] = np.random.uniform(*_TARGET_X_RANGE)
        self.model.body_pos[self._target_body_id, 1] = np.random.uniform(*_TARGET_Y_RANGE)
        self.model.body_pos[self._target_body_id, 2] = np.random.uniform(*_TARGET_Z_RANGE)
        self._target_pos = self.model.body_pos[self._target_body_id].copy()

        mujoco.mj_forward(self.model, self.data)
        return self._get_obs()

    def _running_reward_terms(
        self,
        sensordata: Float[Array, "... nsensor"],
    ) -> tuple[Float[Array, "... 3"], Float[Array, "... 3"], Float[Array, "..."]]:
        """Return (palm_pos, obj_pos, goal_distance)."""
        palm = sensordata[..., self._palm_pos_slice]
        obj = sensordata[..., self._obj_pos_slice]
        goal_dist = np.linalg.norm(obj - self._target_pos, axis=-1)
        return palm, obj, goal_dist

    def running_cost(
        self,
        states: Float[Array, "K H nstate"],
        actions: Float[Array, "K H nu"],
        sensordata: Float[Array, "K H nsensor"] | None = None,
    ) -> Float[Array, "K H"]:
        if sensordata is None:
            raise ValueError(
                "AdroitRelocate.running_cost requires sensordata "
                "(palm and object positions)."
            )

        palm, obj, goal_dist = self._running_reward_terms(sensordata)
        target = self._target_pos
        obj_z = obj[..., 2]

        # Reach pulls the S_reach virtual target (offset above the palm,
        # palm-side) to the ball — keeps the palm below the ball.
        reach_pos = sensordata[..., self._reach_pos_slice]
        reward = -1.0 * np.linalg.norm(reach_pos - obj, axis=-1) * 10

        if USE_GRASP_SHAPING:
            ftouch = sensordata[..., self._fingertip_touch_slice]            # (..., 5)
            ftpos = sensordata[..., self._fingertip_pos_slice]
            ftpos = ftpos.reshape(*ftpos.shape[:-1], 5, 3)                   # (..., 5, 3)
            d2 = ((ftpos - obj[..., None, :]) ** 2).sum(axis=-1)             # (..., 5)
            proximity = np.exp(-d2 / (_GRIP_PROXIMITY_SCALE ** 2))
            grip_score = (ftouch * proximity * _FINGER_GRIP_WEIGHTS).sum(axis=-1)
            grasp = np.tanh(grip_score / _GRIP_TOUCH_SCALE)   # 0..1, fast sat
            tight = np.tanh(grip_score / _TIGHT_GRIP_SCALE)   # 0..1, slow sat
            # Wider Gaussian on thumb-tip → ball distance: gives a
            # gradient pulling the thumb in before contact.
            thumb_proximity = np.exp(-d2[..., 4] / (_THUMB_PROXIMITY_SCALE ** 2))
            reward = (
                reward
                + _GRIP_BONUS * grasp
                + _TIGHT_GRIP_BONUS * tight
                + _THUMB_PROXIMITY_BONUS * thumb_proximity
            )
            lift_gate = grasp * tight
        else:
            lift_gate = 1.0

        # Lift bonus + travel-to-target shaping; gated on grasp so neither
        # palm-balancing nor fling-for-luck unlocks it.
        lifted = obj_z > _LIFT_Z
        lift_bonus = (
            1.0
            - 0.5 * np.linalg.norm(palm - target, axis=-1)
            - 0.5 * np.linalg.norm(obj - target, axis=-1)
        )
        ball_height = obj_z - 0.035   # subtract ball radius (rest z)
        reward += 2 * np.tanh(ball_height / 0.1) * grasp
        reward += np.where(lifted, lift_bonus * lift_gate, 0.0)

        # Milestone bonuses, gated on grasp.
        milestone = (
            np.where(goal_dist < _GOAL_LOOSE, 10.0, 0.0)
            + np.where(goal_dist < _GOAL_TIGHT, 20.0, 0.0)
        ) * 5
        reward = reward + lift_gate * milestone

        # Smoothness: split arm vs finger; object qvel excluded.
        qvel = self.state_qvel(states)
        arm_v2 = (qvel[..., :6] ** 2).sum(axis=-1)
        finger_v2 = (qvel[..., 6:30] ** 2).sum(axis=-1)
        arm_u2 = (actions[..., :6] ** 2).sum(axis=-1)
        reward = (
            reward
            - (_ARM_VEL_PENALTY * arm_v2
               + _FINGER_VEL_PENALTY * finger_v2
               + _ARM_CTRL_PENALTY * arm_u2 * 10)
        )

        return -reward

    def terminal_cost(
        self,
        states: Float[Array, "K nstate"],
        sensordata: Float[Array, "K nsensor"] | None = None,
    ) -> Float[Array, "K"]:
        return np.zeros(states.shape[0])

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        obs, cost, _, info = super().step(action)
        sd = self.data.sensordata
        _, _, goal_dist = self._running_reward_terms(sd)
        success = bool(goal_dist < _GOAL_LOOSE)
        info["success"] = success
        info["goal_distance"] = float(goal_dist)
        # Upstream relocate never terminates early.
        return obs, cost, False, info

    def _build_obs(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
        palm_pos: np.ndarray | None = None,
    ) -> np.ndarray:
        # Object body has identity orientation → OBJT slides world-aligned.
        obj_world = self._obj_body_init_pos + qpos[..., 30:33]
        target = np.broadcast_to(self._target_pos, obj_world.shape)

        if USE_DAPG_OBS:
            if palm_pos is None:
                raise ValueError(
                    "USE_DAPG_OBS=True requires palm_pos."
                )
            # palm-target slot is zeroed (kept for shape stability with
            # older checkpoints / datasets).
            palm_target_zero = np.zeros_like(palm_pos)
            return np.concatenate(
                [
                    qpos[..., :30],          # 30
                    palm_pos - obj_world,    # 3
                    palm_target_zero,        # 3 (zeroed, preserves obs_dim=39)
                    obj_world - target,      # 3
                ],
                axis=-1,
            )

        # 72-D layout.
        obj_vel = qvel[..., 30:33]
        return np.concatenate(
            [
                qpos[..., :30],
                qvel[..., :30],
                obj_world,
                obj_vel,
                target,
                obj_world - target,
            ],
            axis=-1,
        )

    def _get_obs(self) -> np.ndarray:
        palm = (
            self.data.sensordata[self._palm_pos_slice].copy()
            if USE_DAPG_OBS
            else None
        )
        return self._build_obs(
            self.data.qpos.copy(), self.data.qvel.copy(), palm_pos=palm
        )

    def state_to_obs(
        self,
        states: np.ndarray,
        sensordata: np.ndarray | None = None,
    ) -> np.ndarray:
        qpos = self.state_qpos(states)
        qvel = self.state_qvel(states)
        palm = None
        if USE_DAPG_OBS:
            if sensordata is not None:
                # Hot path: palm comes from batch_rollout's sensordata.
                palm = np.asarray(sensordata)[..., self._palm_pos_slice]
            else:
                # Eval / relabel path: per-state mj_kinematics fallback.
                palm = self._batched_palm_pos(qpos)
        return self._build_obs(qpos, qvel, palm_pos=palm)

    def _batched_palm_pos(self, qpos: np.ndarray) -> np.ndarray:
        """Per-state S_grasp site world position via mj_kinematics.

        Fallback for state_to_obs when no sensordata is supplied. O(N)
        Python loop — fine for relabel / eval, NOT for the MPPI inner loop.
        """
        leading = qpos.shape[:-1]
        flat_q = qpos.reshape(-1, qpos.shape[-1])
        out = np.empty((flat_q.shape[0], 3), dtype=np.float64)
        saved_q = self.data.qpos.copy()
        try:
            for i in range(flat_q.shape[0]):
                self.data.qpos[:] = flat_q[i]
                mujoco.mj_kinematics(self.model, self.data)
                out[i] = self.data.site_xpos[self._palm_site_id]
        finally:
            self.data.qpos[:] = saved_q
            mujoco.mj_kinematics(self.model, self.data)
        return out.reshape(*leading, 3)

    @property
    def obs_dim(self) -> int:
        return 39 if USE_DAPG_OBS else 72

    @property
    def noise_scale(self) -> np.ndarray:
        """Per-dim half-range — arm and finger ctrlranges span 10×."""
        low, high = self.action_bounds
        return 0.5 * (high - low)
