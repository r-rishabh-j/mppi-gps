"""Adroit Relocate task — vendored from gymnasium-robotics.

30-DoF Adroit arm+hand picking up a sphere and bringing it to a randomly
sampled 3-D target.

State / action layout
---------------------
* ``model.nq = model.nv = 36`` — 6 arm joints (3 slide ``ARTx/y/z`` + 3 hinge
  ``ARRx/y/z``) + 24 hand joints + 6 object joints (3 slide + 3 hinge).
* ``model.nu = 30`` — 6 arm + 24 hand actuators.
* Object body has identity orientation, so the 3 ``OBJT*`` slides are
  world-aligned: ``xpos[Object] = body_pos[Object] + qpos[30:33]``. The
  body's initial position is randomised at reset by mutating
  ``model.body_pos[Object]`` (matching upstream).

Observations
------------
39-D, derivable from ``(qpos, qvel)`` + per-episode object/target attrs so
the same composition works in both ``_get_obs`` and ``state_to_obs``:

  [0:30)   ``qpos[:30]``  (arm + hand joints)
  [30:33)  ``obj_pos`` world (init body pos + slide qpos)
  [33:36)  ``target_pos`` world (constant within an episode)
  [36:39)  ``obj_pos - target_pos``

We deliberately drop the ``palm-obj`` / ``palm-target`` vectors that
upstream's 39-D obs uses — those require forward kinematics on the hand
which can't be done in vectorised ``state_to_obs`` without an extra
``mj_forward`` per state. The policy can reconstruct an approximate palm
pose from the 6 arm joints; the cost (which has full sensordata in the
batched path) does see the true palm.

Cost
----
Dense, smooth, based on the canonical DAPG / hand_dapg formula plus a
touch-sensor shaping layer that prevents the palm-balancing reward hack
(MPPI sliding the open palm under the ball instead of grasping)::

    reward = -1.0 * ||palm - obj||                   # reach
    if USE_GRASP_SHAPING:                            # anti-cheat layer
        prox_i     = exp(-||fingertip_i - obj||² / GRIP_PROX_SCALE²)
        grip_score = Σ_i prox_i · touch_i
        grasp      = tanh(grip_score / GRIP_TOUCH_SCALE)   # establishment
        tight      = tanh(grip_score / TIGHT_GRIP_SCALE)   # firmness, slow sat
        reward += GRIP_BONUS       * grasp
        reward += TIGHT_GRIP_BONUS * tight
        if PALM_PENALTY > 0:                              # off by default
            palm_near = exp(-||palm - obj||² / GRIP_PROX_SCALE²)
            reward -= PALM_PENALTY * tanh(palm_touch/PALM_TOUCH_SCALE) * palm_near
        lift_gate = grasp                                # gate uses establishment
    else:
        lift_gate = 1.0
    if obj_z > 0.04:                                 # lifted
        reward += lift_gate * 2 * (1 - 0.5*||palm-tgt|| - 0.5*||obj-tgt||)
    if ||obj - target|| < 0.1:  reward += lift_gate * 10
    if ||obj - target|| < 0.05: reward += lift_gate * 20
    reward -= ARM_VEL_PENALTY    * Σ qvel[:6]²        # smoothness (arm)
    reward -= FINGER_VEL_PENALTY * Σ qvel[6:30]²      # smoothness (fingers)
    cost = -reward

Set ``USE_GRASP_SHAPING = False`` (module constant) to revert to the plain
dense reward.

NOTE: gymnasium-robotics' adroit_relocate.py has a typo in its dense
reward — ``reward = +0.1 * ||palm - obj||`` instead of ``-0.1 * ...`` —
which would push the hand AWAY from the object. We use the correct sign
(and a 10x stronger weight) from the original DAPG implementation.
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

# Object lift threshold (z) above which the lift-bonus reward terms fire.
_LIFT_Z = 0.04
# Goal-distance thresholds for the milestone bonuses.
_GOAL_LOOSE = 0.1
_GOAL_TIGHT = 0.05

# ---- Anti-cheat / grasp-shaping (touch-sensor based) ----
# Set to False to revert to the plain dense reward (palm-balancing reward hack
# returns). When True, MPPI is rewarded for fingertip contact, penalised for
# palm contact, and the lift bonus is gated on grasp strength.
USE_GRASP_SHAPING = True
# Palm contact penalty — off by default. The per-finger proximity gate on
# the grip bonus + lift_gate already make pure palm-balancing strictly
# worse than grasping (no fingertip contact → no grip → no lift → no
# milestone). Penalising palm contact additionally forces fingertip-only
# "pinch" grasps and pushes the palm high above the ball, which makes
# fingertips collide with the table on approach. Palmar grasps (palm +
# fingers both on ball) are actually a stable, human-like configuration.
# Re-enable (set > 0) only if you see palm-balancing return.
_PALM_PENALTY = 0.0
_GRIP_BONUS = 4.0         # weight on tanh(grip_score / GRIP_TOUCH_SCALE)
_PALM_TOUCH_SCALE = 0.5   # tanh saturation for palm touch (used iff PENALTY > 0)
_GRIP_TOUCH_SCALE = 1.0   # establishment tanh — saturates fast (~grip_score=2)
                          # so MPPI gets a steep gradient establishing contact.
# Tighter-grip reward: a second tanh with a much larger scale stays
# unsaturated well past the "barely touching" regime, so MPPI keeps being
# rewarded for *firmer* grips that survive arm motion. Without this, the
# basic tanh saturates and MPPI converges to the loosest possible grasp,
# which slips under any perturbation. Saturates around grip_score=15 to
# prevent infinite-force pumping.
_TIGHT_GRIP_BONUS = 2.0
_TIGHT_GRIP_SCALE = 5.0
# Per-finger proximity gate σ (m). exp(-d²/σ²) — finger contributes ≈ 1 when
# touching the ball (ball radius 0.035), ≈ 0 when touching the table or air
# with the ball elsewhere. Stops the "fingers flat on table" cheat.
_GRIP_PROXIMITY_SCALE = 0.05

# ---- Smoothness / "human-like" motion ----
# Per-DoF velocity² penalty on the controlled joints. Split so the arm
# (6 DoF, large links) is gently smoothed for human-like reaching, while
# fingers (24 DoF) can still snap closed quickly during grasping — humans
# actually move their fingers fast (~5-15 rad/s) when committing to a
# grasp; only the arm's coarse motion looks robotic when rushed.
#
# A uniform penalty taxes finger-close velocity equally, so MPPI learns
# the locally-cheaper "hover with open fingers" strategy: doing nothing
# is free, closing fingers has immediate cost, grip reward is far in the
# horizon. Splitting the weights restores the right cost shape.
#
# Object qvel[30:] is always excluded — moving the ball IS the task.
# Set both to 0.0 to disable. Tune up arm penalty if reaches still look
# jerky; tune up finger penalty only if fingers visibly twitch.
_ARM_VEL_PENALTY = 0.02      # qvel[:6]   — ARTx/y/z, ARRx/y/z
_FINGER_VEL_PENALTY = 0.001  # qvel[6:30] — 24 hand joints


class AdroitRelocate(MuJoCoEnv):
    def __init__(self, frame_skip: int = 5, **kwargs) -> None:
        super().__init__(model_path=_XML, frame_skip=frame_skip, **kwargs)
        assert self.model.nq == 36 and self.model.nv == 36 and self.model.nu == 30

        self._obj_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "Object"
        )
        self._target_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "target"
        )

        self._obj_pos_slice = self._sensor_slice("object_pos")
        self._palm_pos_slice = self._sensor_slice("palm_pos")
        # Cached unconditionally — the cost path only reads these when
        # USE_GRASP_SHAPING is True, but caching is cheap and lets the toggle
        # flip at runtime without re-doing name lookups.
        self._palm_touch_slice = self._sensor_slice("ST_Tch_palm")
        self._fingertip_touch_slice = self._fingertip_slice()
        # Fingertip world positions, contiguous: 5 sensors × 3 dims = 15
        # floats laid out as [ff, mf, rf, lf, th]. Same finger order as
        # _fingertip_touch_slice so per-finger proximity weighting is
        # element-wise on the last axis after a reshape.
        self._fingertip_pos_slice = self._fingertip_pos_slice_helper()

        # Per-episode constants (refreshed on every reset).
        self._obj_body_init_pos = self.model.body_pos[self._obj_body_id].copy()
        self._target_pos = self.model.site_pos[self._target_site_id].copy()

    def _sensor_slice(self, name: str) -> slice:
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        adr = int(self.model.sensor_adr[sid])
        dim = int(self.model.sensor_dim[sid])
        return slice(adr, adr + dim)

    def _fingertip_slice(self) -> slice:
        """Contiguous slice over the 5 fingertip touch sensors. Asserted
        contiguous so XML drift is caught loudly rather than silently
        summing the wrong dims."""
        return self._contiguous_sensor_slice(
            ("ST_Tch_fftip", "ST_Tch_mftip", "ST_Tch_rftip",
             "ST_Tch_lftip", "ST_Tch_thtip"),
            stride=1,
        )

    def _fingertip_pos_slice_helper(self) -> slice:
        """Contiguous slice over the 5 fingertip framepos sensors (15 floats).

        Same finger order as the touch slice so per-finger weighting is a
        single element-wise multiply after a (..., 5, 3) reshape.
        """
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

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, state: np.ndarray | None = None) -> np.ndarray:
        if state is not None:
            return super().reset(state=state)

        mujoco.mj_resetData(self.model, self.data)

        # Randomise object body position (in x,y; z stays at 0.035).
        self.model.body_pos[self._obj_body_id, 0] = np.random.uniform(*_OBJ_X_RANGE)
        self.model.body_pos[self._obj_body_id, 1] = np.random.uniform(*_OBJ_Y_RANGE)
        self._obj_body_init_pos = self.model.body_pos[self._obj_body_id].copy()

        # Randomise target site position (x,y,z).
        self.model.site_pos[self._target_site_id, 0] = np.random.uniform(*_TARGET_X_RANGE)
        self.model.site_pos[self._target_site_id, 1] = np.random.uniform(*_TARGET_Y_RANGE)
        self.model.site_pos[self._target_site_id, 2] = np.random.uniform(*_TARGET_Z_RANGE)
        self._target_pos = self.model.site_pos[self._target_site_id].copy()

        mujoco.mj_forward(self.model, self.data)
        return self._get_obs()

    # ------------------------------------------------------------------
    # Cost
    # ------------------------------------------------------------------

    def _running_reward_terms(
        self,
        sensordata: Float[Array, "... nsensor"],
    ) -> tuple[Float[Array, "... 3"], Float[Array, "... 3"], Float[Array, "..."]]:
        """Return ``(palm_pos, obj_pos, goal_distance)`` over arbitrary leading
        dims. ``target_pos`` is read from ``self._target_pos`` (constant per
        episode)."""
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

        # Reach: pull hand toward object.
        reward = -1.0 * np.linalg.norm(palm - obj, axis=-1) * 10

        # Touch-sensor based grasp shaping. Anti-cheat:
        #   - palm-balancing: penalise palm contact directly
        #   - fingers-on-table: per-finger proximity gate so a fingertip's
        #     touch only counts when *that finger* is near the ball
        # Lift bonus is gated on the resulting grasp score so neither cheat
        # unlocks the carry-to-target reward.
        if USE_GRASP_SHAPING:
            ftouch = sensordata[..., self._fingertip_touch_slice]            # (..., 5)
            ftpos = sensordata[..., self._fingertip_pos_slice]               # (..., 15)
            ftpos = ftpos.reshape(*ftpos.shape[:-1], 5, 3)                   # (..., 5, 3)
            d2 = ((ftpos - obj[..., None, :]) ** 2).sum(axis=-1)             # (..., 5)
            proximity = np.exp(-d2 / (_GRIP_PROXIMITY_SCALE ** 2))           # (..., 5)
            grip_score = (ftouch * proximity).sum(axis=-1)                   # (...)
            grasp = np.tanh(grip_score / _GRIP_TOUCH_SCALE)                  # 0..1
            tight = np.tanh(grip_score / _TIGHT_GRIP_SCALE)                  # 0..1, slow
            reward = (
                reward
                + _GRIP_BONUS * grasp           # establishment (saturates fast)
                + _TIGHT_GRIP_BONUS * tight     # firmness (keeps growing)
            )
            if _PALM_PENALTY > 0.0:
                palm_t = sensordata[..., self._palm_touch_slice].sum(axis=-1)
                palm_d2 = ((palm - obj) ** 2).sum(axis=-1)
                palm_near = np.exp(-palm_d2 / (_GRIP_PROXIMITY_SCALE ** 2))
                palm_s = np.tanh(palm_t / _PALM_TOUCH_SCALE) * palm_near
                reward = reward - _PALM_PENALTY * palm_s
            lift_gate = grasp
        else:
            lift_gate = 1.0

        # Lift bonus + travel-to-target shaping (only when the object is up).
        lifted = obj_z > _LIFT_Z
        lift_bonus = (
            1.0
            # - 0.5 * np.linalg.norm(palm - target, axis=-1)
            - 0.5 * np.linalg.norm(obj - target, axis=-1)
        ) * 4.0
        reward = reward + np.where(lifted, lift_bonus * lift_gate, 0.0)

        # Milestone bonuses for object near target — gated on grasp so a
        # fling that happens to bounce through the target zone earns nothing.
        # Without the gate, +10/+20 per step swamps any velocity penalty and
        # MPPI commits to bang-bang arm swings (the "fling-for-luck" exploit
        # that the velocity penalty alone cannot suppress).
        milestone = (
            np.where(goal_dist < _GOAL_LOOSE, 10.0, 0.0)
            + np.where(goal_dist < _GOAL_TIGHT, 20.0, 0.0)
        )
        reward = reward + lift_gate * milestone

        # Smoothness — split arm vs finger so fast finger-closing isn't
        # taxed at the same rate as a robotic arm sweep. Object qvel[30:]
        # excluded (moving the ball is the goal).
        qvel = self.state_qvel(states)
        arm_v2 = (qvel[..., :6] ** 2).sum(axis=-1)
        finger_v2 = (qvel[..., 6:30] ** 2).sum(axis=-1)
        reward = (
            reward
            - _ARM_VEL_PENALTY * arm_v2
            - _FINGER_VEL_PENALTY * finger_v2
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
        # Upstream relocate never terminates early (truncates at max_steps).
        return obs, cost, False, info

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def _build_obs(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
    ) -> np.ndarray:
        # Object body has identity orientation, so OBJTx/y/z are world-aligned:
        #   obj_world = body_init_pos + qpos[30:33]
        obj_world = self._obj_body_init_pos + qpos[..., 30:33]
        target = np.broadcast_to(self._target_pos, obj_world.shape)
        return np.concatenate(
            [
                qpos[..., :30],
                obj_world,
                target,
                obj_world - target,
            ],
            axis=-1,
        )

    def _get_obs(self) -> np.ndarray:
        return self._build_obs(self.data.qpos.copy(), self.data.qvel.copy())

    def state_to_obs(self, states: np.ndarray) -> np.ndarray:
        qpos = self.state_qpos(states)
        qvel = self.state_qvel(states)
        return self._build_obs(qpos, qvel)

    @property
    def obs_dim(self) -> int:
        # 30 + 3 + 3 + 3 = 39
        return 39

    @property
    def noise_scale(self) -> np.ndarray:
        """Per-dim half-range so cfg.noise_sigma reads as a fraction of each
        actuator's range. Critical for relocate: arm slide ranges (0.2-0.5)
        and finger ranges (1.6-2.0) span 10x; without this, a single scalar
        sigma either slams the arm against its limits every step or leaves
        the fingers under-explored.
        """
        low, high = self.action_bounds
        return 0.5 * (high - low)
