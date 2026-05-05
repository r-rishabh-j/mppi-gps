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
Two layouts, switched by the ``USE_DAPG_OBS`` module constant.

**``USE_DAPG_OBS = False`` (default) — 72-D**, derivable from
``(qpos, qvel)`` + per-episode object/target attrs so the same composition
works in both ``_get_obs`` and ``state_to_obs`` with no extra kinematics:

  [0:30)    ``qpos[:30]``         arm + hand joint angles
  [30:60)   ``qvel[:30]``         arm + hand joint velocities
  [60:63)   ``obj_pos`` world     init body pos + slide qpos
  [63:66)   ``obj_vel`` world     ball linear velocity (``qvel[30:33]``)
  [66:69)   ``target_pos`` world  constant within an episode
  [69:72)   ``obj_pos - target_pos``

Velocities are critical for any dynamic manipulation: without them the
policy can't distinguish "ball rising" from "ball at apex" from "ball
falling", and BC plateaus far below the MPPI teacher.

**``USE_DAPG_OBS = True`` — 39-D**, the canonical D4RL/DAPG relocate
observation:

  [0:30)    ``qpos[:30]``                 arm + hand joint angles
  [30:33)   ``palm_pos - obj_pos``        hand-to-ball delta
  [33:36)   ``palm_pos - target_pos``     hand-to-target delta
  [36:39)   ``obj_pos - target_pos``      ball-to-target delta

Translation-invariant by construction (the policy sees displacements, not
world coordinates), and matches the D4RL ``relocate-*`` datasets so D4RL
checkpoints can be loaded directly. No qvel — DAPG deliberately omits it
and the public datasets reflect that. Requires palm position, so
``state_to_obs`` falls back to a per-state ``mj_forward`` loop. Acceptable
because ``state_to_obs`` is only called in the relabel / eval path, never
inside the MPPI inner loop (where the batched ``running_cost`` reads the
true palm directly from ``sensordata``).

Cost
----
Dense, smooth, based on the canonical DAPG / hand_dapg formula plus a
touch-sensor shaping layer. Anti-cheat (palm-balancing, fling-for-luck,
fingers-on-table) is achieved structurally: the per-finger proximity gate
on the grip bonus + the lift_gate on lift / milestone rewards make every
non-grasp strategy strictly dominated::

    reward = -10 * ||reach_pos - obj||               # reach (S_reach site,
                                                     # offset forward of palm)
    if USE_GRASP_SHAPING:
        prox_i     = exp(-||fingertip_i - obj||² / GRIP_PROX_SCALE²)
        grip_score = Σ_i prox_i · touch_i
        grasp      = tanh(grip_score / GRIP_TOUCH_SCALE)   # establishment
        tight      = tanh(grip_score / TIGHT_GRIP_SCALE)   # firmness, slow sat
        reward    += GRIP_BONUS       * grasp
        reward    += TIGHT_GRIP_BONUS * tight
        lift_gate  = grasp                                # gate uses establishment
    else:
        lift_gate = 1.0
    if obj_z > 0.04:                                 # lifted
        reward += lift_gate * 2 * (1 - 0.5*||palm-tgt|| - 0.5*||obj-tgt||)
    if ||obj - target|| < 0.1:  reward += lift_gate * 10
    if ||obj - target|| < 0.05: reward += lift_gate * 20
    reward -= ARM_VEL_PENALTY    * Σ qvel[:6]²        # smoothness (arm vel)
    reward -= FINGER_VEL_PENALTY * Σ qvel[6:30]²      # smoothness (finger vel)
    reward -= ARM_CTRL_PENALTY   * Σ action[:6]²      # arm control magnitude
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

# Set to False to revert to the plain dense reward (palm-balancing reward hack
# returns). When True, MPPI is rewarded for fingertip contact, penalised for
# palm contact, and the lift bonus is gated on grasp strength.
USE_GRASP_SHAPING = True

# Switch the observation layout. False (default) → 72-D obs with qpos+qvel +
# absolute object/target positions + obj-target delta; cheap to compute in
# both _get_obs and the batched state_to_obs path. True → 39-D D4RL/DAPG obs
# (qpos[:30] + palm-obj + palm-target + obj-target deltas, no qvel); requires
# palm position so state_to_obs runs an mj_forward per state. Flip when you
# want to ingest D4RL relocate datasets or test translation-invariant
# representations on small data; keep False for self-collected MPPI data
# where qvel actually helps the policy. See module docstring for the layout.
USE_DAPG_OBS = True

_GRIP_BONUS = 4.0         # weight on tanh(grip_score / GRIP_TOUCH_SCALE)
_GRIP_TOUCH_SCALE = 1  # establishment tanh — saturates fast (~grip_score=2)
                          # so MPPI gets a steep gradient establishing contact.
# Tighter-grip reward: a second tanh with a much larger scale stays
# unsaturated well past the "barely touching" regime, so MPPI keeps being
# rewarded for *firmer* grips that survive arm motion. Without this, the
# basic tanh saturates and MPPI converges to the loosest possible grasp,
# which slips under any perturbation. Saturates around grip_score=15 to
# prevent infinite-force pumping.
_TIGHT_GRIP_BONUS = 2.0
_TIGHT_GRIP_SCALE = 3.0
# Per-finger proximity gate σ (m). exp(-d²/σ²) — finger contributes ≈ 1 when
# touching the ball (ball radius 0.035), ≈ 0 when touching the table or air
# with the ball elsewhere. Stops the "fingers flat on table" cheat.
_GRIP_PROXIMITY_SCALE = 0.05
# Per-finger weights into grip_score. Order is (ff, mf, rf, lf, th) — same
# as the touch / fingertip-pos sensor slices. The thumb is upweighted: an
# opposable grasp NEEDS the thumb specifically, but with uniform weights
# the thumb's signal is averaged 1/5 with the four fingers, so MPPI can
# satisfy ~80% of grip_score with the four fingers alone and leaves the
# thumb folded against the palm. Boosting the thumb makes "no thumb
# contact" cost more than a small loss on the other fingers, pushing
# MPPI toward true opposition grasps.
_FINGER_GRIP_WEIGHTS = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

# Thumb-specific proximity pull, independent of contact. The grip tanh
# saturates fast (at grip_score≈2 for the basic establishment term), so
# once the four fingers wrap the marginal reward for the thumb closing
# the last few cm is dwarfed by sampling noise — MPPI sees no gradient
# pulling the thumb in. The contact-gated proximity factor doesn't help
# either: at the thumb's folded rest pose it's already > 7cm from the
# ball, where exp(-d²/0.05²) is ≈ 0 with no usable gradient.
#
# A Gaussian bonus on thumb-tip → ball distance with a wider σ extends
# the gradient into the open-hand pose so MPPI gets a smooth pull from
# the moment the palm starts approaching. Bounded (saturates near 1 on
# contact) so it can't dominate the reach term, additive to the grip
# bonus (so a wrapped-but-no-thumb pose is strictly worse than a
# wrapped-with-thumb pose at the same palm position).
_THUMB_PROXIMITY_BONUS = 2.0
_THUMB_PROXIMITY_SCALE = 0.10  # σ in metres; ~2× the grip proximity gate

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
_FINGER_VEL_PENALTY = 0.005  # qvel[6:30] — 24 hand joints

# ---- Control-magnitude penalty (arm only) ----
# Penalises Σ action[:6]² — the 6 arm actuators (A_ARTx/y/z, A_ARRx/y/z).
# These are position-controlled, so this pulls commanded positions toward
# the arm's rest pose. The rest pose is far from the ball workspace, so
# this term mildly opposes the task; keep it small (regularization, not
# constraint). Mostly useful as a tie-breaker against MPPI sampling noise
# that briefly slams the arm to its joint limits. Set to 0.0 to disable.
# Hand controls intentionally excluded — finger commands need to span
# their range freely to grasp.
_ARM_CTRL_PENALTY = 0.02


class AdroitRelocate(MuJoCoEnv):
    def __init__(self, frame_skip: int = 5, **kwargs) -> None:
        super().__init__(model_path=_XML, frame_skip=frame_skip, **kwargs)
        assert self.model.nq == 36 and self.model.nv == 36 and self.model.nu == 30

        self._obj_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "Object"
        )
        # Target's authoritative position is now body_pos[target_body]
        # (so the recording cameras can targetbody it). The site under
        # this body is purely visual at body-local origin.
        self._target_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "target_body"
        )

        self._obj_pos_slice = self._sensor_slice("object_pos")
        self._palm_pos_slice = self._sensor_slice("palm_pos")
        # The palm_pos sensor is a framepos of the S_grasp site. Cached so
        # the DAPG-obs fallback path (mj_kinematics + read site_xpos) can
        # bypass mj_name2id and the framepos sensor pipeline per state.
        self._palm_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "S_grasp"
        )
        # Virtual reach target offset above the palm surface (palm-side,
        # the direction curled fingers reach toward) — see S_reach in
        # adroit_model.xml. The reach reward pulls THIS point to the
        # ball, so the palm sits *below* the ball and fingers can wrap
        # from the palm side without the palm crashing into the table.
        self._reach_pos_slice = self._sensor_slice("reach_pos")
        # Cached unconditionally — the cost path only reads this when
        # USE_GRASP_SHAPING is True, but caching is cheap and lets the toggle
        # flip at runtime without re-doing name lookups.
        self._fingertip_touch_slice = self._fingertip_slice()
        # Fingertip world positions, contiguous: 5 sensors × 3 dims = 15
        # floats laid out as [ff, mf, rf, lf, th]. Same finger order as
        # _fingertip_touch_slice so per-finger proximity weighting is
        # element-wise on the last axis after a reshape.
        self._fingertip_pos_slice = self._fingertip_pos_slice_helper()

        # Per-episode constants (refreshed on every reset).
        self._obj_body_init_pos = self.model.body_pos[self._obj_body_id].copy()
        self._target_pos = self.model.body_pos[self._target_body_id].copy()
        # Stiffen finger PD so commanded position translates to more contact
        # force. Default gain=1 gives marginal grip; gain=3 holds a 0.18kg ball
        # through normal lift acceleration.
        # GAIN = 3.0
        # finger_idx = slice(8, 30)   # A_FFJ3 .. A_THJ0 (24 hand actuators)
        # self.model.actuator_gainprm[finger_idx, 0] = GAIN
        # self.model.actuator_biasprm[finger_idx, 1] = -GAIN

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

        # Randomise target body position (x,y,z). The site sits at body
        # origin so it follows automatically; mocap-style writes to body_pos
        # work for this welded body because mj_forward below propagates the
        # change into the kinematic tree.
        self.model.body_pos[self._target_body_id, 0] = np.random.uniform(*_TARGET_X_RANGE)
        self.model.body_pos[self._target_body_id, 1] = np.random.uniform(*_TARGET_Y_RANGE)
        self.model.body_pos[self._target_body_id, 2] = np.random.uniform(*_TARGET_Z_RANGE)
        self._target_pos = self.model.body_pos[self._target_body_id].copy()

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

        # Reach: pull the *virtual reach target* (offset above the palm
        # surface, palm-side) to the ball, not the palm itself. This puts
        # the palm just below the ball with curled fingers wrapping from
        # the palm side — no need for the palm to dive into the table.
        reach_pos = sensordata[..., self._reach_pos_slice]
        reward = -1.0 * np.linalg.norm(reach_pos - obj, axis=-1) * 10

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
            grip_score = (ftouch * proximity * _FINGER_GRIP_WEIGHTS).sum(axis=-1)  # (...)
            grasp = np.tanh(grip_score / _GRIP_TOUCH_SCALE)                  # 0..1
            tight = np.tanh(grip_score / _TIGHT_GRIP_SCALE)                  # 0..1, slow
            # Wider Gaussian on thumb-tip → ball distance so MPPI has a
            # gradient pulling the thumb in even before contact. d2[..., 4]
            # is the squared distance from the thumb tip (last finger).
            thumb_proximity = np.exp(-d2[..., 4] / (_THUMB_PROXIMITY_SCALE ** 2))
            reward = (
                reward
                + _GRIP_BONUS * grasp                       # establishment (saturates fast)
                + _TIGHT_GRIP_BONUS * tight                 # firmness (keeps growing)
                + _THUMB_PROXIMITY_BONUS * thumb_proximity  # contact-independent pull
            )
            lift_gate = grasp * tight
        else:
            lift_gate = 1.0

        # Lift bonus + travel-to-target shaping (only when the object is up).
        lifted = obj_z > _LIFT_Z
        lift_bonus = (
            1.0
            - 0.5 * np.linalg.norm(palm - target, axis=-1)
            - 0.5 * np.linalg.norm(obj - target, axis=-1)
        ) * 4
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
        # Arm-only control magnitude penalty — pulls commanded arm positions
        # toward the rest pose. Hand controls excluded so fingers can fully
        # close during grasp.
        arm_u2 = (actions[..., :6] ** 2).sum(axis=-1)
        reward = (
            reward
            - ( _ARM_VEL_PENALTY * arm_v2
            + _FINGER_VEL_PENALTY * finger_v2
            + _ARM_CTRL_PENALTY * arm_u2 )
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
        palm_pos: np.ndarray | None = None,
    ) -> np.ndarray:
        # Object body has identity orientation, so OBJTx/y/z are world-aligned:
        #   obj_world = body_init_pos + qpos[30:33]
        obj_world = self._obj_body_init_pos + qpos[..., 30:33]
        target = np.broadcast_to(self._target_pos, obj_world.shape)

        if USE_DAPG_OBS:
            # 39-D D4RL/DAPG layout. Caller is responsible for supplying
            # palm_pos (from sensordata in the live path, or via mj_forward
            # in state_to_obs) — this method has no access to data.
            if palm_pos is None:
                raise ValueError(
                    "USE_DAPG_OBS=True requires palm_pos; "
                    "callers must populate it from sensordata or mj_forward."
                )
            return np.concatenate(
                [
                    qpos[..., :30],          # 30  arm + hand joint angles
                    palm_pos - obj_world,    # 3   hand → ball
                    palm_pos - target,       # 3   hand → target
                    obj_world - target,      # 3   ball → target
                ],
                axis=-1,
            )

        # 72-D layout (default). Ball linear velocity comes straight from the
        # slide DoFs (identity body orientation again) — no rotation needed.
        obj_vel = qvel[..., 30:33]
        return np.concatenate(
            [
                qpos[..., :30],         # 30  arm + hand joint angles
                qvel[..., :30],         # 30  arm + hand joint velocities
                obj_world,              # 3   ball world position
                obj_vel,                # 3   ball linear velocity
                target,                 # 3   target position (const per-episode)
                obj_world - target,     # 3   useful shaped feature
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
                # Hot path (MPPI prior): palm already lives in the
                # sensordata produced by batch_rollout. O(1) slice over
                # the leading dims, no kinematics.
                palm = np.asarray(sensordata)[..., self._palm_pos_slice]
            else:
                # Cold path (eval, DAgger relabel, anywhere without a
                # paired sensordata): per-state mj_kinematics fallback.
                palm = self._batched_palm_pos(qpos)
        return self._build_obs(qpos, qvel, palm_pos=palm)

    def _batched_palm_pos(self, qpos: np.ndarray) -> np.ndarray:
        """World-frame palm (S_grasp site) position for each state in qpos.

        Fallback for ``state_to_obs`` when no ``sensordata`` is supplied
        (i.e. callers outside the MPPI prior path). Uses ``mj_kinematics``
        — much cheaper than ``mj_forward`` (skips dynamics, actuation,
        and the sensor pipeline) — and reads ``site_xpos`` directly to
        bypass framepos sensor evaluation entirely. Site positions
        depend only on ``qpos``, so qvel is irrelevant.

        Still O(N) Python-level calls; acceptable for N ≲ a buffer's
        worth of states in the relabel / eval path, NOT for K×H MPPI
        inner-loop calls (those go through the sensordata fast path).
        """
        leading = qpos.shape[:-1]
        flat_q = qpos.reshape(-1, qpos.shape[-1])
        out = np.empty((flat_q.shape[0], 3), dtype=np.float64)
        # Snapshot qpos so we don't clobber the live env on a stray call
        # from outside a rollout.
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
        if USE_DAPG_OBS:
            # 30 (qpos) + 3 (palm-obj) + 3 (palm-target) + 3 (obj-target)
            return 39
        # 30 (qpos) + 30 (qvel) + 3 (obj_pos) + 3 (obj_vel) + 3 (target) + 3 (delta)
        return 72

    @property
    def noise_scale(self) -> np.ndarray:
        """Per-dim half-range so cfg.noise_sigma reads as a fraction of each
        actuator's range. Critical for relocate: arm slide ranges (0.2-0.5)
        and finger ranges (1.6-2.0) span 10x; without this, a single scalar
        sigma either slams the arm against its limits every step or leaves
        the fingers under-explored.

        Thumb actuators (last 5: A_THJ4..A_THJ0 at indices 25-29) get an
        extra boost. The thumb's rest position is folded against the palm,
        and the most proximal joint (THJ4) needs to swing ~1 rad to reach
        opposition. With baseline per-dim noise that's ~10 sigma — almost
        never sampled. Doubling lets MPPI actually discover opposable
        grasps without touching the cost.
        """
        low, high = self.action_bounds
        scale = 0.5 * (high - low)
        # scale[25:30] *= 2.0   # thumb dims: A_THJ4..A_THJ0
        return scale
