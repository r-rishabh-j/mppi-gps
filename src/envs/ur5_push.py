"""UR5 push task — adapted from `jaiselsingh1/ur5_research`.

Mirrors upstream's MPPI path (`mppi_ur5.py` + `ur5_env.py`) for the hot
loop: 6-D ctrl into the C-side `mujoco.rollout` (no Python in the
sample/rollout pipeline). The action space exposed to MPPI is **3-D
joint velocities** (shoulder_pan, shoulder_lift, elbow); the env
internally pads with the wrist setpoints `[init_wrist_1, init_wrist_2,
init_wrist_3]` before mujoco rollout sees the ctrl.

Why the wrist controllers were changed
--------------------------------------
Upstream's XML uses velocity actuators on all 6 joints. The "upright EE"
behaviour visible in their PPO eval comes from a Cartesian-IK controller
that runs only in `ur5_push_env.step()` for policy execution, **not in
their MPPI**. Their MPPI gets away without orientation enforcement only
because they use tiny exploration (`K=5, σ=0.2`) — wrists barely move.

With our larger MPPI noise (K=256, σ ~0.5–0.8), velocity-controlled
wrists drift visibly under sampled actions + contact disturbance. A
soft orientation cost penalty in MPPI's running cost can reduce this
but cannot eliminate it (it competes with reach + push terms and the
noise floor). The clean fix is a **hard kinematic constraint**: switch
the wrist actuators to position controllers (PD servos) in the XML.
The wrists then restore to their setpoint after any disturbance, the
EE stays vertical to within mujoco's solver tolerance, and the hot
loop stays at full `mujoco.rollout` speed.

State / observation layout
--------------------------
* ``model.nq = 13``  — 6 UR5 joint angles + 7 tape free-joint.
* ``model.nv = 12``  — 6 UR5 joint velocities + 6 tape (lin + ang).
* ``model.nu = 6``   — 3 velocity (shoulder+elbow) + 3 position (wrist) actuators.
* Policy obs (34-D):
  ``[qpos (13), qvel (12), ee−tape (3), tape−target (3), ee−target (3)]``

Cost (matches upstream `mppi_ur5.py` verbatim)
----------------------------------------------
``running_cost``  = ``w_reach · ‖ee − tape‖²``
``terminal_cost`` = ``w_term  · ‖tape − target‖``

No orientation term needed — the position-controlled wrists keep the
EE downward by construction.
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
# Wrist setpoints sent to the position actuators (ctrl[3:6]) every step.
# Slice of `_INIT_QPOS_UR5` corresponding to wrist_1 / wrist_2 / wrist_3.
_WRIST_SETPOINTS = _INIT_QPOS_UR5[3:6]

# Active actuator count. The first 3 actuators (velocity, shoulder + elbow)
# are exposed to MPPI / policy; the remaining 3 (position, wrist) are locked
# to the init wrist setpoints by the env. Wrist position controllers hold
# the wrist joints rigidly; the EE's downward orientation depends additionally
# on the cumulative pitch of shoulder_lift + elbow + wrist_1 + wrist_3, so
# the policy can still tilt the EE by pitching shoulder/elbow. The
# orientation cost in `running_cost` shapes MPPI away from those samples.
_ACTIVE_NU = 3

# Target sampling box on the table.
_TARGET_X_RANGE = (0.35, 0.65)
_TARGET_Y_RANGE = (-0.4, 0.4)
_TARGET_Z = -0.1175

# Cost weights.
_W_REACH = 10.0           # running, ‖ee − tape‖² — matches upstream. Squared
                          # form gives strong gradient far from tape but
                          # decays to ~0 at contact, so we add a linear
                          # term below to keep pulling EE in close-up.
_W_REACH_LINEAR = 5.0     # running, ‖ee − tape‖ (linear). Keeps the reach
                          # gradient alive when squared form decays. Without
                          # this, MPPI loses interest in approaching tape
                          # once "near enough" — it idles at the boundary
                          # instead of committing to contact.
_W_TERMINAL = 1000.0       # terminal, ‖tape − target‖. Bumped from upstream's
                          # 10 (and our prior 50) because at terminal=50 with
                          # target ~0.05m away, the push signal was only
                          # ~2.78 mean cost — 15% of total — completely
                          # drowned by reach (~12) + orient (~4). MPPI
                          # optimised "stay near tape, don't tilt" and
                          # barely cared about pushing. With weight 500, a
                          # 0.05m target distance contributes ~25 — now
                          # dominant once EE is in contact. If MPPI still
                          # doesn't push, bump higher (1000–2000).
_W_ORIENT = 10.0          # running, (1 + cyl_z) ∈ [0, 2]. Reduced from 50
                          # because orient and reach FIGHT: keeping cyl_z=-1
                          # requires shoulder_lift + elbow near init, but
                          # reaching the tape requires shoulder_lift to
                          # dive (which tilts the EE). The position-locked
                          # wrists handle MOST of the orientation work; we
                          # only need a small cost nudge against tilt-
                          # inducing samples. 10 keeps tilt under ~5° in
                          # MPPI rollouts without strangling reach.

_SUCCESS_DIST = 0.05


class UR5Push(MuJoCoEnv):
    def __init__(self, frame_skip: int = 20, **kwargs) -> None:
        # Default frame_skip=20 — 0.04s of sim time per env step. Upstream
        # `ur5_env.py` uses 40 (`ur5_push_env.py` uses 50). At our prior
        # default of 5 (0.01s/step), the velocity actuators barely had
        # time to overcome gravity + arm inertia before MPPI replanned,
        # and the planner's horizon (H × frame_skip × dt) covered only
        # ~0.16s — too short to plan a coherent descent + push sequence.
        # 20 is a compromise between full upstream realism and our
        # need for interactive plan-step times (~170ms vs 360ms at 40).
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
    # Action interface — 3-D active dims (shoulder + elbow joint velocities)
    # ------------------------------------------------------------------

    @property
    def action_dim(self) -> int:
        return _ACTIVE_NU

    @property
    def action_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        # First 3 actuators of the model (velocity controllers, ±3.15).
        low, high = super().action_bounds
        return low[:_ACTIVE_NU], high[:_ACTIVE_NU]

    @property
    def noise_scale(self) -> np.ndarray:
        """Per-dim half-range so ``cfg.noise_sigma`` reads as a fraction
        of each actuator's ctrlrange — matches the pattern in
        ``adroit_relocate.noise_scale`` and the upstream convention
        where ``act_rng = 3.15`` and the policy/MPPI works in [-1, 1]
        normalized space.

        Without this override, ``noise_scale = ones(2)`` and a config
        ``noise_sigma = 0.8`` literally meant 0.8 rad/s of MPPI noise
        — only ~25% of our ±3.15 action range. With this override,
        ``σ=0.8`` becomes ``0.8 × 3.15 = 2.52 rad/s`` (~80% of range)
        — actually aggressive exploration. ``σ=0.2`` gives ~0.63 rad/s
        which roughly matches upstream's effective MPPI noise level.
        """
        low, high = self.action_bounds
        return 0.5 * (high - low)

    def _expand_action(self, action: np.ndarray) -> np.ndarray:
        """Pad ``(..., 3)`` action with the wrist position setpoints to
        ``(..., 6)``. The first 3 dims are velocity ctrl on shoulder_pan
        / shoulder_lift / elbow (active MPPI dims); the last 3 dims are
        position setpoints on wrist_1 / wrist_2 / wrist_3 (held). The
        EE's downward orientation is shaped by the orientation cost in
        running_cost — wrists alone don't enforce it. Idempotent on
        already-full-width inputs."""
        action = np.asarray(action, dtype=np.float64)
        if action.shape[-1] == self.model.nu:
            return action
        if action.shape[-1] != _ACTIVE_NU:
            raise ValueError(
                f"expected action shape ending in {_ACTIVE_NU} or "
                f"{self.model.nu}; got {action.shape}"
            )
        leading = action.shape[:-1]
        wrist = np.broadcast_to(_WRIST_SETPOINTS, leading + (3,))
        return np.concatenate([action, wrist], axis=-1)

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
    # Reset / step (parent's batch_rollout = mujoco.rollout C path)
    # ------------------------------------------------------------------

    def reset(self, state: np.ndarray | None = None) -> np.ndarray:
        if state is not None:
            # State-restore path. Parent calls our overridden `set_state`,
            # which unpacks the trailing target_pos entries from the
            # extended state vector and syncs `self._target_pos` + mocap.
            # See module-level note on get_state/set_state below.
            return super().reset(state=state)

        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:6] = _INIT_QPOS_UR5
        # Initialise ctrl[3:6] to the wrist setpoints so the position
        # actuators apply zero error at t=0 (no startup transient).
        self.data.ctrl[3:] = _WRIST_SETPOINTS
        self._target_pos = self._sample_target()
        self.data.mocap_pos[0] = self._target_pos
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs()

    # ------------------------------------------------------------------
    # State capture/restore — extended with target_pos so condition-
    # restore round-trips through MPPI / GPS / DAgger correctly.
    #
    # MuJoCo's `mjSTATE_FULLPHYSICS` in our version does NOT include
    # `mjSTATE_MOCAP_POS`, so the captured state alone doesn't carry the
    # target marker. Without this override, GPS C-step `env.reset(state=
    # ic_state)` would leave `self._target_pos` stale (last value sampled
    # by an in-the-clear `env.reset()`) and mocap reset to its XML pose,
    # so every (obs, action) in n−1 of n conditions trained on the wrong
    # target. Same bug as PointMass — see that env's docstring.
    # ------------------------------------------------------------------

    def get_state(self) -> np.ndarray:
        physics = super().get_state()                       # FULLPHYSICS
        return np.concatenate([physics, self._target_pos])  # +3 target_xyz

    def set_state(self, state: np.ndarray) -> None:
        # Accept extended (state_dim) and raw (FULLPHYSICS) inputs so any
        # legacy caller still works (target cache unchanged in the raw
        # branch — same as prior behaviour).
        if state.shape[-1] == self._nstate + 3:
            super().set_state(state[: self._nstate])
            tgt = state[self._nstate : self._nstate + 3]
            self._target_pos = np.asarray(tgt, dtype=np.float64).copy()
            self.data.mocap_pos[0] = self._target_pos
        else:
            super().set_state(state)

    @property
    def state_dim(self) -> int:
        return self._nstate + 3

    def batch_rollout(
        self,
        initial_state: np.ndarray,
        action_sequences: np.ndarray,         # (K, H, 3)
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Strip the trailing target_pos from the captured state before
        # handing off to mujoco.rollout (which expects FULLPHYSICS shape),
        # then broadcast it into self._target_pos for running_cost /
        # terminal_cost. One MPPI call has one target, so all K rollouts
        # share the same target — taking the first row is safe.
        initial_state = np.asarray(initial_state)
        if initial_state.shape[-1] == self._nstate + 3:
            if initial_state.ndim == 1:
                tgt = initial_state[self._nstate : self._nstate + 3]
            else:
                tgt = initial_state[0, self._nstate : self._nstate + 3]
            self._target_pos = np.asarray(tgt, dtype=np.float64).copy()
            self.data.mocap_pos[0] = self._target_pos
            initial_state = initial_state[..., : self._nstate]
        return super().batch_rollout(
            initial_state, self._expand_action(action_sequences),
        )

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        action = np.clip(action, self.action_bounds[0], self.action_bounds[1])
        action_full = self._expand_action(action)
        obs, cost_arr, done, info = super().step(action_full)

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
    # Cost (matches upstream `mppi_ur5.py` verbatim — no orientation term;
    # wrist position controllers handle EE upright by construction)
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

        # Orientation penalty. mujoco quat is wxyz; cylinder z-axis in
        # world is the third column of R(quat), with closed form
        # cyl_z = 1 − 2(qx² + qy²). Penalty = 1 + cyl_z ∈ [0, 2]:
        # 0 when EE points perfectly down; 2 when fully inverted.
        ee_quat = sensordata[..., self._ee_quat_slice]
        qx = ee_quat[..., 1]
        qy = ee_quat[..., 2]
        cyl_z = 1.0 - 2.0 * (qx ** 2 + qy ** 2)
        orient_penalty = 1.0 + cyl_z

        return (
            _W_REACH * ee_to_tape ** 2
            + _W_REACH_LINEAR * ee_to_tape
            + _W_ORIENT * orient_penalty
        )

    def terminal_cost(
        self,
        states: Float[Array, "K nstate"],
        sensordata: Float[Array, "K nsensor"] | None = None,
    ) -> Float[Array, "K"]:
        tape = sensordata[..., self._tape_pos_slice]
        return _W_TERMINAL * np.linalg.norm(tape - self._target_pos, axis=-1)
