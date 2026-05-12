"""UR5 push task (adapted from jaiselsingh1/ur5_research).

Action exposed to MPPI is 3-D joint velocities (shoulder_pan,
shoulder_lift, elbow). The env pads with ``_WRIST_SETPOINTS`` for the
3 position-controlled wrist actuators before ``mujoco.rollout``.

Layout: nq=13 (6 UR5 + 7 tape free joint), nv=12 (6 UR5 + 6 tape),
nu=6 (3 vel + 3 pos). Obs is 34-D:
``[qpos (13), qvel (12), ee−tape (3), tape−target (3), ee−target (3)]``.

Cost: reach (squared + linear) + orientation + terminal push.
"""

from pathlib import Path

import mujoco
import numpy as np
from jaxtyping import Array, Float
from numpy import ndarray

from src.envs.mujoco_env import MuJoCoEnv

_XML = str(Path(__file__).resolve().parents[2] / "assets" / "ur5" / "scene.xml")

# Initial UR5 pose: EE finger axis points straight down (world -z),
# tip ~0.18m above the tape.
_INIT_QPOS_UR5 = np.array(
    [0.0, -np.pi / 3, 2 * np.pi / 3, -np.pi / 3 - np.pi / 2, -np.pi / 2, 0.0],
    dtype=np.float64,
)
# Wrist setpoints sent to the position actuators (ctrl[3:6]) every step.
_WRIST_SETPOINTS = _INIT_QPOS_UR5[3:6]

# Number of active (MPPI-controlled) actuators; the rest are wrist holds.
_ACTIVE_NU = 3

# Target sampling box on the table.
_TARGET_X_RANGE = (0.35, 0.65)
_TARGET_Y_RANGE = (-0.4, 0.4)
_TARGET_Z = -0.1175

# Cost weights. Squared reach gives gradient far away; linear keeps it
# alive at contact. Orient uses (1 + cyl_z) ∈ [0, 2].
_W_REACH = 10.0
_W_REACH_LINEAR = 5.0
_W_TERMINAL = 1000.0
_W_ORIENT = 10.0

_SUCCESS_DIST = 0.05


class UR5Push(MuJoCoEnv):
    def __init__(self, frame_skip: int = 20, **kwargs) -> None:
        # frame_skip=20 → 0.04 s/step. Compromise between upstream's 40
        # (~360 ms plan-step) and interactive replanning (~170 ms).
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

    @property
    def action_dim(self) -> int:
        return _ACTIVE_NU

    @property
    def action_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        # First 3 actuators (velocity controllers, ±3.15).
        low, high = super().action_bounds
        return low[:_ACTIVE_NU], high[:_ACTIVE_NU]

    @property
    def noise_scale(self) -> np.ndarray:
        """Per-dim half-range — cfg.noise_sigma reads as a fraction of ctrlrange."""
        low, high = self.action_bounds
        return 0.5 * (high - low)

    def _expand_action(self, action: np.ndarray) -> np.ndarray:
        """Pad ``(..., 3)`` to ``(..., 6)`` with wrist setpoints. Idempotent."""
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

    def reset(self, state: np.ndarray | None = None) -> np.ndarray:
        if state is not None:
            return super().reset(state=state)

        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:6] = _INIT_QPOS_UR5
        # Wrist position actuators start at zero error (no t=0 transient).
        self.data.ctrl[3:] = _WRIST_SETPOINTS
        self._target_pos = self._sample_target()
        self.data.mocap_pos[0] = self._target_pos
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs()

    # State extended with target_pos: mjSTATE_FULLPHYSICS does NOT include
    # mocap_pos, so without this GPS C-step state-restore would leave
    # _target_pos stale (same pattern as PointMass).

    def get_state(self) -> np.ndarray:
        physics = super().get_state()
        return np.concatenate([physics, self._target_pos])

    def set_state(self, state: np.ndarray) -> None:
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
        # Strip the trailing target before mujoco.rollout (FULLPHYSICS
        # only) and sync `_target_pos` for the cost.
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

    def running_cost(
        self,
        states: Float[Array, "K H nstate"],
        actions: Float[Array, "K H nu"],
        sensordata: Float[Array, "K H nsensor"] | None = None,
    ) -> Float[Array, "K H"]:
        ee = sensordata[..., self._ee_pos_slice]
        tape = sensordata[..., self._tape_pos_slice]
        ee_to_tape = np.linalg.norm(ee - tape, axis=-1)

        # Orientation penalty: cyl_z (third col of R(quat)) closed form
        # 1 − 2(qx² + qy²). Penalty 1 + cyl_z ∈ [0, 2]; 0 = EE down.
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
