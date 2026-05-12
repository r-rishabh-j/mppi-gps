"""Adroit Pen reorientation (24-DoF Shadow Hand, gymnasium-robotics).

Reposes a free pen to a randomly-sampled target orientation. The
upstream ``AdroitHandPen-v1`` dense reward is mirrored here, negated for
MPPI. nq=nv=30 (24 hand + 6 pen joints); 42-D obs.

Pen position/orientation come from the ``object_top_pos`` /
``object_bottom_pos`` framepos sensors (added in
``assets/adroit/pen.xml``); position is their midpoint, orientation is
``(top - bot) / pen_length``.
"""

from pathlib import Path

import mujoco
import numpy as np
from jaxtyping import Array, Float

from src.envs.mujoco_env import MuJoCoEnv

_XML = str(Path(__file__).resolve().parents[2] / "assets" / "adroit" / "pen.xml")

# Fixed target position (eps_ball site). Orientation is resampled per reset.
_DESIRED_LOC = np.array([0.0, -0.2, 0.25])
# Pen body's XML initial world position (for obs without mj_forward).
_OBJ_BODY_INIT_POS = np.array([0.0, -0.2, 0.25])

_DROP_THRESHOLD_Z = 0.075
_SUCCESS_DIST = 0.075
_SUCCESS_ORIEN_LOOSE = 0.9
_SUCCESS_ORIEN_TIGHT = 0.95


def _euler_to_quat(euler: np.ndarray) -> np.ndarray:
    """XYZ Euler -> wxyz quaternion via mujoco's helper."""
    q = np.zeros(4, dtype=np.float64)
    mujoco.mju_euler2Quat(q, np.asarray(euler, dtype=np.float64), "XYZ")
    return q


class AdroitPen(MuJoCoEnv):
    def __init__(self, frame_skip: int = 5, **kwargs) -> None:
        super().__init__(model_path=_XML, frame_skip=frame_skip, **kwargs)
        assert self.model.nq == 30 and self.model.nv == 30 and self.model.nu == 24

        self._top_slice = self._sensor_slice("object_top_pos")
        self._bot_slice = self._sensor_slice("object_bottom_pos")

        self._target_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "target"
        )
        self._tar_top_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "target_top"
        )
        self._tar_bot_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "target_bottom"
        )
        self._obj_top_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "object_top"
        )
        self._obj_bot_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "object_bottom"
        )

        # Overwritten on every reset.
        self._desired_orien_euler = np.zeros(3, dtype=np.float64)
        self._desired_orien = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        # Pen length, used to normalise orientation vectors.
        mujoco.mj_forward(self.model, self.data)
        self._pen_length = float(np.linalg.norm(
            self.data.site_xpos[self._obj_top_id]
            - self.data.site_xpos[self._obj_bot_id]
        ))

    def _sensor_slice(self, name: str) -> slice:
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        adr = int(self.model.sensor_adr[sid])
        dim = int(self.model.sensor_dim[sid])
        return slice(adr, adr + dim)

    def reset(self, state: np.ndarray | None = None) -> np.ndarray:
        if state is not None:
            return super().reset(state=state)

        mujoco.mj_resetData(self.model, self.data)

        # Sample target orientation: roll, pitch ~ U(-1, 1), yaw = 0.
        self._desired_orien_euler = np.array(
            [np.random.uniform(-1, 1), np.random.uniform(-1, 1), 0.0]
        )
        self.model.body_quat[self._target_body_id] = _euler_to_quat(
            self._desired_orien_euler
        )
        mujoco.mj_forward(self.model, self.data)

        tar_top = self.data.site_xpos[self._tar_top_id].copy()
        tar_bot = self.data.site_xpos[self._tar_bot_id].copy()
        tar_len = float(np.linalg.norm(tar_top - tar_bot))
        self._desired_orien = (tar_top - tar_bot) / tar_len
        return self._get_obs()

    def _running_reward_terms(
        self,
        sensordata: Float[Array, "... nsensor"],
    ) -> tuple[Float[Array, "..."], Float[Array, "..."], Float[Array, "..."]]:
        """Return (goal_distance, orien_similarity, obj_pos_z)."""
        top = sensordata[..., self._top_slice]
        bot = sensordata[..., self._bot_slice]
        obj_pos = (top + bot) * 0.5
        obj_orien = (top - bot) / self._pen_length

        goal_distance = np.linalg.norm(obj_pos - _DESIRED_LOC, axis=-1)
        orien_similarity = np.einsum("...i,i->...", obj_orien, self._desired_orien)
        return goal_distance, orien_similarity, obj_pos[..., 2]

    def running_cost(
        self,
        states: Float[Array, "K H nstate"],
        actions: Float[Array, "K H nu"],
        sensordata: Float[Array, "K H nsensor"] | None = None,
    ) -> Float[Array, "K H"]:
        """Negated upstream dense reward.

        reward = -goal_dist + orien_sim
                 + 10·1[goal<0.075 ∧ orien>0.9]
                 + 50·1[goal<0.075 ∧ orien>0.95]
                 -  5·1[obj_z<0.075]
        """
        if sensordata is None:
            raise ValueError(
                "AdroitPen.running_cost requires sensordata "
                "(object endpoint sites)."
            )

        goal_distance, orien_similarity, obj_z = self._running_reward_terms(
            sensordata
        )
        reward = -goal_distance + orien_similarity
        loose = (goal_distance < _SUCCESS_DIST) & (orien_similarity > _SUCCESS_ORIEN_LOOSE)
        tight = (goal_distance < _SUCCESS_DIST) & (orien_similarity > _SUCCESS_ORIEN_TIGHT)
        reward = reward + np.where(loose, 10.0, 0.0)
        reward = reward + np.where(tight, 50.0, 0.0)
        reward = reward - np.where(obj_z < _DROP_THRESHOLD_Z, 5.0, 0.0)
        return -reward

    def terminal_cost(
        self,
        states: Float[Array, "K nstate"],
        sensordata: Float[Array, "K nsensor"] | None = None,
    ) -> Float[Array, "K"]:
        return np.zeros(states.shape[0])

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        """Step; terminates on success (matches upstream)."""
        obs, cost, _, info = super().step(action)
        sd = self.data.sensordata
        goal_distance, orien_similarity, _ = self._running_reward_terms(sd)
        success = bool(
            (goal_distance < _SUCCESS_DIST)
            and (orien_similarity > _SUCCESS_ORIEN_TIGHT)
        )
        info["success"] = success
        return obs, cost, success, info

    def _build_obs(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
    ) -> np.ndarray:
        """42-D obs.

        [0:24)  hand qpos
        [24:27) pen world pos = init_body_pos + R_body·slide_qpos
        [27:33) pen lin+ang vel (qvel[24:30])
        [33:36) pen rotation angles (qpos[27:30])
        [36:39) pen pos - desired_loc
        [39:42) desired orientation Euler

        Pen body has XML euler="0 1.57 0" (R_y(π/2)), so body slide
        (sx, sy, sz) maps to world (sz, sy, -sx).
        """
        slide = qpos[..., 24:27]
        rot_offset = np.stack(
            [slide[..., 2], slide[..., 1], -slide[..., 0]],
            axis=-1,
        )
        pen_world = _OBJ_BODY_INIT_POS + rot_offset

        return np.concatenate(
            [
                qpos[..., :24],
                pen_world,
                qvel[..., 24:30],
                qpos[..., 27:30],
                pen_world - _DESIRED_LOC,
                np.broadcast_to(self._desired_orien_euler, (*qpos.shape[:-1], 3)),
            ],
            axis=-1,
        )

    def _get_obs(self) -> np.ndarray:
        return self._build_obs(self.data.qpos.copy(), self.data.qvel.copy())

    def state_to_obs(
        self,
        states: np.ndarray,
        sensordata: np.ndarray | None = None,
    ) -> np.ndarray:
        qpos = self.state_qpos(states)
        qvel = self.state_qvel(states)
        return self._build_obs(qpos, qvel)

    @property
    def obs_dim(self) -> int:
        return 42

    @property
    def noise_scale(self) -> np.ndarray:
        """Per-dim half-range — actuator ctrlranges are heterogeneous."""
        low, high = self.action_bounds
        return 0.5 * (high - low)
