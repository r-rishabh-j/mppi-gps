"""Adroit Pen reorientation task — vendored from gymnasium-robotics.

24-DoF Shadow Hand reposing a free pen to match a randomly-sampled target
orientation. The dense reward of the upstream gymnasium-robotics env
(``AdroitHandPen-v1``) is mirrored here, negated to act as an MPPI cost.

State / observation layout
--------------------------
* ``model.nq = model.nv = 30`` — 24 hand joints + 6 pen joints
  (3 slide ``OBJTx/OBJTy/OBJTz`` + 3 hinge ``OBJRx/OBJRy/OBJRz``).
* Policy obs is 42-D, derivable from ``(qpos, qvel)`` plus the per-episode
  target attributes — so the same composition works in ``_get_obs`` (live)
  and ``state_to_obs`` (batched, called by GPS/DAgger relabel).
* Pen orientation is intentionally NOT included in obs as a unit vector
  (which would need ``mj_forward`` per state). Instead we pass the three
  pen rotation joint angles ``qpos[27:30]`` plus the target Euler angles —
  fully sufficient for the policy and free to compute.

Cost
----
``running_cost`` is a verbatim translation of the upstream dense reward
(``-goal_dist + orien_sim`` with milestone bonuses and a drop penalty),
negated so MPPI minimises. Orientation is computed from the
``object_top_pos`` / ``object_bottom_pos`` ``framepos`` sensors added to
``assets/adroit_pen/adroit_pen.xml``; the pen's world position is the
midpoint of those two sites (verified to equal ``data.xpos[Object]``).
"""

from pathlib import Path

import mujoco
import numpy as np
from jaxtyping import Array, Float

from src.envs.mujoco_env import MuJoCoEnv

_XML = str(Path(__file__).resolve().parents[2] / "assets" / "adroit_pen" / "adroit_pen.xml")

# Fixed target position (the eps_ball site). Target orientation is sampled
# at every reset; only the position is constant.
_DESIRED_LOC = np.array([0.0, -0.2, 0.25])
# Pen body's initial world position from the XML (used only to express
# the policy's position-error obs without needing a forward pass).
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

        # Sensor offsets — set once at construction, used in both step and
        # batched cost paths.
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

        # Initial target orientation (identity quat) — overwritten on reset.
        self._desired_orien_euler = np.zeros(3, dtype=np.float64)
        self._desired_orien = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        # Pen length used for normalising orientation vectors; matches the
        # upstream env which recomputes from sites at each reset.
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

    # ------------------------------------------------------------------
    # Reset / step
    # ------------------------------------------------------------------

    def reset(self, state: np.ndarray | None = None) -> np.ndarray:
        if state is not None:
            # Deterministic restore (used by GPS to replay initial conditions).
            return super().reset(state=state)

        mujoco.mj_resetData(self.model, self.data)

        # Sample a target orientation: roll, pitch ~ U(-1, 1), yaw = 0.
        self._desired_orien_euler = np.array(
            [np.random.uniform(-1, 1), np.random.uniform(-1, 1), 0.0]
        )
        self.model.body_quat[self._target_body_id] = _euler_to_quat(
            self._desired_orien_euler
        )
        mujoco.mj_forward(self.model, self.data)

        # Capture the realised target orientation vector (top - bottom)/length.
        tar_top = self.data.site_xpos[self._tar_top_id].copy()
        tar_bot = self.data.site_xpos[self._tar_bot_id].copy()
        tar_len = float(np.linalg.norm(tar_top - tar_bot))
        self._desired_orien = (tar_top - tar_bot) / tar_len
        return self._get_obs()

    # ------------------------------------------------------------------
    # Cost
    # ------------------------------------------------------------------

    def _running_reward_terms(
        self,
        sensordata: Float[Array, "... nsensor"],
    ) -> tuple[Float[Array, "..."], Float[Array, "..."], Float[Array, "..."]]:
        """Return (goal_distance, orien_similarity, obj_pos_z) over arbitrary
        leading dims. Used by both the vectorised MPPI cost and the live
        ``step`` cost via the same sensordata layout."""
        top = sensordata[..., self._top_slice]
        bot = sensordata[..., self._bot_slice]
        obj_pos = (top + bot) * 0.5                       # midpoint == xpos[Object]
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
        """Verbatim translation of the upstream dense reward, negated.

        ``reward = -goal_distance + orien_similarity
                   + 10 * 1[goal<0.075 ∧ orien>0.9]
                   + 50 * 1[goal<0.075 ∧ orien>0.95]
                   - 5  * 1[obj_z<0.075]``

        Cost is ``-reward``. The bonus/drop terms are bounded indicators —
        they don't break MPPI's importance weighting because the smooth
        ``-goal_dist + orien_sim`` term provides the gradient that
        importance-weighting actually rides.
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
        # Dense running cost already drives behaviour; no extra terminal term.
        return np.zeros(states.shape[0])

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        """Single env step. Terminates on success (matches upstream)."""
        obs, cost, _, info = super().step(action)
        # Sensordata is the latest after the frame_skip loop in MuJoCoEnv.step.
        sd = self.data.sensordata
        goal_distance, orien_similarity, _ = self._running_reward_terms(sd)
        success = bool(
            (goal_distance < _SUCCESS_DIST)
            and (orien_similarity > _SUCCESS_ORIEN_TIGHT)
        )
        info["success"] = success
        return obs, cost, success, info

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def _build_obs(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
    ) -> np.ndarray:
        """42-D obs from qpos/qvel + per-episode target attrs.

        Layout (last axis):
          [0:24)   hand joint qpos
          [24:27)  pen world position (init body pos + slide qpos)
                   NOTE: this is exact only when the pen's body initial
                   euler matches the XML default — it does, see XML.
                   The slide joints are in body frame; we transform them
                   by the body's initial rotation to get world coords.
          [27:33)  pen lin+ang vel (qvel[24:30])
          [33:36)  pen rotation joint angles (qpos[27:30])
          [36:39)  pen position - desired_loc
          [39:42)  desired target orientation Euler (constant per episode)
        """
        # The body frame is rotated by euler="0 1.57 0" (R_y(π/2)) so
        # body axes (x, y, z) map to world (-z, y, x). The slide joints
        # OBJTx/OBJTy/OBJTz are along body axes, so:
        #   pen_world = body_init_pos + R_body @ slide_qpos
        # with R_body @ (sx, sy, sz) = (sz, sy, -sx).
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

    def state_to_obs(self, states: np.ndarray) -> np.ndarray:
        qpos = self.state_qpos(states)
        qvel = self.state_qvel(states)
        return self._build_obs(qpos, qvel)

    @property
    def obs_dim(self) -> int:
        # 24 + 3 + 6 + 3 + 3 + 3 = 42
        return 42
