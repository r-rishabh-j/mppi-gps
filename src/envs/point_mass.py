"""2D point-mass goal-reaching task — random target each reset.

Mirrors `jaiselsingh1/mppi-gps`'s point_mass env (cost weights, obs
shape, goal workspace).

* Obs (6-D): `[qpos (2), qvel (2), goal (2)]`. Goal is an explicit
  input — the policy learns goal-conditioning from data.
* Random target sampled uniformly from `[-0.29, 0.29]²` per `reset()`,
  pushed to the mocap body for visualisation.
* Cost: `w_pos·‖qpos−goal‖² + w_vel·‖qvel‖² + w_ctrl·‖u‖²` running;
  `20·‖qpos−goal‖² + 1·‖qvel‖²` terminal.

Goal round-trip through `get_state`/`set_state`: `mjSTATE_FULLPHYSICS`
does NOT include mocap_pos, so a state captured by the parent
`MuJoCoEnv.get_state()` would lose the goal — restoring it would leave
`self._goal` stale and break the GPS C-step (conditions captured with
different goals would all restore to the env's last goal). We extend
the state vector with the goal: `get_state` appends 2 floats,
`set_state` peels them off and syncs `_goal` + `mocap_pos`, and
`batch_rollout` strips them before mujoco.rollout (which expects
FULLPHYSICS-shaped input).
"""

import mujoco
import numpy as np
from pathlib import Path
from jaxtyping import Array, Float
from numpy import ndarray

from src.envs.mujoco_env import MuJoCoEnv

_XML = str(Path(__file__).resolve().parents[2] / "assets" / "point_mass.xml")

# Target sampling box — matches upstream's `_GOAL_WORKSPACE_*`, which uses
# the full arena bounded by walls at ±0.29 so the BC dataset covers the
# whole reachable workspace (not just the easy interior).
_TARGET_LOW = -0.29
_TARGET_HIGH = 0.29
# Z stays fixed at the mocap body's XML pos (0.01 above the ground).

# Task-success criteria (used by `task_metrics`; not part of the MPPI cost).
_GOAL_RADIUS = 0.025
_SUCCESS_VEL = 0.05

# Cost weights. Identical to upstream's defaults — see file docstring.
_POS_COST_WEIGHT = 1.0
_VEL_COST_WEIGHT = 0.05
_TERMINAL_POS_COST_WEIGHT = 20.0
_TERMINAL_VEL_COST_WEIGHT = 1.0


class PointMass(MuJoCoEnv):
    def __init__(
        self,
        frame_skip: int = 1,
        ctrl_cost_weight: float = 0.01,
        **kwargs,
    ) -> None:
        super().__init__(model_path=_XML, frame_skip=frame_skip, **kwargs)
        assert self.model.nmocap == 1, (
            "point_mass.xml must expose target_marker as mocap (set per-reset)"
        )
        self._ctrl_w = float(ctrl_cost_weight)
        self._nq = self.model.nq
        self._nv = self.model.nv
        # Goal in [x, y]. Constant across one MPPI rollout (one plan_step
        # = one goal), so no per-state sensor needed.
        self._goal = np.zeros(2, dtype=np.float64)

    # ------------------------------------------------------------------
    # Reset / step
    # ------------------------------------------------------------------

    def _sample_goal(self) -> np.ndarray:
        return np.random.uniform(_TARGET_LOW, _TARGET_HIGH, size=2).astype(
            np.float64
        )

    def reset(
        self,
        state: Float[ndarray, "state_dim"] | None = None,
    ) -> Float[ndarray, "4"]:
        if state is not None:
            # Parent reset calls our `set_state`, which peels the goal
            # off the state tail and syncs `_goal` + mocap_pos.
            return super().reset(state=state)

        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = np.random.uniform(-0.25, 0.25, size=self._nq)
        self.data.qvel[:] = np.random.normal(0.0, 0.1, size=self._nv)
        self._goal = self._sample_goal()
        # Keep the mocap body's z from its XML pose; only move x, y.
        self.data.mocap_pos[0, :2] = self._goal
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs()

    # State capture/restore extended with goal. See module docstring.

    def get_state(self) -> np.ndarray:
        physics = super().get_state()                 # FULLPHYSICS-size
        return np.concatenate([physics, self._goal])

    def set_state(self, state: np.ndarray) -> None:
        # Accept both extended (state_dim) and raw FULLPHYSICS — legacy
        # callers pass plain states (goal cache stays as-is).
        if state.shape[-1] == self._nstate + 2:
            super().set_state(state[: self._nstate])
            goal = state[self._nstate : self._nstate + 2]
            self._goal = np.asarray(goal, dtype=np.float64).copy()
            self.data.mocap_pos[0, :2] = self._goal
        else:
            super().set_state(state)

    @property
    def state_dim(self) -> int:
        return self._nstate + 2

    def batch_rollout(
        self,
        initial_state: np.ndarray,
        action_sequences: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Strip the trailing goal entries before mujoco.rollout (expects
        # FULLPHYSICS-shaped input). Set `_goal` so the cost / state_to_obs
        # see the goal that was active when this state was captured. MPPI
        # always broadcasts ONE state per call, so one goal per call.
        initial_state = np.asarray(initial_state)
        if initial_state.shape[-1] == self._nstate + 2:
            if initial_state.ndim == 1:
                goal = initial_state[self._nstate : self._nstate + 2]
            else:
                goal = initial_state[0, self._nstate : self._nstate + 2]
            self._goal = np.asarray(goal, dtype=np.float64).copy()
            self.data.mocap_pos[0, :2] = self._goal
            initial_state = initial_state[..., : self._nstate]
        return super().batch_rollout(initial_state, action_sequences)

    # ------------------------------------------------------------------
    # Task metrics — used by tuning + eval to report success rate
    # ------------------------------------------------------------------

    def task_metrics(self) -> dict:
        pos = self.data.qpos.copy()
        vel = self.data.qvel.copy()
        dist = float(np.linalg.norm(pos - self._goal))
        qvel_norm = float(np.linalg.norm(vel))
        return {
            # Upstream `gps_train.py` reads `tip_dist`; we keep that key.
            "tip_dist": dist,
            "goal_dist": dist,
            "qvel_norm": qvel_norm,
            "success": bool(dist <= _GOAL_RADIUS and qvel_norm <= _SUCCESS_VEL),
            "x_pos": float(pos[0]),
            "y_pos": float(pos[1]),
            "goal_x": float(self._goal[0]),
            "goal_y": float(self._goal[1]),
            "target_cost": float(np.sum((pos - self._goal) ** 2)),
        }

    # ------------------------------------------------------------------
    # Cost (vectorised over MPPI's K × H rollout grid)
    # ------------------------------------------------------------------

    def running_cost(
        self,
        states: Float[Array, "K H nstate"],
        actions: Float[Array, "K H nu"],
        sensordata: Float[Array, "K H nsensor"] | None = None,
    ) -> Float[Array, "K H"]:
        qpos = self.state_qpos(states)
        qvel = self.state_qvel(states)
        pos_err = qpos - self._goal[None, None, :]
        pos_cost = _POS_COST_WEIGHT * np.sum(pos_err ** 2, axis=-1)
        vel_cost = _VEL_COST_WEIGHT * np.sum(qvel ** 2, axis=-1)
        ctrl_cost = self._ctrl_w * np.sum(actions ** 2, axis=-1)
        return pos_cost + vel_cost + ctrl_cost

    def terminal_cost(
        self,
        states: Float[Array, "K nstate"],
        sensordata: Float[Array, "K nsensor"] | None = None,
    ) -> Float[Array, "K"]:
        qpos = self.state_qpos(states)
        qvel = self.state_qvel(states)
        pos_err = qpos - self._goal[None, :]
        return (
            _TERMINAL_POS_COST_WEIGHT * np.sum(pos_err ** 2, axis=-1)
            + _TERMINAL_VEL_COST_WEIGHT * np.sum(qvel ** 2, axis=-1)
        )

    # Observation: `[qpos, qvel, goal]` (6-D). Explicit goal channel
    # (rather than delta-to-goal) lets the policy learn the relationship
    # from data and avoids label noise from wall-proximity effects.

    def _get_obs(self) -> Float[ndarray, "6"]:
        return np.concatenate([self.data.qpos, self.data.qvel, self._goal])

    def state_to_obs(
        self,
        states: np.ndarray,
        sensordata: np.ndarray | None = None,
    ) -> np.ndarray:
        qpos = self.state_qpos(states)
        qvel = self.state_qvel(states)
        # One MPPI rollout uses one goal — broadcast `_goal` across the
        # leading dims (K, H, ...).
        goal = np.broadcast_to(self._goal, qpos.shape[:-1] + (2,))
        return np.concatenate([qpos, qvel, goal], axis=-1)

    @property
    def obs_dim(self) -> int:
        return self._nq + self._nv + 2
