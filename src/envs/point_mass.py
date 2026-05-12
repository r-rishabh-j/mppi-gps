"""2D point-mass goal-reaching task with random per-reset target.

Obs (6-D): ``[qpos (2), qvel (2), goal (2)]``.
Cost: ``w_pos·‖qpos−goal‖² + w_vel·‖qvel‖² + w_ctrl·‖u‖²`` running;
``20·‖qpos−goal‖² + 1·‖qvel‖²`` terminal.

The state vector is extended with the goal (2 floats appended) because
``mjSTATE_FULLPHYSICS`` doesn't include ``mocap_pos`` — without this,
GPS C-step states captured under different goals would all restore to
the env's last goal. ``batch_rollout`` strips the goal before calling
``mujoco.rollout``.
"""

import mujoco
import numpy as np
from pathlib import Path
from jaxtyping import Array, Float
from numpy import ndarray

from src.envs.mujoco_env import MuJoCoEnv

_XML = str(Path(__file__).resolve().parents[2] / "assets" / "point_mass.xml")

# Target sampling box — matches the arena's wall bounds at ±0.29.
_TARGET_LOW = -0.29
_TARGET_HIGH = 0.29

# Task-success criteria (for `task_metrics`; not in the MPPI cost).
_GOAL_RADIUS = 0.025
_SUCCESS_VEL = 0.05

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
        # Constant within one MPPI rollout (one plan_step = one goal).
        self._goal = np.zeros(2, dtype=np.float64)

    def _sample_goal(self) -> np.ndarray:
        return np.random.uniform(_TARGET_LOW, _TARGET_HIGH, size=2).astype(
            np.float64
        )

    def reset(
        self,
        state: Float[ndarray, "state_dim"] | None = None,
    ) -> Float[ndarray, "4"]:
        if state is not None:
            return super().reset(state=state)

        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = np.random.uniform(-0.25, 0.25, size=self._nq)
        self.data.qvel[:] = np.random.normal(0.0, 0.1, size=self._nv)
        self._goal = self._sample_goal()
        self.data.mocap_pos[0, :2] = self._goal
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs()

    def get_state(self) -> np.ndarray:
        physics = super().get_state()
        return np.concatenate([physics, self._goal])

    def set_state(self, state: np.ndarray) -> None:
        # Accept both extended (state_dim) and raw FULLPHYSICS shapes.
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
        # Strip the trailing goal before mujoco.rollout (FULLPHYSICS-shaped
        # input only) and sync `_goal` so cost / state_to_obs see it.
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

    def task_metrics(self) -> dict:
        pos = self.data.qpos.copy()
        vel = self.data.qvel.copy()
        dist = float(np.linalg.norm(pos - self._goal))
        qvel_norm = float(np.linalg.norm(vel))
        return {
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

    def _get_obs(self) -> Float[ndarray, "6"]:
        return np.concatenate([self.data.qpos, self.data.qvel, self._goal])

    def state_to_obs(
        self,
        states: np.ndarray,
        sensordata: np.ndarray | None = None,
    ) -> np.ndarray:
        qpos = self.state_qpos(states)
        qvel = self.state_qvel(states)
        goal = np.broadcast_to(self._goal, qpos.shape[:-1] + (2,))
        return np.concatenate([qpos, qvel, goal], axis=-1)

    @property
    def obs_dim(self) -> int:
        return self._nq + self._nv + 2
