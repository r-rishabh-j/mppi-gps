"""2D point-mass goal-reaching task — random target each reset.

Mirrors `jaiselsingh1/mppi-gps`'s point_mass env. Cost weights,
observation shape, and goal workspace match upstream verbatim:

* **Observation** (6-D): ``[qpos (2), qvel (2), goal (2)]`` — goal is
  an explicit input, NOT folded into qpos. The policy learns
  goal-conditioning from data rather than from a hard-baked delta.
* **Random target**, sampled uniformly from the full arena
  ``[-0.29, 0.29]²`` on each ``reset()``. Pushed to the scene's mocap
  body so the goal indicator follows visually.
* **Cost**:

      running   = w_pos · ‖qpos − goal‖²  +  w_vel · ‖qvel‖²  +  w_ctrl · ‖u‖²
      terminal  = w_pos_T · ‖qpos − goal‖²  +  w_vel_T · ‖qvel‖²

  The terminal-pos weight is 20× the running so MPPI pulls hard at the
  horizon end; the terminal-vel weight asks it to land softly.

Goal round-trip through ``get_state`` / ``set_state``
------------------------------------------------------
MuJoCo's ``mjSTATE_FULLPHYSICS`` bitfield in our version does NOT
include ``mjSTATE_MOCAP_POS``, so a state captured by the parent
``MuJoCoEnv.get_state()`` carries qpos / qvel but *not* the goal —
restoring it would leave ``self._goal`` stale and the mocap marker
reset to its XML default. This silently breaks the GPS C-step:
``_sample_initial_conditions`` captures ``(state, goal_i)`` per
condition, but at C-step time ``env.reset(state=ic_state)`` doesn't
restore ``goal_i`` — every (obs, action) in n−1 of n conditions trains
on the wrong goal, and BC learns garbage.

Fix: ``get_state`` appends the 2-D goal to the FULLPHYSICS vector,
``set_state`` peels it off and syncs ``self._goal`` + ``mocap_pos`` from
the trailing 2 entries. ``batch_rollout`` strips the goal tail before
handing the state to ``mujoco.rollout`` (which expects FULLPHYSICS-sized
input). Cost / state_to_obs read ``self._goal`` directly — broadcast
across all K rollouts (one MPPI call has one goal).
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
        # Cached goal in [x, y]. Updated on every reset() and broadcast in
        # `running_cost`/`terminal_cost`. Constant across one MPPI rollout
        # (set before the first plan_step of the episode), so we don't need
        # a per-state sensor for it.
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
            # State-restore path. Parent reset calls our overridden
            # `set_state`, which unpacks the trailing goal entries from
            # the extended state vector and syncs `self._goal` + mocap.
            return super().reset(state=state)

        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = np.random.uniform(-0.25, 0.25, size=self._nq)
        self.data.qvel[:] = np.random.normal(0.0, 0.1, size=self._nv)
        self._goal = self._sample_goal()
        # Keep the mocap body's z from its XML pose; only move x, y.
        self.data.mocap_pos[0, :2] = self._goal
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs()

    # ------------------------------------------------------------------
    # State capture/restore — extended with goal so condition-restore
    # round-trips through MPPI / GPS / DAgger correctly. See module
    # docstring for the bug this prevents.
    # ------------------------------------------------------------------

    def get_state(self) -> np.ndarray:
        physics = super().get_state()                 # FULLPHYSICS-size
        return np.concatenate([physics, self._goal])  # +2 dims for goal_xy

    def set_state(self, state: np.ndarray) -> None:
        # Last 2 entries are the goal; everything before is FULLPHYSICS.
        # Accept both extended (state_dim) and raw (FULLPHYSICS) inputs so
        # any legacy caller passing a plain mujoco state still works (the
        # goal cache simply isn't updated — same as the previous behavior).
        if state.shape[-1] == self._nstate + 2:
            super().set_state(state[: self._nstate])
            goal = state[self._nstate : self._nstate + 2]
            self._goal = np.asarray(goal, dtype=np.float64).copy()
            self.data.mocap_pos[0, :2] = self._goal
        else:
            super().set_state(state)

    @property
    def state_dim(self) -> int:
        # Honest accounting: FULLPHYSICS + goal_xy. Most consumers use
        # MuJoCoEnv's `state_qpos` / `state_qvel` which index into the
        # rollout output (FULLPHYSICS-sized, unaffected), but anyone
        # allocating buffers from `state_dim` will get the right size.
        return self._nstate + 2

    def batch_rollout(
        self,
        initial_state: np.ndarray,
        action_sequences: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # mujoco.rollout expects FULLPHYSICS-shaped initial states. When
        # called by MPPI with a `get_state()` capture, strip the trailing
        # goal entries first and broadcast the goal into `self._goal` so
        # `running_cost` / `terminal_cost` / `state_to_obs` see the goal
        # that was active when this state was captured. Idempotent on
        # already-FULLPHYSICS input (legacy callers).
        initial_state = np.asarray(initial_state)
        if initial_state.shape[-1] == self._nstate + 2:
            # 1-D state (single condition broadcast across K) or 2-D
            # batched per-K (we still expect ONE goal across K — MPPI
            # always broadcasts a single state). Take the first row.
            if initial_state.ndim == 1:
                goal = initial_state[self._nstate : self._nstate + 2]
            else:
                goal = initial_state[0, self._nstate : self._nstate + 2]
            self._goal = np.asarray(goal, dtype=np.float64).copy()
            # Also sync the env's live mocap_pos for visualisation parity;
            # mujoco.rollout uses pool mjData, so this is only for
            # rendering from `env.data` after the call.
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

    # ------------------------------------------------------------------
    # Observation — [qpos, qvel, goal] (6-D), matches upstream verbatim.
    #
    # Why explicit goal (vs delta-to-goal):
    # * MPPI demos aren't *perfectly* translation-invariant — wall-
    #   proximity effects + finite-K MPPI noise mean two states with the
    #   same `qpos − goal` can map to slightly different optimal actions
    #   in practice. With 4-D delta obs, BC sees those as label noise on
    #   identical inputs and learns the average — a flatter policy.
    # * The 6-D form lets the policy learn the relationship from data:
    #   it can collapse to "act on (qpos − goal)" internally, or use qpos
    #   and goal independently if wall proximity matters.
    # ------------------------------------------------------------------

    def _get_obs(self) -> Float[ndarray, "6"]:
        return np.concatenate([self.data.qpos, self.data.qvel, self._goal])

    def state_to_obs(
        self,
        states: np.ndarray,
        sensordata: np.ndarray | None = None,
    ) -> np.ndarray:
        qpos = self.state_qpos(states)
        qvel = self.state_qvel(states)
        # Broadcast the env's current goal across the leading dims of
        # `states` (K, H, ...). One MPPI rollout uses one goal — set on
        # the env via `batch_rollout`'s tail-strip, or via `reset()` for
        # closed-loop rollout. The goal stays the same across the
        # batched K×H grid in either case.
        goal = np.broadcast_to(self._goal, qpos.shape[:-1] + (2,))
        return np.concatenate([qpos, qvel, goal], axis=-1)

    @property
    def obs_dim(self) -> int:
        # qpos (2) + qvel (2) + goal (2) = 6
        return self._nq + self._nv + 2
