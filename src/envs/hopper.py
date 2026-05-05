"""Hopper env — contact-rich locomotion task.

Supports two running-cost modes:
  - ``v1``: the original Gymnasium-style additive forward reward + alive bonus
    with quadratic control penalty.
  - ``v2``: a dm_control-style multiplicative hop reward. The hopper only gets
    rewarded for moving quickly when it is also standing tall.
"""

import mujoco
import numpy as np
from pathlib import Path
from jaxtyping import Array, Float
from numpy import ndarray

from src.envs.mujoco_env import MuJoCoEnv

_XML = str(Path(__file__).resolve().parents[2] / "assets" / "hopper.xml")

# Healthy bounds — the hopper is considered "fallen" (done=True) if
# z position drops below _Z_MIN or torso angle exceeds _ANGLE_MAX.
# These values match Gymnasium's Hopper-v5 defaults.
_Z_MIN = 0.2
_ANGLE_MAX = 1
_VALID_COST_MODES = {"v1", "v2"}


class Hopper(MuJoCoEnv):
    def __init__(
        self,
        frame_skip: int = 4,
        ctrl_cost_weight: float = 0.001,
        forward_reward_weight: float = 1.0,
        healthy_reward: float = 1.0,
        cost_mode: str = "v2",
        **kwargs,
    ) -> None:
        super().__init__(model_path=_XML, frame_skip=frame_skip, **kwargs)
        self._nq = self.model.nq  # 6: rootx, rootz, rooty, thigh, leg, foot
        self._nv = self.model.nv  # 6
        self._ctrl_w = ctrl_cost_weight
        self._fwd_w = forward_reward_weight
        self._healthy_reward = healthy_reward
        if cost_mode not in _VALID_COST_MODES:
            raise ValueError(
                f"Unknown hopper cost_mode={cost_mode!r}. "
                f"Expected one of {sorted(_VALID_COST_MODES)}."
            )
        self._cost_mode = cost_mode

        # Sensor layout from assets/hopper.xml:
        #   torso_pos          framepos(body=torso)         -> 3 floats
        #   foot_pos           framepos(body=foot)          -> 3 floats
        #   torso_subtreelinvel subtreelinvel(body=torso)   -> 3 floats
        self._torso_pos_slice = slice(0, 3)
        self._foot_pos_slice = slice(3, 6)
        self._torso_linvel_slice = slice(6, 9)
        self._public_sensor_dim = 0

    def reset(
        self,
        state: Float[ndarray, "state_dim"] | None = None,
    ) -> np.ndarray:
        if state is not None:
            # Deterministic reset to a captured state (used by GPS to
            # replay the same initial condition across iterations).
            return super().reset(state=state)

        obs = super().reset()
        # Small random perturbation around the default standing pose
        # to diversify initial conditions.
        self.data.qpos[:] += np.random.uniform(-0.005, 0.1, size=self._nq)
        self.data.qvel[:] += np.random.uniform(-0.005, 0.2, size=self._nv)
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs()

    def _is_healthy(self, qpos_z: np.ndarray, qpos_angle: np.ndarray,
                    obs: np.ndarray | None = None) -> np.ndarray:
        """Vectorised health check.  Works for both scalar and batched inputs.

        The hopper is "healthy" when:
          - z position (rootz) is above _Z_MIN  (hasn't fallen)
          - torso angle (rooty) is within [-_ANGLE_MAX, _ANGLE_MAX]  (upright)
          - all observation values are finite and within (-100, 100)
        """
        z_ok = qpos_z > _Z_MIN
        angle_ok = np.abs(qpos_angle) < _ANGLE_MAX
        healthy = z_ok & angle_ok
        if obs is not None:
            state_ok = np.all(np.isfinite(obs)) & np.all(np.abs(obs) < 100.0)
            healthy = healthy & state_ok
        return healthy

    def _tolerance(
        self,
        x: np.ndarray,
        bounds: tuple[float, float],
        margin: float = 0.0,
        sigmoid: str = "gaussian",
        value_at_margin: float = 0.1,
    ) -> np.ndarray:
        """Minimal copy of dm_control's tolerance helper for Hopper reward terms."""
        lower, upper = bounds
        in_bounds = (lower <= x) & (x <= upper)
        if margin == 0.0:
            return np.where(in_bounds, 1.0, 0.0)

        d = np.where(x < lower, lower - x, x - upper) / margin
        if sigmoid == "linear":
            scaled = d * (1.0 - value_at_margin)
            return np.where(np.abs(scaled) < 1.0, 1.0 - scaled, 0.0)
        raise ValueError(f"Unsupported sigmoid={sigmoid!r} for Hopper tolerance.")

    def _running_cost_v1(
        self,
        states: Float[Array, "K H nstate"],
        actions: Float[Array, "K H nu"],
    ) -> Float[Array, "K H"]:
        qpos = self.state_qpos(states)
        qvel = self.state_qvel(states)
        vx = qvel[:, :, 0]
        ctrl_cost = np.sum(np.square(actions), axis=-1)
        z = qpos[:, :, 1]
        angle = qpos[:, :, 2]
        healthy = self._is_healthy(z, angle).astype(np.float64)
        return (
            -self._fwd_w * vx
            - self._healthy_reward * healthy
            + self._ctrl_w * ctrl_cost
        )

    def _running_cost_v2(
        self,
        sensordata: Float[Array, "K H nsensor"],
    ) -> Float[Array, "K H"]:
        torso_z = sensordata[:, :, self._torso_pos_slice.stop - 1]
        foot_z = sensordata[:, :, self._foot_pos_slice.stop - 1]
        torso_vx = sensordata[:, :, self._torso_linvel_slice.start]

        standing = self._tolerance(
            torso_z - foot_z,
            bounds=(0.6, 2.0),
        )
        hopping = self._tolerance(
            self._fwd_w * torso_vx,
            bounds=(2.0, float("inf")),
            margin=1.0,
            value_at_margin=0.5,
            sigmoid="linear",
        )
        reward = standing * hopping
        return -reward

    def _public_sensordata(
        self,
        sensordata: np.ndarray,
    ) -> np.ndarray:
        """Hide dm_control-style reward sensors from external callers."""
        return sensordata[..., :self._public_sensor_dim]

    def running_cost(
        self,
        states: Float[Array, "K H nstate"],
        actions: Float[Array, "K H nu"],
        sensordata: Float[Array, "K H nsensor"] | None = None,
    ) -> Float[Array, "K H"]:
        """Vectorised per-step cost for K×H (sample, horizon) grid."""
        if self._cost_mode == "v1":
            return self._running_cost_v1(states, actions)
        return self._running_cost_v2(sensordata)

    def terminal_cost(
        self,
        states: Float[Array, "K nstate"],
        sensordata: Float[Array, "K nsensor"] | None = None,
    ) -> Float[Array, "K"]:
        # Both Hopper cost modes are fully specified by the running cost.
        return np.zeros(states.shape[0])

    def batch_rollout(
        self,
        initial_state: np.ndarray,
        action_sequences: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run Hopper rollouts while keeping reward-only sensors private.

        The v2 cost uses extra MuJoCo sensors internally, but callers still see
        the pre-v2 public sensor payload (empty for Hopper).
        """
        actions_expanded = np.repeat(action_sequences, self._frame_skip, axis=1)
        states_full, sensordata_full = self._rollout_ctx.rollout(
            self.model,
            self._data_pool,
            initial_state,
            actions_expanded,
        )

        states = states_full[:, ::self._frame_skip, :]
        private_sensordata = sensordata_full[:, ::self._frame_skip, :]
        costs = (
            self.running_cost(states, action_sequences, private_sensordata).sum(axis=1)
            + self.terminal_cost(states[:, -1, :], private_sensordata[:, -1, :])
        )
        public_sensordata = self._public_sensordata(private_sensordata)
        return states, costs, public_sensordata

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        """Single environment step with termination on unhealthy state.

        Unlike running_cost (which never terminates, for MPPI planning),
        this method returns done=True when the hopper falls.
        """
        obs, cost, _, info = super().step(action)
        z = self.data.qpos[1]
        angle = self.data.qpos[2]
        done = not self._is_healthy(
            np.array(z), np.array(angle), obs
        ).item()
        return obs, cost, done, info

    def _get_obs(self) -> np.ndarray:
        """Policy observation: qpos[1:] (skip root x) + clipped qvel.

        Root x is excluded because it grows unboundedly — the policy
        should be translation-invariant.  Velocities are clipped to [-10, 10]
        to prevent outlier observations from destabilising training.
        """
        return np.concatenate([
            self.data.qpos[1:],
            np.clip(self.data.qvel, -10.0, 10.0),
        ])

    def state_to_obs(
        self,
        states: np.ndarray,
        sensordata: np.ndarray | None = None,
    ) -> np.ndarray:
        """Convert batched full physics states to policy observations.

        Mirrors _get_obs() but works on (..., nstate) arrays from batch_rollout.
        Uses state_qpos / state_qvel which handle the time-offset in the
        full physics state vector [time, qpos, qvel, act, ...].
        """
        qpos = self.state_qpos(states)[..., 1:]  # skip root x
        qvel = np.clip(self.state_qvel(states), -10.0, 10.0)
        return np.concatenate([qpos, qvel], axis=-1)

    @property
    def obs_dim(self) -> int:
        # 5 qpos (skip root x) + 6 qvel = 11
        return (self._nq - 1) + self._nv
