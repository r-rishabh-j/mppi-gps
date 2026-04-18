"""Hopper env — contact-rich locomotion task.

This is the "contact-rich" evaluation environment from the project proposal.
The hopper has intermittent ground contact, making it significantly harder
than the acrobot swing-up task.

Model details (from hopper.xml):
  - 6 generalised positions (qpos): rootx, rootz, rooty, thigh, leg, foot
  - 6 generalised velocities (qvel): corresponding joint velocities
  - 3 actuators: thigh_joint, leg_joint, foot_joint (range [-1, 1])
  - Observation: qpos[1:] (skip root x) + clip(qvel, -10, 10) → 11 dims

The cost function negates the standard Gymnasium Hopper-v5 reward:
  cost = -(forward_velocity_reward + healthy_reward) + ctrl_cost
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
_Z_MIN = 0.7
_ANGLE_MAX = 0.2


class Hopper(MuJoCoEnv):
    def __init__(
        self,
        frame_skip: int = 4,
        ctrl_cost_weight: float = 0.001,
        forward_reward_weight: float = 1.0,
        healthy_reward: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(model_path=_XML, frame_skip=frame_skip, **kwargs)
        self._nq = self.model.nq  # 6: rootx, rootz, rooty, thigh, leg, foot
        self._nv = self.model.nv  # 6
        self._ctrl_w = ctrl_cost_weight
        self._fwd_w = forward_reward_weight
        self._healthy_reward = healthy_reward

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
        self.data.qpos[:] += np.random.uniform(-0.005, 0.005, size=self._nq)
        self.data.qvel[:] += np.random.uniform(-0.005, 0.005, size=self._nv)
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

    def running_cost(
        self,
        states: Float[Array, "K H nstate"],
        actions: Float[Array, "K H nu"],
        sensordata: Float[Array, "K H nsensor"] | None = None,
    ) -> Float[Array, "K H"]:
        """Vectorised per-step cost for K×H (sample, horizon) grid.

        Cost = -(forward_velocity × fwd_w) - (healthy_reward × is_healthy) + ctrl_w × ||u||²

        Note: during MPPI planning rollouts we do NOT terminate on unhealthy
        states — we just let the high cost accumulate so that MPPI naturally
        avoids falling.  Termination only happens in the real step().
        """
        qpos = self.state_qpos(states)  # (K, H, 6) — uses the +1 offset for time
        qvel = self.state_qvel(states)  # (K, H, 6)

        # Forward velocity = d(rootx)/dt = qvel[0]
        vx = qvel[:, :, 0]

        # Quadratic control cost: penalise large actuator efforts
        ctrl_cost = np.sum(np.square(actions), axis=-1)

        # Healthy reward: +1 per step when the hopper is upright, else 0
        z = qpos[:, :, 1]       # rootz (vertical position)
        angle = qpos[:, :, 2]   # rooty (torso pitch angle)
        healthy = self._is_healthy(z, angle).astype(np.float64)

        # Combine into cost (we minimise, so negate the reward terms)
        cost = (
            -self._fwd_w * vx
            - self._healthy_reward * healthy
            + self._ctrl_w * ctrl_cost
        )
        return cost

    def terminal_cost(
        self,
        states: Float[Array, "K nstate"],
        sensordata: Float[Array, "K nsensor"] | None = None,
    ) -> Float[Array, "K"]:
        # No terminal cost — the running cost fully specifies the objective.
        return np.zeros(states.shape[0])

    def running_cost_torch(self, states, actions, sensordata=None):
        qpos = self.state_qpos(states)
        qvel = self.state_qvel(states)
        vx = qvel[:, :, 0]
        ctrl_cost = (actions ** 2).sum(dim=-1)
        z = qpos[:, :, 1]
        angle = qpos[:, :, 2]
        healthy = ((z > _Z_MIN) & (angle.abs() < _ANGLE_MAX)).to(vx.dtype)
        return (
            -self._fwd_w * vx
            - self._healthy_reward * healthy
            + self._ctrl_w * ctrl_cost
        )

    def terminal_cost_torch(self, states, sensordata=None):
        import torch
        return torch.zeros(states.shape[0], dtype=states.dtype, device=states.device)

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

    def state_to_obs(self, states: np.ndarray) -> np.ndarray:
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
