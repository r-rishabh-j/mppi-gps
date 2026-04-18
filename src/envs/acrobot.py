"""acrobot swing up env"""

import mujoco
import numpy as np
from pathlib import Path
from jaxtyping import Array, Float
from numpy import ndarray

from src.envs.mujoco_env import MuJoCoEnv

_XML = str(Path(__file__).resolve().parents[2] / "assets" / "acrobot.xml")

_TARGET = np.array([0.0, 0.0, 4.0])

class Acrobot(MuJoCoEnv):
    def __init__(self, frame_skip: int = 1, **kwargs) -> None:
        super().__init__(model_path=_XML, frame_skip=frame_skip, **kwargs)
        self._nq = self.model.nq #2
        self._nv = self.model.nv #2

        self._w_terminal = 3.0
        # Pseudo-energy shaping (Spong-style): penalise only the deficit
        # (E_target - E)+². KE proxy 0.5||qvel||², PE proxy tip_z. One-sided
        # so MPPI always wants to add energy when below target, and there's
        # no degenerate "pump at the bottom" optimum at E=E_target.
        self._energy_target = 4.0
        self._energy_weight = 0.08
        self._near_top_radius = 0.5
        self._vel_weight = 0.002           # tiny global damping
        self._vel_weight_near_top = 0.05   # stabilisation near the top
        self._ctrl_weight = 0.001


    def reset(
            self,
            state: Float[ndarray, "state_dim"] | None = None,
    ) -> Float[ndarray, "6"]:
        if state is not None:
            return super().reset(state = state)

        obs = super().reset()
        self.data.qpos[:] = np.random.uniform(-np.pi, np.pi, size=self._nq)
        self.data.qvel[:] = np.random.normal(0.0, 0.05, size=self._nv) * 0.0
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs()

    def running_cost(
            self,
            states: Float[Array, "K H nstate"],
            actions: Float[Array, "K H nu"],
            sensordata: Float[Array, "K H nsensor"] | None = None,
    ) -> Float[Array, "K H"]:
        tip_pos = sensordata[:, :, :3]
        tip_z = tip_pos[..., 2]                                 # (K, H) — PE proxy
        qvel = self.state_qvel(states)                          # (K, H, nv)
        vel_sq = np.sum(qvel ** 2, axis=-1)                     # (K, H)

        dist = np.linalg.norm(tip_pos - _TARGET, axis=-1)       # (K, H)

        # Linear far-field pull toward the target direction. Energy cost
        # alone only matches magnitudes; this term says *which way* is up.
        dist_cost = dist / 4.0                                  # in [0, 2]

        # Flat-bottomed Gaussian-tolerance bonus: sharp attractor near the
        # top so the planner strictly prefers "at top" over any same-energy
        # state elsewhere.
        target_radius = 0.20
        value_at_margin = 0.1
        margin = 4.0
        scale = np.sqrt(-2.0 * np.log(value_at_margin))
        d_beyond = np.maximum(dist - target_radius, 0.0)
        reward = np.where(
            dist <= target_radius,
            1.0,
            np.exp(-0.5 * (d_beyond * scale / margin) ** 2),
        )
        tip_cost = dist_cost + (1.0 - reward)

        # One-sided pseudo-energy deficit: max(E_target - E, 0)².
        # Monotone-decreasing in KE for any state with E < E_target, so
        # random MPPI samples that add velocity get lower cost and bias the
        # weighted-mean U toward a pumping sequence.
        E = 0.5 * vel_sq + tip_z
        energy_deficit = np.maximum(self._energy_target - E, 0.0)
        energy_cost = self._energy_weight * energy_deficit ** 2

        # Global light damping + near-top stabilisation (energy cost goes
        # to zero once E ≥ E_target, so the near-top term is what finally
        # kills residual KE at the upright).
        near_top = (dist < self._near_top_radius).astype(vel_sq.dtype)
        vel_cost = (self._vel_weight * vel_sq
                    + self._vel_weight_near_top * near_top * vel_sq)

        ctrl_cost = self._ctrl_weight * np.sum(actions ** 2, axis=-1)

        return tip_cost + energy_cost + vel_cost + ctrl_cost

    def terminal_cost(
            self,
            states: Float[Array, "K nstate"],
            sensordata: Float[Array, "K nsensor"] | None = None,
    ) -> Float[Array, "K"]:
        tip_pos = sensordata[:, :3]
        dist = np.linalg.norm(tip_pos - _TARGET, axis=-1)
        qvel = self.state_qvel(states)
        vel_cost = np.sum(qvel**2, axis=-1)
        return self._w_terminal * (2.0 * dist + 5.0 * vel_cost)

    def running_cost_torch(self, states, actions, sensordata=None):
        import torch
        tip_pos = sensordata[:, :, :3]
        tip_z = tip_pos[..., 2]
        qvel = self.state_qvel(states)
        vel_sq = (qvel ** 2).sum(dim=-1)

        target = torch.as_tensor(_TARGET, dtype=tip_pos.dtype, device=tip_pos.device)
        dist = torch.linalg.vector_norm(tip_pos - target, dim=-1)

        dist_cost = dist / 4.0

        target_radius = 0.20
        value_at_margin = 0.1
        margin = 4.0
        scale = float(np.sqrt(-2.0 * np.log(value_at_margin)))
        d_beyond = torch.clamp(dist - target_radius, min=0.0)
        reward = torch.where(
            dist <= target_radius,
            torch.ones_like(dist),
            torch.exp(-0.5 * (d_beyond * scale / margin) ** 2),
        )
        tip_cost = dist_cost + (1.0 - reward)

        E = 0.5 * vel_sq + tip_z
        energy_deficit = torch.clamp(self._energy_target - E, min=0.0)
        energy_cost = self._energy_weight * energy_deficit ** 2

        near_top = (dist < self._near_top_radius).to(vel_sq.dtype)
        vel_cost = (self._vel_weight * vel_sq
                    + self._vel_weight_near_top * near_top * vel_sq)

        ctrl_cost = self._ctrl_weight * (actions ** 2).sum(dim=-1)

        return tip_cost + energy_cost + vel_cost + ctrl_cost

    def terminal_cost_torch(self, states, sensordata=None):
        import torch
        tip_pos = sensordata[:, :3]
        target = torch.as_tensor(_TARGET, dtype=tip_pos.dtype, device=tip_pos.device)
        dist = torch.linalg.vector_norm(tip_pos - target, dim=-1)
        qvel = self.state_qvel(states)
        vel_cost = (qvel ** 2).sum(dim=-1)
        return self._w_terminal * (2.0 * dist + 5.0 * vel_cost)

    def _get_obs(self) -> Float[ndarray, "6"]:
        # sin/cos encoding removes the ±π wrap-around discontinuity the MLP
        # would otherwise have to memorise around the upright state.
        q = self.data.qpos
        v = self.data.qvel
        return np.concatenate([np.sin(q), np.cos(q), v])

    def state_to_obs(self, states: np.ndarray) -> np.ndarray:
        qpos = self.state_qpos(states)
        qvel = self.state_qvel(states)
        return np.concatenate([np.sin(qpos), np.cos(qpos), qvel], axis=-1)

    @property
    def obs_dim(self) -> int:
        return 2 * self._nq + self._nv
