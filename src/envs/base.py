"""BaseEnv: abstract interface for MPPI-compatible envs."""
from abc import ABC, abstractmethod
import numpy as np


class BaseEnv(ABC):
    @abstractmethod
    def reset(self, state: np.ndarray | None = None) -> np.ndarray:
        """Reset to initial (or specified) state; return obs."""

    @abstractmethod
    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        """Advance one step. Returns (obs, cost, done, info)."""

    @abstractmethod
    def get_state(self) -> np.ndarray:
        """Capture full simulator state (sufficient to restore exactly)."""

    @abstractmethod
    def set_state(self, state: np.ndarray) -> None:
        """Restore simulator to a previously captured state."""

    @abstractmethod
    def running_cost(self, states: np.ndarray, actions: np.ndarray, sensordata: np.ndarray | None = None) -> np.ndarray:
        """Vectorized: states (K,H,nstate), actions (K,H,nu), sensordata (K,H,nsensor) → (K,H)."""

    @abstractmethod
    def terminal_cost(self, states: np.ndarray, sensordata: np.ndarray | None = None) -> np.ndarray:
        """Vectorized: states (K,nstate), sensordata (K,nsensor) → (K,)."""

    @property
    @abstractmethod
    def state_dim(self) -> int: ...

    @property
    @abstractmethod
    def action_dim(self) -> int: ...

    @property
    @abstractmethod
    def obs_dim(self) -> int: ...

    @property
    @abstractmethod
    def action_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """(low, high), each shape (action_dim,)."""

    @property
    def noise_scale(self) -> np.ndarray:
        """Per-dim multiplier on MPPI exploration noise.

        Effective sigma is ``cfg.noise_sigma * env.noise_scale``. Envs with
        heterogeneous ctrlranges should override to ``(high - low) / 2`` so
        cfg.noise_sigma reads as a uniform fraction of each dim's half-range.
        """
        return np.ones(self.action_dim)

    @abstractmethod
    def state_to_obs(
        self,
        states: np.ndarray,
        sensordata: np.ndarray | None = None,
    ) -> np.ndarray:
        """Convert physics states to policy obs: (..., nstate) → (..., obs_dim).

        ``sensordata`` is optional. Envs whose obs needs kinematic outputs
        only available via ``mj_forward`` (e.g. adroit_relocate DAPG mode)
        use it to skip per-state forward kinematics in the MPPI hot loop.
        """

    @abstractmethod
    def batch_rollout(
            self,
            initial_state: np.ndarray,
            action_sequences: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Roll out K action sequences of length H.

        Args:
            initial_state: state vector from ``get_state()``.
            action_sequences: (K, H, action_dim).

        Returns:
            states:     (K, H, nstate)
            costs:      (K,)
            sensordata: (K, H, nsensor)
        """
