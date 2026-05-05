
"""base class definition for MPPI compatible envs"""
from abc import ABC, abstractmethod 
import numpy as np 

class BaseEnv(ABC):
    @abstractmethod
    def reset(self, state: np.ndarray | None = None) -> np.ndarray:
        """reset to the initial or specified state and returns the observation"""

    @abstractmethod
    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        """Advance one timestep. Returns (obs, cost, done, info)"""

    @abstractmethod
    def get_state(self) -> np.ndarray:
        """Capture full simulator state (sufficient to restore exactly)"""

    @abstractmethod
    def set_state(self, state: np.ndarray) -> None:
        """Restore simulator to a previously captured state"""
    
    @abstractmethod
    def running_cost(self, states: np.ndarray, actions: np.ndarray, sensordata: np.ndarray | None = None) -> np.ndarray:
      """Vectorized running cost. states: (K,H,nstate), actions: (K,H,nu), sensordata: (K,H,nsensor) → (K,H)"""

    @abstractmethod
    def terminal_cost(self, states: np.ndarray, sensordata: np.ndarray | None = None) -> np.ndarray:
      """Vectorized terminal cost. states: (K,nstate), sensordata: (K,nsensor) → (K,)"""

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
        """Returns (low, high) arrays of shape (action_dim,)"""

    @property
    def noise_scale(self) -> np.ndarray:
        """Per-dim multiplier on MPPI's exploration noise (shape (action_dim,)).

        MPPI's effective sigma is ``cfg.noise_sigma * env.noise_scale`` —
        i.e. cfg.noise_sigma keeps a unit-free meaning, and the env can opt in
        to per-dim scaling when its actuators have heterogeneous ctrlranges.

        Default is ``ones(action_dim)`` (backward compatible — sigma stays
        absolute everywhere). Envs whose actuators span very different ranges
        (e.g. Adroit's 6 arm joints + 24 hand joints, ratio ~10x) should
        override to return ``(high - low) / 2`` so cfg.noise_sigma becomes
        a uniform fraction of each dim's half-range.
        """
        return np.ones(self.action_dim)

    @abstractmethod
    def state_to_obs(
        self,
        states: np.ndarray,
        sensordata: np.ndarray | None = None,
    ) -> np.ndarray:
        """Convert full physics states to policy observations.

        states: (..., nstate) → (..., obs_dim)

        ``sensordata`` (optional, shape ``(..., nsensor)``) is the matching
        sensor output from ``batch_rollout``. Most envs ignore it — they
        derive obs purely from qpos/qvel. Envs whose obs depends on
        kinematic outputs only available via ``mj_forward`` (e.g.
        ``adroit_relocate`` in DAPG mode, which needs palm site position)
        use it to avoid running per-state forward kinematics in the MPPI
        K×H hot loop. Callers in the MPPI prior path pass it; callers in
        eval / relabel paths don't and accept the slower fallback.
        """

    @abstractmethod
    def batch_rollout(
            self, 
            initial_state: np.ndarray, 
            action_sequences: np.ndarray, 
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Roll out K action sequences of length H from initial_state.

        Args:
        initial_state: state vector from get_state()
        action_sequences: (K, H, action_dim)

        Returns:
        states:     (K, H, nstate) state at each step
        costs:      (K,) total cost per trajectory
        sensordata: (K, H, nsensor) sensor readings at each step
        """
        