
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

    @abstractmethod
    def state_to_obs(self, states: np.ndarray) -> np.ndarray:
        """Convert full physics states to policy observations.
        states: (..., nstate) → (..., obs_dim)"""

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
        