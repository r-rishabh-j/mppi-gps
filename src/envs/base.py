
"""base class definition for MPPI compatible envs"""
from abc import ABC, abstractmethod 
import numpy as np 

class BaseEnv(ABC):
    @abstractmethod
    def reset(self, state: np.ndarray | None = None) -> np.ndarray:
        """reset to the initial or specified state and returns None"""

    @abstractmethod
    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        """Advance one timestep. Returns (obs, cost, done, info)."""

    @abstractmethod
    def get_state(self) -> np.ndarray:
        """Capture full simulator state (sufficient to restore exactly)."""

    @abstractmethod
    def set_state(self, state: np.ndarray) -> None:
        """Restore simulator to a previously captured state."""

    @abstractmethod
    def cost(self, state: np.ndarray, action: np.ndarray) -> float:
        """Running cost at (state, action)."""

    @abstractmethod
    def terminal_cost(self, state: np.ndarray) -> float:
        """Terminal cost at final state."""

    @property
    @abstractmethod
    def state_dim(self) -> int: ...

    @property
    @abstractmethod
    def action_dim(self) -> int: ...

    @property
    @abstractmethod
    def action_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns (low, high) arrays of shape (action_dim,)."""

    def batch_rollout(
            self, 
            initial_state: np.ndarray, 
            action_sequences: np.ndarray, 
    ) -> tuple[np.ndarray, np.ndarray]:
        """rollout K action sequences of length H from the initial state"""
        pass 