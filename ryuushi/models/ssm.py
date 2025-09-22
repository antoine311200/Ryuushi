from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np

class StateSpaceModel(ABC):
    """Abstract base class for state space models"""
    @abstractmethod
    def transition(self, state: np.ndarray, time: int = 0, dt: float = 0.1, parameters: Optional[np.ndarray] = None) -> np.ndarray: # type: ignore
        pass

    @abstractmethod
    def log_likelihood(self, state: np.ndarray, observation: Any, time: int, parameters: Optional[np.ndarray] = None) -> float:
        pass

    @abstractmethod
    def initial_state(self) -> np.ndarray:
        pass

    @abstractmethod
    def initial_parameters(self) -> np.ndarray:
        pass