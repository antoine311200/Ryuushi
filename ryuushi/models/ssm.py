from abc import ABC, abstractmethod
from typing import Any

import numpy as np

class StateSpaceModel(ABC):
    """Abstract base class for state space models"""
    @abstractmethod
    def transition(self, state: np.ndarray, time: int, dt: float = None) -> np.ndarray: # type: ignore
        pass

    @abstractmethod
    def log_likelihood(self, state: np.ndarray, observation: Any, time: int) -> float:
        pass

    @abstractmethod
    def initial_state(self) -> np.ndarray:
        pass
