from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from ryuushi.particle import Particle

class ImportanceDistribution(ABC):
    """Abstract base class for importance distributions"""
    @abstractmethod
    def sample(self, particle: Particle, data: Any, time: int) -> np.ndarray:
        pass

    @abstractmethod
    def log_density(self, state: np.ndarray, particle: Particle, data: Any, time: int) -> float:
        pass
