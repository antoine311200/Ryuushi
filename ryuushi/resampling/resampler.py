from enum import Enum
from typing import List
from abc import ABC, abstractmethod

from ryuushi.particle import Particle


class ResamplingScheme(Enum):
    """Enumeration of resampling schemes"""
    MULTINOMIAL = "multinomial"
    SYSTEMATIC = "systematic"
    STRATIFIED = "stratified"
    RESIDUAL = "residual"
    ADAPTIVE = "adaptive"

class Resampler(ABC):
    """Abstract base class for resamplers"""
    @abstractmethod
    def resample(self, particles: List[Particle]) -> List[Particle]:
        """Resample particles according to their weights"""
        pass