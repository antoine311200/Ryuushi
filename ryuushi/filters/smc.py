from abc import ABC, abstractmethod
from typing import Any, List

import numpy as np

from ryuushi.models.ssm import StateSpaceModel
from ryuushi.particle import Particle
from ryuushi.resampling import ResamplingScheme, ResamplerFactory
from ryuushi.output import Output
from ryuushi.observation import Observation

class SequentialMonteCarloFilter(ABC):
    """Abstract base class for SMC algorithms"""
    def __init__(self, model: StateSpaceModel, resampling_scheme: ResamplingScheme = ResamplingScheme.MULTINOMIAL):
        self.model = model
        self.resampler = ResamplerFactory.create(resampling_scheme)
        self.particles = []
        self.log_likelihood = 0.0
        self.time_step = 0
        self.time = 0

    @abstractmethod
    def initialize(self, num_particles: int, data: Any = None) -> None:
        pass

    @abstractmethod
    def step(self, observation: Observation, prev_observation: Observation = None) -> Output: # type: ignore
        pass

    @abstractmethod
    def run(self, data_sequence: List[Observation]) -> List[Output]:
        pass

    def effective_sample_size(self, particles: List[Particle]) -> float:
        """Calculate effective sample size"""
        weights = np.array([p.weight for p in particles])
        normalized_weights = weights / np.sum(weights)
        return 1.0 / np.sum(normalized_weights ** 2)

    def normalize_weights(self, particles: List[Particle]) -> List[Particle]:
        """Normalize particle weights"""
        total_weight = sum(p.weight for p in particles)
        for p in particles:
            p.weight /= total_weight
        return particles