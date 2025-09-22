from typing import Any

import numpy as np

from ryuushi.distributions.importance import ImportanceDistribution
from ryuushi.models.ssm import StateSpaceModel
from ryuushi.particle import Particle
from ryuushi.observation import Observation

class BootstrapImportance(ImportanceDistribution):
    """Bootstrap proposal distribution (uses transition kernel)"""
    def __init__(self, model: StateSpaceModel):
        self.model = model

    def sample(self, particle: Particle, observation: Observation, prev_observation: Observation = None) -> np.ndarray: # type: ignore
        # Bootstrap importance consists in using the transition kernel as the importance distribution
        return self.model.transition(particle.state, observation.time, (observation.time - prev_observation.time) if prev_observation else observation.time)

    def log_density(self, state: np.ndarray, particle: Particle, data: Any, time: int) -> float:
        # For bootstrap filter, we don't need the log density of the importance distribution
        # as weights are computed using likelihood only
        return 0.0