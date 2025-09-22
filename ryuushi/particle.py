from typing import Optional

import numpy as np

class Particle:
    """Represents a single particle in SMC

    Attributes:
        state (np.ndarray): The state of the particle.
        weight (float): The weight of the particle.
        parent (Optional[Particle]): The parent particle from the previous time step.
        log_likelihood (float): The log-likelihood of the particle.
    """
    def __init__(self, state: np.ndarray, weight: float, parent: Optional['Particle'] = None):
        self.state = state
        self.weight = weight
        self.parent = parent
        self.log_likelihood = 0.0
