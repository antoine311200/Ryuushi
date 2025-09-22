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
    def __init__(self, state: np.ndarray, weight: float, parent: Optional['Particle'] = None, parameters: Optional[np.ndarray] = None):
        self.state = state
        self.parameters = parameters
        self.weight = weight
        self.parent = parent
        self.log_likelihood = 0.0

    def history(self) -> list:
        """Retrieve the ancestry of the particle"""
        lineage = []
        current = self
        while current is not None:
            lineage.append(current)
            current = current.parent
        return lineage[::-1]

    def __repr__(self):
        return f"Particle(state={self.state}, weight={self.weight}, log_likelihood={self.log_likelihood})"