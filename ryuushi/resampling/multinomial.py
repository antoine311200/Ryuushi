from typing import List
import numpy as np

from ryuushi.particle import Particle
from ryuushi.resampling.resampler import Resampler

class MultinomialResampler(Resampler):
    """Multinomial resampling implementation"""
    def resample(self, particles: List[Particle]) -> List[Particle]:
        weights = np.array([p.weight for p in particles])
        normalized_weights = (weights / np.sum(weights)).flatten()
        indices = np.random.choice(len(particles), size=len(particles), p=normalized_weights)

        new_particles = []
        for idx in indices:
            parent = particles[idx]
            new_particle = Particle(
                state=parent.state.copy(),
                weight=1.0/len(particles),
                parent=parent
            )
            new_particles.append(new_particle)

        return new_particles
