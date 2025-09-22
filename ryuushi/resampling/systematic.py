from typing import List, Any

import numpy as np

from ryuushi.resampling import Resampler
from ryuushi.particle import Particle

class SystematicResampler(Resampler):
    """Systematic resampling implementation"""

    def resample(self, particles: List[Particle]) -> List[Particle]:
        weights = np.array([p.weight for p in particles])
        normalized_weights = weights / np.sum(weights)

        n = len(particles)
        positions = (np.arange(n) + np.random.uniform(0, 1)) / n
        cumulative_weights = np.cumsum(normalized_weights)

        new_particles = []
        i, j = 0, 0
        while i < n:
            if positions[i] < cumulative_weights[j]:
                parent = particles[j]
                new_particle = Particle(
                    state=parent.state.copy(), weight=1.0 / n, parent=parent
                )
                new_particles.append(new_particle)
                i += 1
            else:
                j += 1

        return new_particles
