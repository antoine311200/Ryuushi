from typing import Optional

import numpy as np

from ryuushi.filters.sir import SIRFilter
from ryuushi.models.ssm import StateSpaceModel
from ryuushi.output import Output
from ryuushi.observation import Observation
from ryuushi.particle import Particle
from ryuushi.resampling.resampler import ResamplingScheme

class LiuWestFilter(SIRFilter):

    def __init__(self, model: StateSpaceModel, resampling_scheme: ResamplingScheme, shrinkage_factor: float = 0.95, kernel_bandwidth: float = 0.1):
        super().__init__(model, resampling_scheme)
        self.shrinkage_factor = shrinkage_factor  # a in Liu-West paper
        self.kernel_bandwidth = kernel_bandwidth  # h in Liu-West paper

    def step(self, observation: Observation, prev_observation: Optional[Observation] = None) -> Output: # type: ignore
        """Liu-West filter step with parameter updating"""
        output = super().step(observation, prev_observation)

        if len(output.particles) > 0:
            updated_particles = self._update_parameters(output.particles)

            output = Output(
                particles=updated_particles,
                log_likelihood=output.log_likelihood,
                effective_sample_size=output.effective_sample_size,
                diagnostics=output.diagnostics
            )
        return output

    def _update_parameters(self, particles: list) -> list:
        """Apply Liu-West parameter update to particles"""
        params = np.array([p.parameters for p in particles])
        weights = np.array([p.weight for p in particles])

        weighted_mean = np.average(params, axis=0, weights=weights)
        weighted_cov = np.cov(params.T, aweights=weights)

        a = self.shrinkage_factor
        h = self.kernel_bandwidth
        h2 = h ** 2

        updated_particles = []
        for particle in particles:
            # Shrinkage towards the mean
            shrunk_params = a * particle.parameters + (1 - a) * weighted_mean

            # Add kernel noise
            if weighted_cov.ndim == 0:
                noise = np.random.normal(0, np.sqrt(h2 * weighted_cov))
            else:
                noise = np.random.multivariate_normal(
                    np.zeros(len(particle.parameters)),
                    h2 * weighted_cov + 1e-6 * np.eye(len(particle.parameters))  # Regularization term
                )
            new_params = shrunk_params + noise

            updated_particle = Particle(particle.state.copy(), particle.weight, particle.parent, new_params)
            updated_particles.append(updated_particle)

        return updated_particles