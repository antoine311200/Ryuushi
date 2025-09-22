from typing import Any, List
import numpy as np

from ryuushi.filters.smc import SequentialMonteCarloFilter
from ryuushi.models.ssm import StateSpaceModel
from ryuushi.output import Output
from ryuushi.particle import Particle
from ryuushi.distributions.bootstrap import BootstrapImportance
from ryuushi.resampling import ResamplingScheme, ResamplerFactory
from ryuushi.filters.sir import SIRFilter

class SISFilter(SIRFilter):
    """Sequential Importance Sampling (SIS) filter"""

    def step(self, observation: Any, prev_observation: Any = None) -> Output:
        """Perform one Sequential Importance Sampling (SIS) step"""
        new_particles = []
        log_likelihood_increment = 0.0

        for particle in self.particles:
            # Propose new state pi(x_t | y_t)
            proposed_state = self.importance_distribution.sample(particle, observation, prev_observation)

            # Calculate weight w_t = w_{t-1} * p(y_t | x_t)
            log_likelihood = self.model.log_likelihood(proposed_state, observation, self.time)
            weight = particle.weight * np.exp(log_likelihood)

            # Create new particle at new time step
            new_particle = Particle(proposed_state, weight, particle)
            new_particle.log_likelihood = log_likelihood
            new_particles.append(new_particle)

            log_likelihood_increment += weight * log_likelihood

        # Normalize weights
        new_particles = self.normalize_weights(new_particles)

        # Calculate effective sample size and resample if necessary
        ess = self.effective_sample_size(new_particles)

        self.particles = new_particles
        self.log_likelihood += log_likelihood_increment
        self.time += 1

        return Output(
            particles=self.particles,
            log_likelihood=self.log_likelihood,
            effective_sample_size=ess,
            diagnostics={"resampled": False}
        )