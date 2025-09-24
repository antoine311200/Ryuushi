from typing import Any, List, Optional
import numpy as np

from ryuushi.filters.sis import SISFilter
from ryuushi.filters.smc import SequentialMonteCarloFilter
from ryuushi.models.ssm import StateSpaceModel
from ryuushi.distributions.bootstrap import BootstrapImportance
from ryuushi.resampling import ResamplingScheme
from ryuushi.output import Output
from ryuushi.particle import Particle
from ryuushi.observation import Observation

class SIRFilter(SISFilter):
    """Sequential Importance Resampling (SIR) filter"""

    def step(self, observation: Observation, prev_observation: Optional[Observation] = None) -> Output:
        """Perform one Sequential Importance Resampling (SIR) step"""
        new_particles = []
        log_likelihood_increment = 0.0

        for particle in self.particles:
            # Propose new state pi(x_t | y_t)
            proposed_state = self.importance_distribution.sample(particle, observation, prev_observation)

            # Calculate weight w_t = w_{t-1} * p(y_t | x_t)
            log_likelihood = self.model.log_likelihood(proposed_state, observation, self.time)
            weight = particle.weight * np.exp(log_likelihood)
            if np.isnan(weight) or np.isinf(weight): weight = 0.0

            # Create new particle at new time step
            new_particle = Particle(proposed_state, weight, particle, parameters=particle.parameters)
            new_particle.log_likelihood = log_likelihood
            new_particles.append(new_particle)

            log_likelihood_increment += weight * log_likelihood

        # Normalize weights
        new_particles = self.normalize_weights(new_particles)

        # Calculate effective sample size and resample if necessary
        ess = self.effective_sample_size(new_particles)
        if ess < len(new_particles) / 2:
            new_particles = self.resampler.resample(new_particles)

        self.particles = new_particles
        self.log_likelihood += log_likelihood_increment
        self.time_step += 1

        return Output(
            particles=self.particles,
            log_likelihood=self.log_likelihood,
            effective_sample_size=ess,
            diagnostics={"resampled": ess < len(new_particles) / 2}
        )
