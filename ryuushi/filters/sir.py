from typing import Any, List
import numpy as np

from ryuushi.filters.smc import SequentialMonteCarloFilter
from ryuushi.models.ssm import StateSpaceModel
from ryuushi.output import Output
from ryuushi.particle import Particle
from ryuushi.distributions.bootstrap import BootstrapImportance
from ryuushi.resampling import ResamplingScheme, ResamplerFactory

class SIRFilter(SequentialMonteCarloFilter):
    """Sequential Importance Resampling (SIR) filter"""
    def __init__(self, model: StateSpaceModel, resampling_scheme: ResamplingScheme = ResamplingScheme.MULTINOMIAL):
        super().__init__(model, resampling_scheme)
        self.importance_distribution = BootstrapImportance(model)

    def initialize(self, num_particles: int, data: Any = None) -> None:
        """Initialize particles from initial state distribution"""
        self.particles = []
        for _ in range(num_particles):
            state = self.model.initial_state()
            particle = Particle(state, 1.0/num_particles)
            self.particles.append(particle)
        self.time = 0

    def step(self, observation: Any) -> Output:
        """Perform one Sequential Importance Resampling (SIR) step"""
        new_particles = []
        log_likelihood_increment = 0.0

        for particle in self.particles:
            # Propose new state pi(x_t | y_t)
            proposed_state = self.importance_distribution.sample(particle, observation, self.time)

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
        if ess < len(new_particles) / 2:
            new_particles = self.resampler.resample(new_particles)

        self.particles = new_particles
        self.log_likelihood += log_likelihood_increment
        self.time += 1

        return Output(
            particles=self.particles,
            log_likelihood=self.log_likelihood,
            effective_sample_size=ess,
            diagnostics={"resampled": ess < len(new_particles) / 2}
        )

    def run(self, data_sequence: List[Any]) -> List[Output]:
        """Run SIR on a sequence of data"""
        outputs = []
        for observation in data_sequence:
            output = self.step(observation)
            outputs.append(output)
        return outputs
