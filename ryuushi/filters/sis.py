from typing import Any, List
import numpy as np

from ryuushi.filters.smc import SequentialMonteCarloFilter
from ryuushi.models.ssm import StateSpaceModel
from ryuushi.observation import Observation
from ryuushi.output import Output
from ryuushi.particle import Particle
from ryuushi.distributions.bootstrap import BootstrapImportance
from ryuushi.resampling import ResamplingScheme

class SISFilter(SequentialMonteCarloFilter):
    """Sequential Importance Sampling (SIS) filter"""
    def __init__(self, model: StateSpaceModel, resampling_scheme: ResamplingScheme = ResamplingScheme.MULTINOMIAL):
        super().__init__(model, resampling_scheme)
        self.importance_distribution = BootstrapImportance(model)

    def initialize(self, num_particles: int, data: Any = None) -> None:
        """Initialize particles from initial state distribution"""
        self.particles = []
        for _ in range(num_particles):
            state = self.model.initial_state()
            parameters = None
            if hasattr(self.model, "initial_parameters"):
                parameters = self.model.initial_parameters()
                if parameters is not None:
                    parameters = np.atleast_1d(np.asarray(parameters, dtype=float))
            particle = Particle(state, 1.0/num_particles, parameters=parameters)
            self.particles.append(particle)
        self.time_step = 0
        self.time = 0

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

        self.particles = new_particles
        self.log_likelihood += log_likelihood_increment
        self.time += 1

        return Output(
            particles=self.particles,
            log_likelihood=self.log_likelihood,
            effective_sample_size=ess,
            diagnostics={"resampled": False}
        )

    def run(self, data_sequence: List[Any]) -> List[Output]:
        """Run SIR on a sequence of data"""
        outputs = []
        for observation, prev_observation in zip(data_sequence, [None] + data_sequence[:-1]):
            output = self.step(observation, prev_observation) # type: ignore
            outputs.append(output)
        return outputs

