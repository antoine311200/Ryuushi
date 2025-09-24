
from typing import Any, List, Optional
import numpy as np

from ryuushi.filters.sis import SISFilter
from ryuushi.models.ssm import StateSpaceModel
from ryuushi.distributions.bootstrap import BootstrapImportance
from ryuushi.resampling import ResamplingScheme
from ryuushi.output import Output
from ryuushi.particle import Particle
from ryuushi.observation import Observation

class AuxiliaryParticleFilter(SISFilter):
    """Auxiliary Particle Filter (APF)"""

    def predictive_weights(self, observation: Observation, dt: float) -> List[float]: # type: ignore
        # First stage: generate mu_t from p(x_t | x_t-1) and compute predictive weights lambda ~ w_{t-1} * p(y_t | mu_t)
        predictive_weights = []
        for particle in self.particles:
            predicted_state = self.model.transition(particle.state, self.time, dt)
            log_likelihood = self.model.log_likelihood(predicted_state, observation, self.time)
            predictive_weight = particle.weight * np.exp(log_likelihood)
            if np.isnan(predictive_weight) or np.isinf(predictive_weight): predictive_weight = 0.0
            predictive_weights.append(predictive_weight)

        # Normalize predictive weights
        total_predictive_weight = sum(predictive_weights)
        if total_predictive_weight == 0:
            predictive_weights = [1.0 / len(self.particles)] * len(self.particles)
        else:
            predictive_weights = [w / total_predictive_weight for w in predictive_weights]

        return predictive_weights


    def step(self, observation: Observation, prev_observation: Optional[Observation] = None) -> Output: # type: ignore
        """Perform one Auxiliary Particle Filter (APF) step assuming bootstrap importance distribution and mu_t = p(x_t | x_t-1)"""
        new_particles = []
        log_likelihood_increment = 0.0
        dt = observation.time - (prev_observation.time if prev_observation else 0)

        predictive_weights = self.predictive_weights(observation, dt)

        # Resample indices based on predictive weights
        resampled_indices = self.resampler.resample_indices(predictive_weights)
        resampled_particles = [self.particles[i] for i in resampled_indices]

        for particle in resampled_particles:
            # Propose new state pi(x_t | y_t)
            proposed_state = self.importance_distribution.sample(particle, observation, prev_observation)

            # Calculate weight w_t = p(y_t | x_t) / p(y_t | mu_t)
            log_likelihood = self.model.log_likelihood(proposed_state, observation, self.time)
            # predictive_log_likelihood = self.model.log_likelihood(self.model.transition(particle.state, self.time, dt), observation, self.time)

            # APF weight correction
            weight = np.exp(log_likelihood)# - predictive_log_likelihood) if predictive_log_likelihood > -np.inf else 1.0
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