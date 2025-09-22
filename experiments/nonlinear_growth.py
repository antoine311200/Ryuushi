from typing import Any

import numpy as np

from ryuushi.models import StateSpaceModel
from ryuushi.filters import SIRFilter, SISFilter
from ryuushi.resampling import ResamplingScheme
from ryuushi.observation import Observation


class NonlinearGrowthModel(StateSpaceModel):
    """Simple nonlinear growth model with saturation"""

    def __init__(self, growth_rate: float = 0.8, capacity: float = 100.0, process_noise: float = 0.1, measurement_noise: float = 0.3):
        self.growth_rate = growth_rate
        self.capacity = capacity
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

    def transition(self, state: np.ndarray, time: int, dt: float) -> np.ndarray: # type: ignore
        """Logistic growth with noise"""
        # Logistic growth: x_{t+1} = x_t + r * x_t * (1 - x_t/K)
        current_population = state[0]
        growth = (
            self.growth_rate
            * current_population
            * (1 - current_population / self.capacity)
        )
        deterministic_update = current_population + growth * dt

        noise = np.random.normal(0, self.process_noise)
        new_population = max(0.1, deterministic_update + noise)

        return np.array([new_population])

    def log_likelihood(self, state: np.ndarray, observation: Any, time: int) -> float:
        """Log-normal likelihood for positive observations"""
        if observation is None: return 0.0
        if self.measurement_noise <= 0: return 0.0
        true_value = state[0]
        if true_value <= 0: return -np.inf  # Invalid state

        log_diff = np.log(observation.data) - np.log(true_value)
        return -0.5 * (log_diff**2) / (self.measurement_noise**2) - np.log(
            observation.data * self.measurement_noise
        )

    def initial_state(self) -> np.ndarray:
        """Start with small population"""
        return np.array([1.0])

model = NonlinearGrowthModel(
    growth_rate=1,
    capacity=100.0,
    process_noise=0.01,
    measurement_noise=5.0
)
model_clean = NonlinearGrowthModel(
    growth_rate=1,
    capacity=100.0,
    process_noise=0.0,
    measurement_noise=0.0
)

def generate_nonlinear_growth_data(model: NonlinearGrowthModel, time_steps: np.ndarray):
    """Generate synthetic data from the nonlinear growth model"""
    true_states = []
    observations = []
    current_state = model.initial_state()

    prev_t = 0.0
    for t in time_steps:
        dt = t - prev_t
        current_state = model.transition(current_state, t, dt)
        true_states.append(current_state.copy())

        observation = max(0.1, current_state[0] + np.random.normal(0, model.measurement_noise))
        observations.append(Observation(data=observation, time=t))

        prev_t = t

    return true_states, observations

n_time_steps = 150
final_time = 10.0
time_steps = np.linspace(0, final_time, n_time_steps)

true_states, observations = generate_nonlinear_growth_data(model, time_steps)
clean_states, _ = generate_nonlinear_growth_data(model_clean, time_steps)

filter = SIRFilter(model, resampling_scheme=ResamplingScheme.MULTINOMIAL)
filter.initialize(num_particles=150)
outputs = filter.run(observations)

estimated_populations = [np.mean([p.state for p in output.particles]) for output in outputs]


import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.plot(time_steps, true_states, label="True Population", color="blue")
plt.plot(time_steps, clean_states, label="Clean Population", color="green", linestyle=":")
plt.plot(time_steps, estimated_populations, label="Estimated Population", color="orange", linestyle="--")
plt.scatter([obs.time for obs in observations], [obs.data for obs in observations], label="Observations", color="red", s=10, alpha=0.5)
plt.xlabel("Time")
plt.ylabel("Population")
plt.title("Nonlinear Growth Model: True vs Estimated Population")
plt.legend()
plt.grid()
plt.show()
