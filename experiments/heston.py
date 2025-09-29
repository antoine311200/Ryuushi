from typing import Any
from time import time

import numpy as np

from ryuushi.models import StateSpaceModel
from ryuushi.filters import SIRFilter, SISFilter, AuxiliaryParticleFilter
from ryuushi.resampling import ResamplingScheme
from ryuushi.observation import Observation

class HestonModel(StateSpaceModel):

    def __init__(self, mu: float, kappa: float, theta: float, sigma: float, rho: float):
        self.mu = mu          # Drift of the asset price
        self.kappa = kappa    # Rate of mean reversion of variance
        self.theta = theta    # Long-term mean of variance
        self.sigma = sigma    # Volatility of variance
        self.rho = rho        # Correlation between asset and variance

    def transition(self, state: np.ndarray, time: int, dt: float, parameters: Any = None) -> np.ndarray: # type: ignore
        """ S_t+1 = S_t + mu * S_t * dt + sqrt(V_t) * S_t * dW1
            V_t+1 = V_t + kappa * (theta - V_t) * dt + sigma * sqrt(V_t) * dW2
            dW1, dW2 are correlated Brownian motions with correlation rho
        """
        S_t, V_t = state
        dW1 = np.random.randn() * np.sqrt(dt)
        dW2 = self.rho * dW1 + np.sqrt(1 - self.rho**2) * np.random.randn() * np.sqrt(dt)
        V_next = np.abs(V_t + self.kappa * (self.theta - V_t) * dt + self.sigma * np.sqrt(V_t) * dW2)
        S_next = S_t + self.mu * S_t * dt + np.sqrt(V_t) * S_t * dW1
        return np.array([S_next, V_next])

    def log_likelihood(self, state: np.ndarray, observation: Any, time: int, parameters: np.ndarray = None) -> float: # type: ignore
        """ Observation model: Y_t = S_t + epsilon_t, epsilon_t ~ N(0, (sqrt(V_t) * S_t)^2) """
        S_t, V_t = state
        y_t = observation.data
        obs_variance = (np.sqrt(V_t) * S_t)**2
        if obs_variance <= 0:
            return -np.inf
        log_lik = -0.5 * np.log(2 * np.pi * obs_variance) - 0.5 * ((y_t - S_t)**2 / obs_variance)
        return log_lik

    def initial_state(self) -> np.ndarray:
        S_0 = 100.0  # Initial asset price
        V_0 = self.theta  # Start at long-term mean of variance
        return np.array([S_0, V_0])

    def initial_parameters(self) -> np.ndarray:
        return np.array([self.mu, self.kappa, self.theta, self.sigma, self.rho])


mu, kappa, theta, sigma, rho = 0.05, 1.0, 0.04, 0.2, -0.7
model = HestonModel(mu=mu, kappa=kappa, theta=theta, sigma=sigma, rho=rho)

def generate_data(model: HestonModel, num_steps: int, dt: float = 1.0) -> tuple[list[np.ndarray], list[Observation]]:
    states = []
    observations = []
    current_state = model.initial_state()
    for t in range(num_steps):
        current_state = model.transition(current_state, t, dt)
        states.append(current_state)
        observation_noise = np.sqrt(current_state[1]) * current_state[0] * np.random.randn()
        observations.append(Observation(data=current_state[0] + observation_noise, time=t))
    return states, observations

T = 1.0
num_steps = 250
true_states, observations = generate_data(model, num_steps, dt=1.0)#T / num_steps)

filters = {
    "SIR-systematic": SIRFilter(model, ResamplingScheme.SYSTEMATIC),
    "APF-systematic": AuxiliaryParticleFilter(model, ResamplingScheme.SYSTEMATIC),
}

num_particles = 500
results = {}
for name, filter_instance in filters.items():
    filter_instance.initialize(num_particles)
    start_time = time()
    outputs = filter_instance.run(observations)
    end_time = time()
    results[name] = {
        "outputs": outputs,
        "time": end_time - start_time
    }
    print(f"Filter {name} completed in {end_time - start_time:.2f} seconds")

    rmse = np.sqrt(np.mean([(np.mean([p.state[0] for p in output.particles]) - true_states[t][0])**2 for t, output in enumerate(outputs)]))
    print(f"Filter {name} RMSE: {rmse:.4f}")

##################################################################
##############          Plotting Results           ###############
##################################################################

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context("paper")
sns.set_style("whitegrid")
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot([s[0] for s in true_states], label='True Asset Price', color='black')
for name, result in results.items():
    estimated_prices = [np.mean([p.state[0] for p in output.particles]) for output in result["outputs"]]
    plt.plot(estimated_prices, label=f'Estimated by {name}')
plt.xlabel('Time Step')
plt.ylabel('Asset Price')
plt.title('Heston Model: True vs Estimated Asset Prices')
plt.legend()
plt.subplot(2, 1, 2)
plt.plot([s[1] for s in true_states], label='True Variance', color='black')
for name, result in results.items():
    estimated_variances = [np.mean([p.state[1] for p in output.particles]) for output in result["outputs"]]
    plt.plot(estimated_variances, label=f'Estimated by {name}')
plt.xlabel('Time Step')
plt.ylabel('Variance')
plt.title('Heston Model: True vs Estimated Variance')
plt.legend()
plt.show()
