from typing import Any
from time import time

import numpy as np

from ryuushi.models import StateSpaceModel
from ryuushi.filters import SIRFilter, SISFilter, AuxiliaryParticleFilter
from ryuushi.resampling import ResamplingScheme
from ryuushi.observation import Observation

class HiddenMarkovSV(StateSpaceModel):

    def __init__(self, rho: float, sigma: float, tau: float):
        self.rho = rho      # Persistence of volatility
        self.sigma = sigma  # Long-term mean of log-volatility
        self.tau = tau      # Volatility of log-volatility

    def transition(self, state: np.ndarray, time: int, dt: float, parameters: Any = None) -> np.ndarray: # type: ignore
        # X_n+1 = rho * X_n + sigma * eta_n, eta_n ~ N(0,1)
        return self.rho * state + self.sigma * np.random.randn() #* np.sqrt(dt)

    def log_likelihood(self, state: np.ndarray, observation: Any, time: int, parameters: np.ndarray = None) -> float: # type: ignore
        # Y_n = tau * exp(X_n / 2) * epsilon_n, epsilon_n ~ N(0,1)
        if observation is None: return 0.0
        if self.tau <= 0: return 0.0

        xt, yt = state, observation.data
        variance = (self.tau ** 2) * np.exp(xt)
        if variance <= 0: return -np.inf
        return -0.5 * (yt**2) / variance - 0.5 * np.log(2 * np.pi * variance)

    def initial_state(self) -> np.ndarray:
        # X0 ~ N(0, sigma^2 / (1 - rho^2))
        return np.array([self.sigma / np.sqrt(1 - self.rho**2) * np.random.randn()])

    def initial_parameters(self) -> np.ndarray:
        return np.array([self.rho, self.sigma, self.tau])

model = HiddenMarkovSV(rho=0.9, sigma=1.05, tau=0.7)

def generate_data(model: HiddenMarkovSV, num_steps: int, dt: float = 1.0) -> tuple[list[np.ndarray], list[Observation]]:
    states = []
    observations = []
    current_state = model.initial_state()
    for t in range(num_steps):
        current_state = model.transition(current_state, t, dt)
        states.append(current_state)
        observation_noise = model.tau * np.exp(current_state[0] / 2) * np.random.randn()
        observations.append(Observation(data=observation_noise, time=t))
    return states, observations

num_steps = 500
true_states, observations = generate_data(model, num_steps)

filters = {
    "SIR-systematic": SIRFilter(model, ResamplingScheme.SYSTEMATIC),
    "SIR-multinomial": SIRFilter(model, ResamplingScheme.MULTINOMIAL),
    "APF-systematic": AuxiliaryParticleFilter(model, ResamplingScheme.SYSTEMATIC),
    "APF-multinomial": AuxiliaryParticleFilter(model, ResamplingScheme.MULTINOMIAL),
}

results = {}
num_particles = 500
for name, filter_instance in filters.items():
    print(f"Running filter: {name}")
    start_time = time()
    filter_instance.initialize(num_particles)
    outputs = filter_instance.run(observations)
    rmse = np.sqrt(np.mean([(np.mean([p.state for p in output.particles]) - true_states[t][0])**2 for t, output in enumerate(outputs)]))
    results[name] = {"outputs": outputs, "rmse": rmse}
    end_time = time()
    print(f"Completed in {end_time - start_time:.2f} seconds with RMSE: {rmse:.4f}")

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context("paper")
sns.set_style("whitegrid")

time_steps = np.arange(num_steps)
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.suptitle("Hidden Markov Model with Stochastic Volatility")
plt.plot(time_steps, [s[0] for s in true_states], label="True Log-Volatility", color="black", linestyle="--")
for name, outputs in results.items():
    estimated_states = [np.mean([p.state for p in output.particles]) for output in outputs["outputs"]]
    plt.plot(time_steps, estimated_states, label=f"Log-Volatility ({name}) | RMSE: {outputs['rmse']:.4f}")
plt.xlabel("Time")
plt.ylabel("Log-Volatility")
plt.title("Estimated Log-Volatilities")
plt.legend()
plt.subplot(2, 1, 2)
plt.subplots_adjust(hspace=0.4)
# Plot return observations scatter and estimated returns
plt.plot(time_steps, [obs.data for obs in observations], label="Observed Returns", color="black", linestyle="--", linewidth=1)
for name, outputs in results.items():
    # sigma = model.tau * np.exp(np.mean([p.state for p in outputs["outputs"][-1].particles]) / 2)
    # epsilon = np.array([obs.data / sigma for obs in observations]) / sigma
    # plt.plot(time_steps, 2 *epsilon, label="+2 std dev ("+name+")", linestyle=":", linewidth=0.5)
    # plt.plot(time_steps, -2 *epsilon, label="-2 std dev ("+name+")", linestyle=":", linewidth=0.5)
    # plt.fill_between(time_steps, 2 * epsilon, -2 * epsilon, alpha=0.5, label="±2 std dev")
    # plt.fill_between(time_steps, 1 * epsilon, -1 * epsilon, alpha=0.8, label="±1 std dev")

    estimated_returns = [np.mean([p.state for p in output.particles]) * model.tau * np.exp(np.mean([p.state for p in output.particles]) / 2) for output in outputs["outputs"]]
    plt.plot(time_steps, estimated_returns, label=f"Estimated Returns ({name})", alpha=0.75, linewidth=0.5)
    # fill ±1 and ±2 std dev
    estimated_std_devs = [np.std([p.state for p in output.particles]) * model.tau * np.exp(np.mean([p.state for p in output.particles]) / 2) for output in outputs["outputs"]]
    plt.fill_between(time_steps, np.array(estimated_returns) + np.array(estimated_std_devs), np.array(estimated_returns) - np.array(estimated_std_devs), alpha=0.3, label=f"±1 std dev ({name})")
    plt.fill_between(time_steps, np.array(estimated_returns) + 2 * np.array(estimated_std_devs), np.array(estimated_returns) - 2 * np.array(estimated_std_devs), alpha=0.1, label=f"±2 std dev ({name})")
plt.xlabel("Time")
plt.ylabel("Returns")
plt.title("Observed vs Estimated Returns")
plt.legend()
# for name, outputs in results.items():
#     estimated_variances = [np.var([p.state for p in output.particles]) for output in outputs]
#     plt.plot(time_steps, estimated_variances, label=f"Estimated Variance ({name})")
# plt.xlabel("Time")
# plt.ylabel("Variance")
# plt.title("Estimated Variance of Log-Volatility")
# plt.legend()
plt.show()