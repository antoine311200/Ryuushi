from typing import Any, List, Optional, Tuple
import numpy as np

from ryuushi.models import StateSpaceModel
from ryuushi.filters import SIRFilter, SISFilter, LiuWestFilter
from ryuushi.resampling import ResamplingScheme
from ryuushi.observation import Observation

class StochasticVolatilityModel(StateSpaceModel):
    """Stochastic Volatility Model for financial time series

    Model equations:
    x_t = μ + φ(x_{t-1} - μ) + σ η_t, η_t ~ N(0,1)
    y_t = β exp(x_t / 2) ε_t, ε_t ~ N(0,1)

    where x_t is log-volatility, y_t is observed log-return
    Parameters: θ = (μ, φ, σ, β)
    """
    def __init__(self, mu: float = 0.0, phi: float = 0.95, sigma: float = 0.15, beta: float = 0.65, prior_params: Optional[dict] = None):
        self.mu = mu
        self.phi = phi
        self.sigma = sigma
        self.beta = beta

        # Prior distributions for parameters
        self.prior_params = prior_params or {
            'mu_mean': 0.0, 'mu_std': 1.0,
            'phi_mean': 0.9, 'phi_std': 0.1,
            'sigma_shape': 2.0, 'sigma_scale': 0.1,
            'beta_shape': 2.0, 'beta_scale': 0.5
        }

    def sample_prior(self) -> Tuple[float, float, float, float]:
        """Sample parameters from prior distributions"""
        params = self.prior_params

        mu = np.random.normal(params['mu_mean'], params['mu_std'])
        phi = np.random.normal(params['phi_mean'], params['phi_std'])
        phi = np.clip(phi, -0.99, 0.99)
        sigma = np.random.gamma(params['sigma_shape'], params['sigma_scale'])
        beta = np.random.gamma(params['beta_shape'], params['beta_scale'])

        return mu, phi, sigma, beta

    def transition(self, state: np.ndarray, time: int = 0, dt: float = 0.1, parameters: Optional[np.ndarray] = None) -> np.ndarray: # type: ignore
        """State transition: x_t = μ + φ(x_{t-1} - μ) + σ η_t"""
        x_prev = state
        mu, phi, sigma, beta = parameters if parameters is not None else (self.mu, self.phi, self.sigma, self.beta)
        x_new = mu + phi * (x_prev - mu) + sigma * np.random.normal()
        return x_new

    def log_likelihood(self, state: np.ndarray, observation: Any, time: int, parameters: Optional[np.ndarray] = None) -> float:
        """Observation likelihood: y_t ~ N(0, β^2 exp(x_t))"""
        if observation is None or parameters is None: return 0.0

        x_t = state
        mu, phi, sigma, beta = parameters
        y_t = observation.data

        variance = (beta ** 2) * np.exp(x_t)
        if variance <= 0: return -np.inf
        return -0.5 * (np.log(2 * np.pi * variance) + (y_t ** 2) / variance)

    def initial_state(self) -> np.ndarray:
        """Initial state: sample volatility and parameters from prior"""
        mu, phi, sigma, beta = self.sample_prior()

        # Initial volatility from stationary distribution: N(μ, σ^2/(1-φ^2))
        if abs(phi) < 1.0:
            stationary_var = (sigma ** 2) / (1 - phi ** 2)
            x0 = np.random.normal(mu, np.sqrt(stationary_var))
        else:
            x0 = np.random.normal(0, 1)
        return np.array(x0)

    def initial_parameters(self) -> np.ndarray:
        """Sample initial parameters from prior"""
        return np.array(self.sample_prior())


def generate_stochastic_volatility_data(mu: float = -1.0, phi: float = 0.95, sigma: float = 0.15, beta: float = 0.65, n_steps: int = 500) -> Tuple[List[float], List[float]]:
    np.random.seed(42)

    x = mu
    log_volatilities = [x]
    returns = []

    for t in range(n_steps):
        x = mu + phi * (x - mu) + sigma * np.random.normal()
        log_volatilities.append(x)

        volatility = beta * np.exp(x / 2)
        return_t = volatility * np.random.normal()
        returns.append(return_t)

    return log_volatilities[1:], returns

true_mu, true_phi, true_sigma, true_beta = -1.0, 0.95, 0.15, 0.65
n_steps = 500
log_volatilities, returns = generate_stochastic_volatility_data(true_mu, true_phi, true_sigma, true_beta, n_steps)

observations = [Observation(data=r, time=t) for t, r in enumerate(returns)]

model = StochasticVolatilityModel(
    mu=0.0, phi=0.8, sigma=0.3, beta=1.0,  # Initial guesses
    prior_params={
        'mu_mean': 0.0, 'mu_std': 2.0,      # Vague prior for mu
        'phi_mean': 0.9, 'phi_std': 0.2,    # Vague prior for phi
        'sigma_shape': 2.0, 'sigma_scale': 0.3,  # Vague prior for sigma
        'beta_shape': 2.0, 'beta_scale': 1.0     # Vague prior for beta
    }
)

filter = LiuWestFilter(
    model=model,
    resampling_scheme=ResamplingScheme.SYSTEMATIC,
    shrinkage_factor=0.99,
    kernel_bandwidth=0.05
)
filter.initialize(num_particles=1000)
outputs = filter.run(observations)

estimated_log_volatilities = [np.mean([p.state[0] for p in output.particles]) for output in outputs]
estimated_mus = [np.mean([p.parameters[0] for p in output.particles]) for output in outputs]
estimated_phis = [np.mean([p.parameters[1] for p in output.particles]) for output in outputs]
estimated_sigmas = [np.mean([p.parameters[2] for p in output.particles]) for output in outputs]
estimated_betas = [np.mean([p.parameters[3] for p in output.particles]) for output in outputs]

import matplotlib.pyplot as plt
plt.figure(figsize=(14, 10))
plt.subplot(3, 2, 1)
plt.plot(log_volatilities, label='True Log-Volatility', color='blue')
plt.plot(estimated_log_volatilities, label='Estimated Log-Volatility', color='orange', alpha=0.7)
plt.legend()
plt.title('Log-Volatility')
plt.subplot(3, 2, 2)
plt.plot(returns, label='Returns', color='green')
plt.legend()
plt.title('Returns')
plt.subplot(3, 2, 3)
plt.plot(estimated_mus, label='Estimated μ', color='red')
plt.axhline(true_mu, color='black', linestyle='--', label='True μ')
plt.legend()
plt.title('Parameter μ')
plt.subplot(3, 2, 4)
plt.plot(estimated_phis, label='Estimated φ', color='purple')
plt.axhline(true_phi, color='black', linestyle='--', label='True φ')
plt.legend()
plt.title('Parameter φ')
plt.subplot(3, 2, 5)
plt.plot(estimated_sigmas, label='Estimated σ', color='brown')
plt.axhline(true_sigma, color='black', linestyle='--', label='True σ')
plt.legend()
plt.title('Parameter σ')
plt.subplot(3, 2, 6)
plt.plot(estimated_betas, label='Estimated β', color='cyan')
plt.axhline(true_beta, color='black', linestyle='--', label='True β')
plt.legend()
plt.title('Parameter β')
plt.tight_layout()
plt.show()