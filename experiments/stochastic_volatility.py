from typing import Any, List, Optional, Tuple
from matplotlib.pylab import beta
import numpy as np
from scipy.stats import norm

from ryuushi.models import StateSpaceModel
from ryuushi.filters import SIRFilter, SISFilter, LiuWestFilter
from ryuushi.resampling import ResamplingScheme
from ryuushi.observation import Observation

# class StochasticVolatilityModel(StateSpaceModel):
#     """Stochastic Volatility Model for financial time series

#     Model equations:
#     x_t = μ + φ(x_{t-1} - μ) + σ η_t, η_t ~ N(0,1)
#     y_t = β exp(x_t / 2) ε_t, ε_t ~ N(0,1)

#     where x_t is log-volatility, y_t is observed log-return
#     Parameters: θ = (μ, φ, σ, β)
#     """
#     def __init__(self, mu: float = 0.0, phi: float = 0.95, sigma: float = 0.15, beta: float = 0.65, prior_params: Optional[dict] = None):
#         self.mu = mu
#         self.phi = phi
#         self.sigma = sigma
#         self.beta = beta

#         # Prior distributions for parameters
#         self.prior_params = prior_params or {
#             'mu_mean': 0.0, 'mu_std': 1.0,
#             'phi_mean': 0.9, 'phi_std': 0.1,
#             'sigma_shape': 2.0, 'sigma_scale': 0.1,
#             'beta_shape': 2.0, 'beta_scale': 0.5
#         }

#     def sample_prior(self) -> Tuple[float, float, float, float]:
#         """Sample parameters from prior distributions"""
#         params = self.prior_params

#         mu = np.random.normal(params['mu_mean'], params['mu_std'])
#         phi = np.random.normal(params['phi_mean'], params['phi_std'])
#         phi = np.clip(phi, -0.99, 0.99)
#         sigma = np.random.gamma(params['sigma_shape'], params['sigma_scale'])
#         beta = np.random.gamma(params['beta_shape'], params['beta_scale'])

#         return mu, phi, sigma, beta

#     def transition(self, state: np.ndarray, time: int = 0, dt: float = 0.1, parameters: Optional[np.ndarray] = None) -> np.ndarray: # type: ignore
#         """State transition: x_t = μ + φ(x_{t-1} - μ) + σ η_t"""
#         x_prev = state
#         mu, phi, sigma, beta = parameters if parameters is not None else (self.mu, self.phi, self.sigma, self.beta)
#         x_new = mu + phi * (x_prev - mu) + sigma * np.random.normal()
#         return x_new

#     def log_likelihood(self, state: np.ndarray, observation: Any, time: int, parameters: Optional[np.ndarray] = None) -> float:
#         """Observation likelihood: y_t ~ N(0, β^2 exp(x_t))"""
#         if observation is None or parameters is None: return 0.0

#         x_t = state
#         mu, phi, sigma, beta = parameters
#         y_t = observation.data

#         variance = (beta ** 2) * np.exp(x_t)
#         if variance <= 0: return -np.inf
#         return -0.5 * (np.log(2 * np.pi * variance) + (y_t ** 2) / variance)

#     def initial_state(self) -> np.ndarray:
#         """Initial state: sample volatility and parameters from prior"""
#         # mu, phi, sigma, beta = self.sample_prior()
#         mu, phi, sigma, beta = self.mu, self.phi, self.sigma, self.beta

#         # Initial volatility from stationary distribution: N(μ, σ^2/(1-φ^2))
#         if abs(phi) < 1.0:
#             stationary_var = (sigma ** 2) / (1 - phi ** 2)
#             x0 = np.random.normal(mu, np.sqrt(stationary_var))
#         else:
#             x0 = np.random.normal(0, 1)
#         return np.array(x0)

#     def initial_parameters(self) -> np.ndarray:
#         """Sample initial parameters from prior"""
#         return np.array(self.sample_prior())


class StochasticVolatilityModel(StateSpaceModel):
    """
    Stochastic Volatility Model for financial time series

    State equation (latent volatility):
        x_t = μ + φ*(x_{t-1} - μ) + σ_η * η_t, where η_t ~ N(0,1)

    Observation equation (log returns):
        y_t = ε_t * exp(x_t/2), where ε_t ~ N(0,1)
        or equivalently: log(y_t²) ≈ x_t + log(ε_t²)

    Parameters: θ = [μ, φ, σ_η]
    - μ: long-term mean of log-volatility
    - φ: persistence parameter (|φ| < 1 for stationarity)
    - σ_η: volatility of volatility
    """

    def __init__(self, true_parameters: Optional[np.ndarray] = None):
        """
        Initialize with either true parameters or reasonable defaults
        """
        if true_parameters is None:
            # Reasonable default parameters for financial data
            self.true_parameters = np.array([-1.0, 0.95, 0.15])
        else:
            self.true_parameters = true_parameters.copy()

        # Fixed parameters (unknown in practice, used for simulation)
        self._mu = self.true_parameters[0]
        self._phi = self.true_parameters[1]
        self._sigma_eta = self.true_parameters[2]

        # Validate parameters
        assert (
            abs(self._phi) < 1.0
        ), "Persistence parameter |φ| must be < 1 for stationarity"
        assert self._sigma_eta > 0, "Volatility of volatility must be positive"

    def transition(
        self,
        state: np.ndarray,
        time: int = 0,
        dt: float = 1.0,
        parameters: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        State transition: x_t = μ + φ*(x_{t-1} - μ) + σ_η * η_t
        """
        if parameters is not None:
            mu, phi, sigma_eta = parameters
        else:
            mu, phi, sigma_eta = self._mu, self._phi, self._sigma_eta

        # Sample innovation
        eta = np.random.normal(0, 1, state.shape)

        # AR(1) process for log-volatility
        next_state = mu + phi * (state - mu) + sigma_eta * eta * np.sqrt(dt)

        return next_state

    def log_likelihood(
        self,
        state: np.ndarray,
        observation: Any,
        time: int,
        parameters: Optional[np.ndarray] = None,
    ) -> float:
        """
        Observation likelihood: p(y_t | x_t)
        Using the log-squared returns transformation for better numerical stability
        """
        # Transform observation: z_t = log(y_t² + c) where c is small constant for stability
        c = 1e-6  # small constant to avoid log(0)
        observation = observation.data
        z_t = np.log(observation**2 + c)

        # The approximate observation equation: z_t ≈ x_t + ν_t
        # where ν_t = log(ε_t²) has mean ≈ -1.27 and variance ≈ π²/2
        nu_mean = -1.27036  # E[log(ε_t²)] where ε_t ~ N(0,1)
        nu_variance = np.pi**2 / 2  # Var[log(ε_t²)]

        # Log-likelihood using Gaussian approximation
        residual = z_t - state - nu_mean
        log_lik = norm.logpdf(residual, 0, np.sqrt(nu_variance))

        return float(log_lik)

    def initial_state(self) -> np.ndarray:
        """
        Initial state distribution: x_0 ~ N(μ, σ_η²/(1-φ²))
        """
        # Stationary distribution of AR(1) process
        stationary_variance = self._sigma_eta**2 / (1 - self._phi**2)
        x0 = np.random.normal(self._mu, np.sqrt(stationary_variance))

        return np.array([x0])

    def initial_parameters(self) -> np.ndarray:
        """
        Initial guess for parameter estimation
        Reasonable starting values for EM algorithm
        """
        # Start with values slightly different from true parameters
        return np.array([-0.5, 0.8, 0.2])  # [μ_init, φ_init, σ_η_init]

    def sample_prior(self) -> Tuple[float, float, float, float]:
        """
        Sample from prior distribution over parameters
        Useful for Bayesian methods or initialization
        """
        # Priors for stochastic volatility parameters
        mu_prior = np.random.normal(-1.0, 1.0)  # N(-1, 1)
        phi_prior = np.random.beta(20, 1.5)  # Beta favoring high persistence
        sigma_eta_prior = np.random.gamma(2, 10)  # Gamma for positive values
        x0_prior = np.random.normal(mu_prior, 0.5)  # Initial state

        return (
            float(mu_prior),
            float(phi_prior),
            float(sigma_eta_prior),
            float(x0_prior),
        )


def generate_stochastic_volatility_data(
    mu: float = -1.0,
    phi: float = 0.95,
    sigma: float = 0.15,
    # beta: float = 0.65,
    n_steps: int = 500,
) -> Tuple[List[float], List[float]]:
    np.random.seed(42)

    x = mu
    log_volatilities = [x]
    returns = []

    for t in range(n_steps):
        x = mu + phi * (x - mu) + sigma * np.random.normal()
        log_volatilities.append(x)

        beta = 1.0
        volatility = beta * np.exp(x / 2)
        return_t = volatility * np.random.normal()
        returns.append(return_t)

    return log_volatilities[1:], returns


true_mu, true_phi, true_sigma, true_beta = -1.0, 0.5, 0.15, 0.65
n_steps = 200
log_volatilities, returns = generate_stochastic_volatility_data(
    true_mu, true_phi, true_sigma, n_steps
)

observations = [Observation(data=r, time=t) for t, r in enumerate(returns)]

model = StochasticVolatilityModel()
    # mu=-1.0,
    # phi=0.5,
    # sigma=0.3,
    # beta=1.0,  # Initial guesses
    # prior_params={
    #     "mu_mean": -1.0,
    #     "mu_std": 2.0,  # Vague prior for mu
    #     "phi_mean": 0.9,
    #     "phi_std": 0.2,  # Vague prior for phi
    #     "sigma_shape": 2.0,
    #     "sigma_scale": 0.3,  # Vague prior for sigma
    #     "beta_shape": 2.0,
    #     "beta_scale": 1.0,  # Vague prior for beta
    # },
# )

filters = {
    # "SIR": SIRFilter(model, ResamplingScheme.SYSTEMATIC),
    "Liu-West": LiuWestFilter(
        model=model,
        resampling_scheme=ResamplingScheme.SYSTEMATIC,
        shrinkage_factor=0.95,
        kernel_bandwidth=0.1,
    ),
}

results = {}
for name, filter in filters.items():
    print(f"Running filter: {name}")
    filter.initialize(num_particles=500)
    outputs = filter.run(observations)

    estimated_log_volatilities = [
        np.mean([p.state for p in output.particles]) for output in outputs
    ]
    estimated_mus = [
        np.mean([p.parameters[0] for p in output.particles]) for output in outputs
    ]
    estimated_phis = [
        np.mean([p.parameters[1] for p in output.particles]) for output in outputs
    ]
    estimated_sigmas = [
        np.mean([p.parameters[2] for p in output.particles]) for output in outputs
    ]

    results[name] = {
        "log_volatilities": estimated_log_volatilities,
        "mus": estimated_mus,
        "phis": estimated_phis,
        "sigmas": estimated_sigmas,
    }

import matplotlib.pyplot as plt

# Create axes for subplots
fig, axs = plt.subplots(3, 2, figsize=(14, 10))
axs = axs.flatten()

# Plot log-volatility
for name, res in results.items():
    axs[0].plot(
        res["log_volatilities"],
        label=f"Estimated Log-Volatility ({name})",
        alpha=0.7,
    )

    estimated_returns = [
        np.exp(res["log_volatilities"][t] / 2) * np.random.normal()
        for t in range(len(res["log_volatilities"]))
    ]
    axs[1].scatter(
        range(len(estimated_returns)),
        estimated_returns,
        label=f"Estimated Returns ({name})",
        s=10,
        alpha=0.5,
    )

    axs[2].plot(res["mus"], label=f"Estimated μ ({name})")
    axs[3].plot(res["phis"], label=f"Estimated φ ({name})")
    axs[4].plot(res["sigmas"], label=f"Estimated σ ({name})")

axs[0].plot(log_volatilities, label="True Log-Volatility", color="black")
axs[0].set_title("Log-Volatility")
axs[0].legend()
axs[1].scatter(range(len(returns)), returns, label="Observed Returns", color="green", s=10)
axs[1].set_title("Returns")
axs[1].legend()
axs[2].axhline(true_mu, color="black", linestyle="--", label="True μ")
axs[2].set_title("Parameter μ")
axs[2].legend()
axs[3].axhline(true_phi, color="black", linestyle="--", label="True φ")
axs[3].set_title("Parameter φ")
axs[3].legend()
axs[4].axhline(true_sigma, color="black", linestyle="--", label="True σ")
axs[4].set_title("Parameter σ")
axs[4].legend()


# plt.figure(figsize=(14, 6))
# plt.subplot(3, 2, 1)
# plt.plot(log_volatilities, label="True Log-Volatility", color="blue")
# plt.plot(
#     estimated_log_volatilities,
#     label="Estimated Log-Volatility",
#     color="orange",
#     alpha=0.7,
# )
# plt.legend()
# plt.title("Log-Volatility")
# plt.subplot(3, 2, 2)
# plt.scatter(range(len(returns)), returns, label="Observed Returns", color="green", s=10)
# # Compute estimated returns from estimated volatilities
# estimated_returns = [
#     estimated_betas[t] * np.exp(estimated_log_volatilities[t] / 2) * np.random.normal()
#     for t in range(len(estimated_log_volatilities))
# ]
# plt.scatter(
#     range(len(estimated_returns)),
#     estimated_returns,
#     label="Estimated Returns",
#     color="orange",
#     s=10,
#     alpha=0.5,
# )
# plt.legend()
# plt.title("Returns")
# plt.subplot(3, 2, 3)
# plt.plot(estimated_mus, label="Estimated μ", color="red")
# plt.axhline(true_mu, color="black", linestyle="--", label="True μ")
# plt.legend()
# plt.title("Parameter μ")
# plt.subplot(3, 2, 4)
# plt.plot(estimated_phis, label="Estimated φ", color="purple")
# plt.axhline(true_phi, color="black", linestyle="--", label="True φ")
# plt.legend()
# plt.title("Parameter φ")
# plt.subplot(3, 2, 5)
# plt.plot(estimated_sigmas, label="Estimated σ", color="brown")
# plt.axhline(true_sigma, color="black", linestyle="--", label="True σ")
# plt.legend()
# plt.title("Parameter σ")
# plt.subplot(3, 2, 6)
# plt.plot(estimated_betas, label="Estimated β", color="cyan")
# plt.axhline(true_beta, color="black", linestyle="--", label="True β")
# plt.legend()
# plt.title("Parameter β")
# plt.tight_layout()
# plt.show()
