from typing import Any, Optional

import numpy as np

from ryuushi.models import StateSpaceModel
from ryuushi.filters import SIRFilter, SISFilter
from ryuushi.resampling import ResamplingScheme
from ryuushi.observation import Observation


class PendulumTrackingModel(StateSpaceModel):

    def __init__(
        self,
        process_noise: float = 0.1,
        measurement_noise: float = 0.5,
        damping: float = 0.05,
        gravity: float = 9.81,
        length: float = 1.0,
        method: str = "Euler"
    ):
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.damping = damping
        self.gravity = gravity
        self.length = length
        self.method = method
        self.prev_time = 0

    def pendulum_dynamics(self, state: np.ndarray, dt: float = 0.1) -> np.ndarray:

        def f(state):
            # Pendulum equation: d²θ/dt² + damping*dθ/dt + (g/L)*sin(θ) = 0
            angle, angular_vel = state
            angular_accel = (
                -self.damping * angular_vel
                - (self.gravity / self.length) * np.sin(angle)
            )
            return np.array([angular_vel, angular_accel])

        if self.method == "Euler":
            # Euler integration
            angle, angular_vel = state
            angular_vel, angular_accel = f(state)
            new_angle = angle + angular_vel * dt
            new_angular_vel = angular_vel + angular_accel * dt
            new_state = np.array([new_angle, new_angular_vel])
        elif self.method == "RK4":
            # Runge-Kutta 4th order integration
            k1 = f(state)
            k2 = f(state + 0.5 * dt * k1)
            k3 = f(state + 0.5 * dt * k2)
            k4 = f(state + dt * k3)
            new_state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        else:
            raise ValueError("Unknown integration method")

        new_state[0] = (new_state[0] + np.pi) % (2 * np.pi) - np.pi  # Wrap angle to [-π, π]
        return new_state

    def transition(self, state: np.ndarray, time: int, dt: float, parameters: Optional[np.ndarray] = None) -> np.ndarray: # type: ignore
        deterministic_update = self.pendulum_dynamics(state, dt=dt)
        noise = np.random.normal(0, self.process_noise, size=state.shape)
        return deterministic_update + noise

    def log_likelihood(self, state: np.ndarray, observation: Observation, time: int, parameters: Optional[np.ndarray] = None) -> float:
        # Compute p(y_t | x_t) assuming Gaussian noise
        if observation is None: return 0.0
        if self.measurement_noise == 0:
            return 0.0
        observed_angle = state[0]
        angle_diff = observation.data - observed_angle

        return (
            -0.5 * np.log(2 * np.pi * self.measurement_noise**2)
            - (angle_diff**2) / (2 * self.measurement_noise**2)
        )

    def initial_state(self) -> np.ndarray:
        return np.array([np.pi / 4, 0.1])  # [angle, angular_velocity]

    def initial_parameters(self) -> np.ndarray:
        return np.array([self.process_noise, self.measurement_noise, self.damping])

model = PendulumTrackingModel(
    process_noise=0.05,
    measurement_noise=0.3,
    damping=0.5,
    method="RK4"
)

def generate_pendulum_data(model: PendulumTrackingModel, time_steps: np.ndarray):
    """Generate synthetic pendulum data"""
    true_states = []
    observations = []
    current_state = model.initial_state()

    prev_t = 0.0
    for t in time_steps:
        dt = t - prev_t
        current_state = model.transition(current_state, t, dt)
        true_states.append(current_state.copy())

        true_angle = current_state[0]
        observation = true_angle + np.random.normal(0, model.measurement_noise)
        observations.append(Observation(data=observation, time=t))

        prev_t = t

    return true_states, observations

n_time_steps = 300
final_time = 10.0
# time_steps = np.sort(np.random.uniform(0, final_time, size=n_time_steps))
time_steps = np.linspace(0, final_time, n_time_steps)

true_states, observations = generate_pendulum_data(model, time_steps)
true_angles = [state[0] for state in true_states]
true_velocities = [state[1] for state in true_states]

filters = {
    "SIR-SYSTEMATIC": SIRFilter(model, ResamplingScheme.SYSTEMATIC),
    "SIR-MULTINOMIAL": SIRFilter(model, ResamplingScheme.MULTINOMIAL),
    # "SIS": SISFilter(model)
}

results = {}
n_particles = 250
for name, filter in filters.items():
    print(f"Running filter: {name}")
    filter.initialize(n_particles)
    outputs = filter.run(observations)

    estimated_angles = [np.mean([p.state[0] for p in output.particles]) for output in outputs]
    estimated_velocities = [np.mean([p.state[1] for p in output.particles]) for output in outputs]

    angle_errors = [abs(estimated - true) for estimated, true in zip(estimated_angles, true_angles)]
    velocity_errors = [abs(estimated - true) for estimated, true in zip(estimated_velocities, true_velocities)]
    angle_rmse = np.sqrt(np.mean([err**2 for err in angle_errors]))
    velocity_rmse = np.sqrt(np.mean([err**2 for err in velocity_errors]))

    results[name] = {
        "estimated_angles": estimated_angles,
        "estimated_velocities": estimated_velocities,
        "angle_rmse": angle_rmse,
        "velocity_rmse": velocity_rmse
    }
    print(f"{name} - Angle RMSE: {angle_rmse:.4f}, Velocity RMSE: {velocity_rmse:.4f}")

# Plotting
import matplotlib.pyplot as plt

observations = [obs.data for obs in observations]

plt.figure(figsize=(14, 8))
plt.subplot(2, 1, 1)
plt.plot(time_steps, true_angles, label="True Angle", color='black')
plt.scatter(time_steps, observations, label="Observations", color='red', s=10, alpha=0.5)
for name, result in results.items():
    plt.plot(time_steps, result["estimated_angles"], label=f"Estimated Angle ({name})")
plt.xlabel("Time")
plt.ylabel("Angle (radians)")
plt.title("Pendulum Angle Tracking")
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(time_steps, true_velocities, label="True Angular Velocity", color='black')
for name, result in results.items():
    plt.plot(time_steps, result["estimated_velocities"], label=f"Estimated Velocity ({name})")
plt.xlabel("Time")
plt.ylabel("Angular Velocity (radians/s)")
plt.title("Pendulum Angular Velocity Tracking")
plt.legend()
plt.tight_layout()
plt.show()