import numpy as np
import matplotlib.pyplot as plt

from ryuushi.filters import SISFilter, SIRFilter
from ryuushi.models import RandomWalkSSM

model = RandomWalkSSM(process_noise=2.0, measurement_noise=2.5)

# Initialize filters
sis_filter = SISFilter(model)
sir_filter = SIRFilter(model)

# Initialize particles
sis_filter.initialize(num_particles=100)
sir_filter.initialize(num_particles=100)

# Generate synthetic data for observations
time_steps = 100
random_times = np.random.rand(time_steps)
random_times.sort()

true_states = [model.transition(model.initial_state(), t) for t in random_times]
observations = []
for t in random_times:
    true_state = model.transition(model.initial_state(), t)
    observation = true_state + np.random.normal(0, model.measurement_noise, size=true_state.shape)
    observations.append(observation)

# Run filters on the observations
sis_outputs = sis_filter.run(observations)
sir_outputs = sir_filter.run(observations)

# Extract means for plotting
sis_means = [np.mean([p.state for p in output.particles], axis=0) for output in sis_outputs]
sir_means = [np.mean([p.state for p in output.particles], axis=0) for output in sir_outputs]

# RMSE
sis_rmse = np.sqrt(np.mean([(sis_mean - true_state) ** 2 for sis_mean, true_state in zip(sis_means, true_states)]))
sir_rmse = np.sqrt(np.mean([(sir_mean - true_state) ** 2 for sir_mean, true_state in zip(sir_means, true_states)]))

print(f"SIS RMSE: {sis_rmse}")
print(f"SIR RMSE: {sir_rmse}")

# Plot results
plt.figure(figsize=(12, 6))
plt.plot([obs[0] for obs in observations], 'k.', label='Observations')
plt.plot([mean[0] for mean in sis_means], 'b-', label='SIS Estimate')
plt.plot([mean[0] for mean in sir_means], 'r-', label='SIR Estimate')
plt.plot([state[0] for state in true_states], 'g--', label='True State')
plt.legend()
plt.title('SIS vs SIR Filter Estimates')
plt.xlabel('Time')
plt.ylabel('State Estimate')
plt.show()