import numpy as np
import matplotlib.pyplot as plt

from ryuushi.filters import SISFilter, SIRFilter
from ryuushi.models import RandomWalkSSM

model = RandomWalkSSM(process_noise=1.0, measurement_noise=0.5)

# Initialize filters
sis_filter = SISFilter(model)
sir_filter = SIRFilter(model)

# Initialize particles
sis_filter.initialize(num_particles=100)
sir_filter.initialize(num_particles=100)

# Generate synthetic data for observations
time_steps = 50
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

# Plot results
plt.figure(figsize=(12, 6))
plt.plot([obs[0] for obs in observations], 'k.', label='Observations')
plt.plot([mean[0] for mean in sis_means], 'b-', label='SIS Estimate')
plt.plot([mean[0] for mean in sir_means], 'r-', label='SIR Estimate')
plt.plot([state[0] for state in true_states], 'g--', label='True State')
# Add color line when resampling occurs in SIR
# resample_times = [i for i, output in enumerate(sir_outputs) if output.diagnostics["resampled"]]
# for rt in resample_times:
#     plt.axvline(x=rt, color='r', linestyle='--', alpha=0.5)
plt.legend()
plt.title('SIS vs SIR Filter Estimates')
plt.xlabel('Time')
plt.ylabel('State Estimate')
plt.show()