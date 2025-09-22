from typing import Any

import numpy as np

from ryuushi.models.ssm import StateSpaceModel

class RandomWalkSSM(StateSpaceModel):
    """Example state space model for demonstration"""
    def __init__(self, process_noise: float = 0.1, measurement_noise: float = 0.1):
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

    def transition(self, state: np.ndarray, time: int) -> np.ndarray:
        return state + np.random.normal(0, self.process_noise, size=state.shape)

    def log_likelihood(self, state: np.ndarray, observation: Any, time: int) -> float:
        if observation is None:
            return 0.0
        return -0.5 * np.sum((observation - state) ** 2) / self.measurement_noise ** 2

    def initial_state(self) -> np.ndarray:
        return np.array([0.0])
