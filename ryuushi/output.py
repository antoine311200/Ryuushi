from dataclasses import dataclass
from typing import Any, Dict, List

from ryuushi.particle import Particle

@dataclass
class Output:
    """Container for SMC algorithm output"""
    particles: List[Particle] # List of particles at the final time step
    log_likelihood: float # Log-likelihood estimate
    effective_sample_size: float # Effective sample size at the final time step for resampling
    diagnostics: Dict[str, Any] # Additional diagnostics information