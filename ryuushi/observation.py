from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class Observation:
    """Container for observations"""
    data: Any  # Observation data
    time: int  # Time index