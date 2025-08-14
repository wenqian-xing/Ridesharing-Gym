"""
Naive A/B testing with random treatment assignment.
"""

import numpy as np
from typing import Optional, Any
from treatment_effect_gym import ExperimentDesign


class RandomizedDesign(ExperimentDesign):
    """Naive A/B testing with random treatment assignment."""
    
    def __init__(self, p: float = 0.5):
        """
        Args:
            p: Probability of treatment assignment
        """
        self.p = p
        self._seed = None
    
    def reset(self, seed: Optional[int] = None) -> None:
        """Reset design state."""
        if seed is not None:
            self._seed = seed
            np.random.seed(seed)
    
    def assign_treatment(self, t: int, state: Any, **kwargs) -> int:
        """Assign treatment randomly."""
        return int(np.random.random() < self.p)
    
    @property
    def name(self) -> str:
        return f"RandomizedAB(p={self.p})"