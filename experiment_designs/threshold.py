"""
Treatment assignment based on state threshold.
"""

import numpy as np
from typing import Optional, Any
from treatment_effect_gym import ExperimentDesign


class ThresholdDesign(ExperimentDesign):
    """Treatment assignment based on state threshold."""
    
    def __init__(self, threshold: float = 0.5, feature_index: int = 0):
        """
        Args:
            threshold: Threshold value for treatment assignment
            feature_index: Which feature to use for thresholding
        """
        self.threshold = threshold
        self.feature_index = feature_index
        self._seed = None
    
    def reset(self, seed: Optional[int] = None) -> None:
        """Reset design state."""
        if seed is not None:
            self._seed = seed
            np.random.seed(seed)
    
    def assign_treatment(self, t: int, state: Any, **kwargs) -> int:
        """Assign treatment based on state threshold."""
        if isinstance(state, (int, float)):
            value = state
        elif hasattr(state, '__len__') and len(state) > self.feature_index:
            value = state[self.feature_index]
        else:
            value = 0
        
        return int(value > self.threshold)
    
    @property
    def name(self) -> str:
        return f"Threshold(th={self.threshold},feat={self.feature_index})"