"""
Inverse Propensity Weighting estimator.
"""

import numpy as np
from treatment_effect_gym import TreatmentEstimator


class IPWEstimator(TreatmentEstimator):
    """Inverse Propensity Weighting estimator."""
    
    def __init__(self, propensity_score: float = 0.5):
        """
        Args:
            propensity_score: Known propensity score (for randomized experiments)
        """
        self.propensity_score = propensity_score
    
    def estimate(self, rewards: np.ndarray, actions: np.ndarray, states: np.ndarray, **kwargs) -> float:
        """Estimate using IPW."""
        # IPW weights
        weights = np.where(
            actions == 1,
            1.0 / self.propensity_score,
            1.0 / (1.0 - self.propensity_score)
        )
        
        # Weighted outcomes
        weighted_outcomes = rewards * weights
        
        treatment_mask = actions == 1
        control_mask = actions == 0
        
        if np.sum(treatment_mask) == 0 or np.sum(control_mask) == 0:
            return 0.0
        
        treatment_mean = np.mean(weighted_outcomes[treatment_mask])
        control_mean = np.mean(weighted_outcomes[control_mask])
        
        return treatment_mean - control_mean
    
    @property
    def name(self) -> str:
        return f"IPW(π={self.propensity_score})"