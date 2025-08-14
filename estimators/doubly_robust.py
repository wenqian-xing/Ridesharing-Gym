"""
Doubly Robust estimator (simplified version).
"""

import numpy as np
from treatment_effect_gym import TreatmentEstimator


class DoublyRobustEstimator(TreatmentEstimator):
    """Doubly Robust estimator (simplified version)."""
    
    def __init__(self, propensity_score: float = 0.5):
        """
        Args:
            propensity_score: Known propensity score
        """
        self.propensity_score = propensity_score
    
    def estimate(self, rewards: np.ndarray, actions: np.ndarray, states: np.ndarray, **kwargs) -> float:
        """Estimate using doubly robust method."""
        # Simple outcome regression (linear)
        if states.ndim == 1:
            states = states.reshape(-1, 1)
        
        # Fit outcome models (very simplified)
        treat_mask = actions == 1
        control_mask = actions == 0
        
        if np.sum(treat_mask) == 0 or np.sum(control_mask) == 0:
            return 0.0
        
        # Simple means as outcome predictions
        mu1 = np.mean(rewards[treat_mask])
        mu0 = np.mean(rewards[control_mask])
        
        # DR estimate
        n = len(rewards)
        dr_treat = np.mean(
            (actions * rewards) / self.propensity_score +
            ((1 - actions) / self.propensity_score) * mu1
        )
        dr_control = np.mean(
            ((1 - actions) * rewards) / (1 - self.propensity_score) +
            (actions / (1 - self.propensity_score)) * mu0
        )
        
        return dr_treat - dr_control
    
    @property
    def name(self) -> str:
        return f"DoublyRobust(π={self.propensity_score})"