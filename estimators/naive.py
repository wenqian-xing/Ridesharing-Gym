"""
Naive difference-in-means estimator (Direct Method).
"""

import numpy as np
from treatment_effect_gym import TreatmentEstimator


class NaiveEstimator(TreatmentEstimator):
    """Naive difference-in-means estimator (Direct Method)."""
    
    def __init__(self, discount_factor: float = 1.0):
        """
        Args:
            discount_factor: Discount factor for future rewards
        """
        self.discount_factor = discount_factor
    
    def estimate(self, rewards: np.ndarray, actions: np.ndarray, states: np.ndarray, **kwargs) -> float:
        """Estimate treatment effect using naive DM."""
        # Apply discounting
        discount_weights = self.discount_factor ** np.arange(len(rewards))
        discounted_rewards = rewards * discount_weights
        
        # Compute difference in means
        treatment_mask = actions == 1
        control_mask = actions == 0
        
        if np.sum(treatment_mask) == 0 or np.sum(control_mask) == 0:
            return 0.0
        
        treatment_return = np.sum(discounted_rewards[treatment_mask])
        control_return = np.sum(discounted_rewards[control_mask])
        
        # Normalize by number of observations
        treatment_mean = treatment_return / np.sum(treatment_mask)
        control_mean = control_return / np.sum(control_mask)
        
        return treatment_mean - control_mean
    
    @property
    def name(self) -> str:
        return f"NaiveDM(γ={self.discount_factor})"