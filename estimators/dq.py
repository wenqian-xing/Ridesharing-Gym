"""
Difference-in-Q's estimator.
"""

import numpy as np
from treatment_effect_gym import TreatmentEstimator


class DQEstimator(TreatmentEstimator):
    """Difference-in-Q's estimator."""
    
    def __init__(self, discount_factor: float = 1.0):
        """
        Args:
            discount_factor: Discount factor for future rewards
        """
        self.discount_factor = discount_factor
    
    def _compute_q_values(self, rewards: np.ndarray, discount_factor: float) -> np.ndarray:
        """Compute Q-values (value-to-go) for each state."""
        T = len(rewards)
        q_values = np.zeros(T)
        
        for i in range(T):
            future_rewards = rewards[i:]
            discount_weights = discount_factor ** np.arange(len(future_rewards))
            q_values[i] = np.sum(future_rewards * discount_weights)
        
        return q_values
    
    def estimate(self, rewards: np.ndarray, actions: np.ndarray, states: np.ndarray, **kwargs) -> float:
        """Estimate treatment effect using DQ."""
        # Compute Q-values
        q_values = self._compute_q_values(rewards, self.discount_factor)
        
        # Apply discounting to Q-values
        discount_weights = self.discount_factor ** np.arange(len(q_values))
        discounted_q_values = q_values * discount_weights
        
        # Compute difference in Q's
        treatment_mask = actions == 1
        control_mask = actions == 0
        
        if np.sum(treatment_mask) == 0 or np.sum(control_mask) == 0:
            return 0.0
        
        treatment_q = np.sum(discounted_q_values[treatment_mask])
        control_q = np.sum(discounted_q_values[control_mask])
        
        # Normalize
        treatment_mean = treatment_q / np.sum(treatment_mask)
        control_mean = control_q / np.sum(control_mask)
        
        return treatment_mean - control_mean
    
    @property
    def name(self) -> str:
        return f"DQ(γ={self.discount_factor})"