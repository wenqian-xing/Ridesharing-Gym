"""
Truncated Difference-in-Q's estimator.
"""

import numpy as np
from treatment_effect_gym import TreatmentEstimator


class TruncatedDQEstimator(TreatmentEstimator):
    """Truncated Difference-in-Q's estimator."""
    
    def __init__(self, k: int = 1, discount_factor: float = 1.0):
        """
        Args:
            k: Truncation parameter (horizon length)
            discount_factor: Discount factor for future rewards
        """
        self.k = k
        self.discount_factor = discount_factor
    
    def _compute_truncated_q_values(self, rewards: np.ndarray, k: int, discount_factor: float) -> np.ndarray:
        """Compute k-truncated Q-values."""
        T = len(rewards)
        q_values = np.zeros(T)
        
        for i in range(T):
            end = min(i + k + 1, T)
            future_rewards = rewards[i:end]
            discount_weights = discount_factor ** np.arange(len(future_rewards))
            q_values[i] = np.sum(future_rewards * discount_weights)
        
        return q_values
    
    def estimate(self, rewards: np.ndarray, actions: np.ndarray, states: np.ndarray, **kwargs) -> float:
        """Estimate treatment effect using truncated DQ."""
        # Compute truncated Q-values
        q_values = self._compute_truncated_q_values(rewards, self.k, self.discount_factor)
        
        # Apply discounting
        discount_weights = self.discount_factor ** np.arange(len(q_values))
        discounted_q_values = q_values * discount_weights
        
        # Compute difference
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
        return f"TruncatedDQ(k={self.k},γ={self.discount_factor})"