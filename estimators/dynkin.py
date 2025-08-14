"""
Dynkin-based value difference estimator using LSTD.
"""

import numpy as np
from scipy.linalg import solve
from treatment_effect_gym import TreatmentEstimator


class DynkinEstimator(TreatmentEstimator):
    """Dynkin-based value difference estimator using LSTD."""
    
    def __init__(self, discount_factor: float = 1.0, alpha: float = 1.0):
        """
        Args:
            discount_factor: Discount factor
            alpha: Regularization parameter
        """
        self.discount_factor = discount_factor
        self.alpha = alpha
    
    def estimate(self, rewards: np.ndarray, actions: np.ndarray, states: np.ndarray, **kwargs) -> float:
        """Estimate using Dynkin method."""
        # Ensure states are 2D
        if states.ndim == 1:
            states = states.reshape(-1, 1)
        
        if len(states) < 2:
            return 0.0
        
        ss = states[:-1]  # Current states
        snews = states[1:]  # Next states
        rs = rewards[:-1]  # Rewards
        actions_prev = actions[:-1]  # Actions taken
        
        # LSTD-DQ equations (matching notebook implementation)
        A = ss.T @ (ss - snews)  # No discount factor in notebook
        rbar = np.mean(rs)
        b = ss.T @ (rs - rbar)
        
        # Split by treatment assignment for state averages
        treat_mask = actions_prev == 1
        control_mask = actions_prev == 0
        
        if np.sum(treat_mask) == 0 or np.sum(control_mask) == 0:
            return 0.0
        
        # Use current states ss (not next states snews) as in notebook
        ss_treated = ss[treat_mask]  # Current states when treated
        ss_control = ss[control_mask]  # Current states when control
        
        # Compute state difference
        delta_xbar = np.mean(ss_treated, axis=0) - np.mean(ss_control, axis=0)
        
        # Regularization (use 1e-3 as in notebook)
        reg_matrix = 1e-3 * np.eye(A.shape[0])
        
        try:
            theta_dq = solve(A + reg_matrix, b)
            estimate = theta_dq @ delta_xbar
            return float(estimate)
        except np.linalg.LinAlgError:
            return 0.0
    
    @property
    def name(self) -> str:
        return f"Dynkin(γ={self.discount_factor},α={self.alpha})"