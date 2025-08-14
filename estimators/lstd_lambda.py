"""
LSTD-λ based treatment effect estimator.
"""

import numpy as np
from scipy.linalg import solve
from treatment_effect_gym import TreatmentEstimator


class LSTDLambdaEstimator(TreatmentEstimator):
    """LSTD-λ based treatment effect estimator."""
    
    def __init__(self, 
                 discount_factor: float = 1.0, 
                 alpha: float = 1.0, 
                 lambda_: float = 0.0):
        """
        Args:
            discount_factor: Discount factor
            alpha: Regularization parameter
            lambda_: Eligibility trace decay parameter
        """
        self.discount_factor = discount_factor
        self.alpha = alpha
        self.lambda_ = lambda_
    
    def estimate(self, rewards: np.ndarray, actions: np.ndarray, states: np.ndarray, **kwargs) -> float:
        """OPE Estimator (matching notebook implementation)."""
        if states.ndim == 1:
            states = states.reshape(-1, 1)
        
        if len(states) < 2:
            return 0.0
        
        ss = states[:-1]  # Current states
        snews = states[1:]  # Next states
        rs = rewards[:-1]  # Rewards
        actions_prev = actions[:-1]  # Actions taken
        
        # Split by treatment assignment
        treat_mask = actions_prev == 1
        control_mask = actions_prev == 0
        
        if np.sum(treat_mask) == 0 or np.sum(control_mask) == 0:
            return 0.0
        
        # Treatment group data
        ss1 = ss[treat_mask]
        snews1 = snews[treat_mask]
        rs1 = rs[treat_mask]
        
        # Control group data
        ss0 = ss[control_mask]
        snews0 = snews[control_mask]
        rs0 = rs[control_mask]
        
        try:
            # OPE for treatment group (eta1)
            A1 = ss1.T @ (ss1 - snews1)
            b1 = ss1.T @ rs1
            eta1 = solve(A1 + 1e-3 * np.eye(A1.shape[0]), b1)[0]
            
            # OPE for control group (eta0)  
            A0 = ss0.T @ (ss0 - snews0)
            b0 = ss0.T @ rs0
            eta0 = solve(A0 + 1e-3 * np.eye(A0.shape[0]), b0)[0]
            
            # Treatment effect is difference
            estimate = eta1 - eta0
            return float(estimate)
            
        except (np.linalg.LinAlgError, IndexError):
            return 0.0
    
    @property
    def name(self) -> str:
        return f"LSTD-λ(γ={self.discount_factor},α={self.alpha},λ={self.lambda_})"