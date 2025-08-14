"""
Adaptive treatment assignment based on running estimates.
"""

import numpy as np
from typing import Optional, Any
from treatment_effect_gym import ExperimentDesign


class AdaptiveDesign(ExperimentDesign):
    """Adaptive treatment assignment based on running estimates."""
    
    def __init__(self, 
                 exploration_rate: float = 0.1,
                 window_size: int = 100,
                 initial_p: float = 0.5):
        """
        Args:
            exploration_rate: Probability of random assignment
            window_size: Window for computing running estimates
            initial_p: Initial treatment probability
        """
        self.exploration_rate = exploration_rate
        self.window_size = window_size
        self.initial_p = initial_p
        
        self._history = []  # Store (action, reward) pairs
        self._seed = None
    
    def reset(self, seed: Optional[int] = None) -> None:
        """Reset design state."""
        if seed is not None:
            self._seed = seed
            np.random.seed(seed)
        self._history = []
    
    def assign_treatment(self, t: int, state: Any, reward: Optional[float] = None, **kwargs) -> int:
        """Assign treatment adaptively based on history."""
        # Store previous reward if provided
        if reward is not None and len(self._history) > 0:
            self._history[-1] = (self._history[-1][0], reward)
        
        # Exploration phase or random exploration
        if t < self.window_size or np.random.random() < self.exploration_rate:
            action = int(np.random.random() < self.initial_p)
        else:
            # Compute treatment effect estimate from recent history
            recent_history = self._history[-self.window_size:]
            
            treat_rewards = [r for a, r in recent_history if a == 1 and r is not None]
            control_rewards = [r for a, r in recent_history if a == 0 and r is not None]
            
            if len(treat_rewards) > 0 and len(control_rewards) > 0:
                treat_mean = np.mean(treat_rewards)
                control_mean = np.mean(control_rewards)
                
                # Assign treatment if it appears better
                action = int(treat_mean > control_mean)
            else:
                # Fall back to random if insufficient data
                action = int(np.random.random() < self.initial_p)
        
        # Store action (reward will be filled in later)
        self._history.append((action, None))
        
        return action
    
    @property
    def name(self) -> str:
        return f"Adaptive(eps={self.exploration_rate},win={self.window_size})"