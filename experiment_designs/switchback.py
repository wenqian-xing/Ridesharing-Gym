"""
Switchback experimental design with time-based treatment switching.
"""

import numpy as np
from typing import Optional, Any
from treatment_effect_gym import ExperimentDesign


class SwitchbackDesign(ExperimentDesign):
    """Switchback experimental design with time-based treatment switching."""
    
    def __init__(self, switch_every: int = 600, p: float = 0.5):
        """
        Args:
            switch_every: Duration of each treatment period
            p: Probability of treatment in each period
        """
        self.switch_every = switch_every
        self.p = p
        self._treatment_schedule = None
        self._seed = None
    
    def reset(self, seed: Optional[int] = None) -> None:
        """Reset design state."""
        if seed is not None:
            self._seed = seed
            np.random.seed(seed)
        self._treatment_schedule = None
    
    def _generate_schedule(self, max_time: int) -> np.ndarray:
        """Generate treatment schedule for switchback periods."""
        if self._treatment_schedule is not None:
            return self._treatment_schedule
        
        # Determine number of periods needed
        n_periods = (max_time // self.switch_every) + 1
        
        # Randomly assign treatment to each period
        period_treatments = np.random.binomial(1, self.p, n_periods)
        
        # Expand to time steps
        schedule = np.repeat(period_treatments, self.switch_every)
        self._treatment_schedule = schedule
        
        return schedule
    
    def assign_treatment(self, t: int, state: Any, **kwargs) -> int:
        """Assign treatment based on switchback schedule.
        
        Args:
            t: Step index (for backward compatibility)
            state: Current state (may contain simulation time)
            **kwargs: Additional parameters including 'simulation_time' for rideshare
        """
        # Use simulation time if provided (for rideshare environment)
        simulation_time = kwargs.get('simulation_time', None)
        
        if simulation_time is not None:
            # Use actual simulation time for switching decisions
            period_id = int(simulation_time // self.switch_every)
            
            # Generate random assignment for this period (deterministic given seed)
            if self._seed is not None:
                np.random.seed(self._seed + period_id)  # Deterministic per period
            
            return int(np.random.binomial(1, self.p))
        else:
            # Fallback to step-based switching for other environments
            max_time = kwargs.get('max_time', t + 10000)
            schedule = self._generate_schedule(max_time)
            
            if t < len(schedule):
                return int(schedule[t])
            else:
                return int(schedule[-1])
    
    @property
    def name(self) -> str:
        return f"Switchback(every={self.switch_every},p={self.p})"