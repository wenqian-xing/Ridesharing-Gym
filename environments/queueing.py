"""
Non-stationary queueing system environment.
"""

import numpy as np
from typing import Tuple, Optional, Any
from treatment_effect_gym import TreatmentEnvironment, ExperimentDesign


class QueueingEnvironment(TreatmentEnvironment):
    """Non-stationary queueing system environment."""
    
    def __init__(self, 
                 data_file: str = "data0.csv",
                 service_rate_control: float = 1.0,
                 service_rate_treatment: float = 1.2):
        """
        Args:
            data_file: Path to patient arrival data
            service_rate_control: Service rate for control
            service_rate_treatment: Service rate for treatment
        """
        self.data_file = data_file
        self.service_rate_control = service_rate_control
        self.service_rate_treatment = service_rate_treatment
        
        # Load arrival data if available
        try:
            import pandas as pd
            self.arrival_data = pd.read_csv(data_file, header=None)
            self.arrival_rates = self.arrival_data.iloc[:, 1].values  # Second column
        except:
            # Generate synthetic arrival data if file not found
            self.arrival_rates = self._generate_synthetic_arrivals()
        
        self._seed = None
    
    def _generate_synthetic_arrivals(self, n_hours: int = 168) -> np.ndarray:
        """Generate synthetic arrival rates (weekly cycle)."""
        # Simple sinusoidal pattern with day/night variation
        hours = np.arange(n_hours)
        base_rate = 5.0
        daily_variation = 3.0 * np.sin(2 * np.pi * hours / 24)
        weekly_variation = 1.0 * np.sin(2 * np.pi * hours / (24 * 7))
        noise = np.random.normal(0, 0.5, n_hours)
        
        arrival_rates = base_rate + daily_variation + weekly_variation + noise
        return np.maximum(arrival_rates, 0.1)  # Ensure positive
    
    def reset(self, seed: Optional[int] = None) -> None:
        """Reset environment state."""
        if seed is not None:
            self._seed = seed
            np.random.seed(seed)
    
    def _simulate_queue(self, policy: str, T: int, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Simulate queueing system."""
        if seed is not None:
            np.random.seed(seed)
        
        rewards, actions, states = [], [], []
        queue_length = 0
        
        # Map T steps to arrival rate pattern
        rate_indices = np.arange(T) % len(self.arrival_rates)
        
        for t in range(T):
            # Current state (queue length)
            states.append(queue_length)
            
            # Choose action (service rate)
            if policy == "treatment":
                a_t = 1
                service_rate = self.service_rate_treatment
            elif policy == "control":
                a_t = 0
                service_rate = self.service_rate_control
            elif policy == "random":
                a_t = np.random.choice([0, 1])
                service_rate = self.service_rate_treatment if a_t else self.service_rate_control
            else:
                raise ValueError(f"Unknown policy: {policy}")
            
            actions.append(a_t)
            
            # Simulate arrivals and departures
            arrival_rate = self.arrival_rates[rate_indices[t]]
            arrivals = np.random.poisson(arrival_rate)
            
            if queue_length > 0:
                departures = np.random.poisson(service_rate)
                departures = min(departures, queue_length)
            else:
                departures = 0
            
            # Update queue
            queue_length = max(0, queue_length + arrivals - departures)
            
            # Reward: negative of queue length (minimize waiting)
            reward = -queue_length
            rewards.append(reward)
        
        return np.array(rewards), np.array(actions), np.array(states)
    
    def simulate_treatment_policy(self, T: int, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Simulate pure treatment policy."""
        return self._simulate_queue("treatment", T, seed)
    
    def simulate_control_policy(self, T: int, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Simulate pure control policy."""
        return self._simulate_queue("control", T, seed)
    
    def simulate_experiment(self, design: ExperimentDesign, T: int, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Simulate experiment according to design."""
        if seed is not None:
            np.random.seed(seed)
        
        design.reset(seed)
        
        rewards, actions, states = [], [], []
        queue_length = 0
        rate_indices = np.arange(T) % len(self.arrival_rates)
        
        for t in range(T):
            states.append(queue_length)
            
            # Get action from design
            a_t = design.assign_treatment(t, queue_length)
            actions.append(a_t)
            
            service_rate = self.service_rate_treatment if a_t else self.service_rate_control
            
            # Simulate queue dynamics
            arrival_rate = self.arrival_rates[rate_indices[t]]
            arrivals = np.random.poisson(arrival_rate)
            
            if queue_length > 0:
                departures = np.random.poisson(service_rate)
                departures = min(departures, queue_length)
            else:
                departures = 0
            
            queue_length = max(0, queue_length + arrivals - departures)
            reward = -queue_length
            rewards.append(reward)
        
        return np.array(rewards), np.array(actions), np.array(states)
    
    def compute_true_ate(self, T: int, seed: Optional[int] = None) -> float:
        """Compute true average treatment effect."""
        r_treat, _, _ = self.simulate_treatment_policy(T, seed)
        r_control, _, _ = self.simulate_control_policy(T, seed + 1000 if seed else None)
        return (np.sum(r_treat) - np.sum(r_control)) / T
    
    @property
    def name(self) -> str:
        return "Queueing"