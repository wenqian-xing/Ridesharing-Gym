"""
Two-state non-stationary MDP environment.
"""

import numpy as np
from typing import Tuple, Optional, Any
from treatment_effect_gym import TreatmentEnvironment, ExperimentDesign


class TwoStateMDPEnvironment(TreatmentEnvironment):
    """Two-state non-stationary MDP environment."""
    
    def __init__(self, 
                 mixing_coeff: float = 0.5,
                 treatment_bias: float = 0.1,
                 smoothness: float = 0.5,
                 noise_std: float = 0.02,
                 reward_std: float = 0.1):
        """
        Args:
            mixing_coeff: Total variation distance between transition rows
            treatment_bias: Treatment effect on transition probabilities
            smoothness: AR(1) mean reversion parameter
            noise_std: Gaussian noise in transitions
            reward_std: Standard deviation of reward noise
        """
        self.mixing_coeff = mixing_coeff
        self.treatment_bias = treatment_bias
        self.smoothness = smoothness
        self.noise_std = noise_std
        self.reward_std = reward_std
        
        # Reward matrix: state x action
        self.reward_matrix = np.array([[0, 1], [5, 6]])
        
        self._seed = None
        self._transition_kernels = None
    
    def reset(self, seed: Optional[int] = None) -> None:
        """Reset environment state."""
        if seed is not None:
            self._seed = seed
            np.random.seed(seed)
    
    def _generate_transition_kernels(self, T: int) -> list:
        """Generate time-varying transition kernels."""
        # Always regenerate to ensure reproducibility with seed
        kernels = self._generate_mean_reverting_kernels(T)
        return kernels
    
    def _generate_mean_reverting_kernels(self, T: int) -> list:
        """Generate non-stationary 2-state, 2-action kernels with AR(1) mean reversion."""
        kernels = []
        
        def construct_random_rows_with_exact_tv(mixing_coeff):
            """Construct two valid 2D probability vectors with exact total variation distance."""
            assert 0 < mixing_coeff < 1, "mixing_coeff must be in (0, 1)"
            delta = mixing_coeff / 2
            
            if np.random.rand() < 0.5:
                row0 = np.array([0.5 + delta, 0.5 - delta])
                row1 = np.array([0.5 - delta, 0.5 + delta])
            else:
                row0 = np.array([0.5 - delta, 0.5 + delta])
                row1 = np.array([0.5 + delta, 0.5 - delta])
            
            return row0, row1
        
        row0_mean, row1_mean = construct_random_rows_with_exact_tv(self.mixing_coeff)
        
        # Initialize current rows
        row_0 = row0_mean.copy()
        row_1 = row1_mean.copy()
        
        for t in range(T):
            kernel = np.zeros((2, 2, 2))
            
            # AR(1) with mean reversion
            row_0 = (self.smoothness * row_0 + 
                    (1 - self.smoothness) * row0_mean + 
                    np.random.normal(0, self.noise_std, 2))
            row_1 = (self.smoothness * row_1 + 
                    (1 - self.smoothness) * row1_mean + 
                    np.random.normal(0, self.noise_std, 2))
            
            # Clip and normalize
            for row in [row_0, row_1]:
                row[:] = np.clip(row, 0.01, 0.99)
                row[:] /= row.sum()
            
            # Action 0 (control)
            kernel[0, 0] = row_0.copy()
            kernel[1, 0] = row_1.copy()
            
            # Action 1 (treatment): shift toward state 1
            for s in [0, 1]:
                biased_row = kernel[s, 0] + np.array([-self.treatment_bias, self.treatment_bias])
                biased_row = np.clip(biased_row, 0.01, 0.99)
                kernel[s, 1] = biased_row / biased_row.sum()
            
            kernels.append(kernel)
        
        return kernels
    
    def _simulate_mdp(self, policy: str, T: int, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Simulate MDP with given policy."""
        if seed is not None:
            np.random.seed(seed)
        
        transition_kernels = self._generate_transition_kernels(T)
        exo_chain = np.arange(T)  # Time-indexed chain
        
        rewards, actions, states = [], [], []
        current_state = np.random.choice(2)  # Start randomly
        
        for t in range(T):
            z_t = exo_chain[t]
            P = transition_kernels[z_t]
            
            # Choose action based on policy
            if policy == "treatment":
                a_t = 1
            elif policy == "control":
                a_t = 0
            elif policy == "random":
                a_t = np.random.choice([0, 1])
            else:
                raise ValueError(f"Unknown policy: {policy}")
            
            # Generate reward
            mean_reward = self.reward_matrix[current_state, a_t]
            reward = np.random.normal(loc=mean_reward, scale=self.reward_std)
            
            rewards.append(reward)
            actions.append(a_t)
            states.append(current_state)
            
            # Transition to next state
            next_state = np.random.choice(2, p=P[current_state, a_t])
            current_state = next_state
        
        return np.array(rewards), np.array(actions), np.array(states)
    
    def simulate_treatment_policy(self, T: int, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Simulate pure treatment policy."""
        return self._simulate_mdp("treatment", T, seed)
    
    def simulate_control_policy(self, T: int, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Simulate pure control policy."""
        return self._simulate_mdp("control", T, seed)
    
    def simulate_experiment(self, design: ExperimentDesign, T: int, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Simulate experiment according to design."""
        if seed is not None:
            np.random.seed(seed)
        
        transition_kernels = self._generate_transition_kernels(T)
        exo_chain = np.arange(T)
        
        design.reset(seed)
        
        rewards, actions, states = [], [], []
        current_state = np.random.choice(2)
        
        for t in range(T):
            z_t = exo_chain[t]
            P = transition_kernels[z_t]
            
            # Get action from experimental design
            a_t = design.assign_treatment(t, current_state)
            
            # Generate reward
            mean_reward = self.reward_matrix[current_state, a_t]
            reward = np.random.normal(loc=mean_reward, scale=self.reward_std)
            
            rewards.append(reward)
            actions.append(a_t)
            states.append(current_state)
            
            # Transition
            next_state = np.random.choice(2, p=P[current_state, a_t])
            current_state = next_state
        
        return np.array(rewards), np.array(actions), np.array(states)
    
    def compute_true_ate(self, T: int, seed: Optional[int] = None) -> float:
        """Compute true average treatment effect."""
        # Use deterministic seeds for treatment and control to ensure reproducibility
        treat_seed = seed if seed is not None else 1000
        control_seed = treat_seed + 10000  # Large offset to avoid overlap
        
        r_treat, _, _ = self.simulate_treatment_policy(T, treat_seed)
        r_control, _, _ = self.simulate_control_policy(T, control_seed)
        return (np.sum(r_treat) - np.sum(r_control)) / T
    
    @property
    def name(self) -> str:
        return "TwoStateMDP"