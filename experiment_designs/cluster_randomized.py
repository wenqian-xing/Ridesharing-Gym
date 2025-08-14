"""
Cluster randomized design based on state features.
"""

import numpy as np
from typing import Optional, Any
from treatment_effect_gym import ExperimentDesign


class ClusterRandomizedDesign(ExperimentDesign):
    """Cluster randomized design based on state features."""
    
    def __init__(self, cluster_feature: str = "mod2", p: float = 0.5):
        """
        Args:
            cluster_feature: How to define clusters ("mod2", "threshold", etc.)
            p: Probability of treatment assignment per cluster
        """
        self.cluster_feature = cluster_feature
        self.p = p
        self._cluster_assignments = {}
        self._seed = None
    
    def reset(self, seed: Optional[int] = None) -> None:
        """Reset design state."""
        if seed is not None:
            self._seed = seed
            np.random.seed(seed)
        self._cluster_assignments = {}
    
    def _get_cluster(self, state: Any) -> int:
        """Map state to cluster ID."""
        if self.cluster_feature == "mod2":
            # Simple binary clustering
            if isinstance(state, (int, float)):
                return int(state) % 2
            elif hasattr(state, '__len__') and len(state) > 0:
                return int(state[0]) % 2
            else:
                return 0
        elif self.cluster_feature == "threshold":
            # Threshold-based clustering
            threshold = 50  # Default threshold
            if isinstance(state, (int, float)):
                return int(state > threshold)
            elif hasattr(state, '__len__') and len(state) > 0:
                return int(state[0] > threshold)
            else:
                return 0
        else:
            # Default to single cluster
            return 0
    
    def assign_treatment(self, t: int, state: Any, **kwargs) -> int:
        """Assign treatment based on cluster membership."""
        cluster_id = self._get_cluster(state)
        
        # Assign treatment to cluster if not done yet
        if cluster_id not in self._cluster_assignments:
            self._cluster_assignments[cluster_id] = int(np.random.random() < self.p)
        
        return self._cluster_assignments[cluster_id]
    
    @property
    def name(self) -> str:
        return f"ClusterRandomized({self.cluster_feature},p={self.p})"