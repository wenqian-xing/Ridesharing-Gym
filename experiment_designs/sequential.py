"""
Sequential experimental design with alternating treatment.
"""

import numpy as np
from typing import Optional, Any
from treatment_effect_gym import ExperimentDesign


class SequentialDesign(ExperimentDesign):
    """Sequential experimental design with alternating treatment."""
    
    def __init__(self, block_size: int = 1):
        """
        Args:
            block_size: Size of treatment blocks
        """
        self.block_size = block_size
        self._seed = None
    
    def reset(self, seed: Optional[int] = None) -> None:
        """Reset design state."""
        if seed is not None:
            self._seed = seed
            np.random.seed(seed)
    
    def assign_treatment(self, t: int, state: Any, **kwargs) -> int:
        """Assign treatment sequentially."""
        # Alternate treatment in blocks
        block_id = t // self.block_size
        return block_id % 2
    
    @property
    def name(self) -> str:
        return f"Sequential(block={self.block_size})"