"""
Experimental designs for treatment assignment.

This package contains individual experimental design implementations that can be used
with the treatment effect benchmark framework.
"""

from .randomized import RandomizedDesign
from .switchback import SwitchbackDesign
from .cluster_randomized import ClusterRandomizedDesign
from .threshold import ThresholdDesign
from .adaptive import AdaptiveDesign
from .sequential import SequentialDesign

__all__ = [
    'RandomizedDesign',
    'SwitchbackDesign',
    'ClusterRandomizedDesign',
    'ThresholdDesign',
    'AdaptiveDesign',
    'SequentialDesign'
]