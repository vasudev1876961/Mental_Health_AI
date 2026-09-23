"""
Continual Learning and Elastic Weight Consolidation Package.
"""

from .ewc import ElasticWeightConsolidation
from .active_learning import FederatedActiveLearner, simulate_active_learning_curve

__all__ = [
    "ElasticWeightConsolidation",
    "FederatedActiveLearner",
    "simulate_active_learning_curve",
]
