"""
Federated Module: Flower Client, Server, FedAvg, FedProx, FedAsync, Clustered FL, and FedDG.
"""

from .client import MentalHealthFlowerClient
from .fedprox import FedProxStrategy
from .metrics import FLMetricsTracker
from .async_fl import AsyncFLServer, AsyncFLClient, simulate_heterogeneous_async_session
from .clustered_fl import ClusteredFLServer, ClinicalClusterManager, simulate_clustered_fl_session
from .domain_generalization import (
    DomainInvariantAggregator,
    InvariantFeatureProjector,
    simulate_feddg_cross_domain_adaptation,
)

__all__ = [
    "MentalHealthFlowerClient",
    "FedProxStrategy",
    "FLMetricsTracker",
    "AsyncFLServer",
    "AsyncFLClient",
    "simulate_heterogeneous_async_session",
    "ClusteredFLServer",
    "ClinicalClusterManager",
    "simulate_clustered_fl_session",
    "DomainInvariantAggregator",
    "InvariantFeatureProjector",
    "simulate_feddg_cross_domain_adaptation",
]
