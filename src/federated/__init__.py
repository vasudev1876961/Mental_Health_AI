"""
Federated Module: Flower Client, Server, FedAvg, FedProx, and Non-IID metrics tracker.
"""

from .client import MentalHealthFlowerClient
from .fedprox import FedProxStrategy
from .metrics import FLMetricsTracker
from .async_fl import AsyncFLServer, AsyncFLClient, simulate_heterogeneous_async_session
from .clustered_fl import ClusteredFLServer, ClinicalClusterManager, simulate_clustered_fl_session

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
]

