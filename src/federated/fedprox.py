"""
FedProx Strategy and Parameter Aggregation Utilities.
"""

from typing import List, Tuple, Dict
import numpy as np


def aggregate_weights(client_results: List[Tuple[List[np.ndarray], int]]) -> List[np.ndarray]:
    """Computes sample-weighted average of client parameter updates (FedAvg / FedProx aggregation)."""
    total_samples = sum(num_samples for _, num_samples in client_results)
    if total_samples == 0:
        return client_results[0][0]

    # Initialize zero arrays matching model parameter shapes
    first_weights = client_results[0][0]
    avg_weights = [np.zeros_like(w, dtype=np.float64) for w in first_weights]

    for weights, num_samples in client_results:
        weight_factor = num_samples / total_samples
        for i, layer_weight in enumerate(weights):
            avg_weights[i] += layer_weight.astype(np.float64) * weight_factor

    return [w.astype(np.float32) for w in avg_weights]


class FedProxStrategy:
    """FedProx aggregation strategy wrapper."""

    def __init__(self, mu: float = 0.01):
        self.mu = mu

    def aggregate(self, client_results: List[Tuple[List[np.ndarray], int]]) -> List[np.ndarray]:
        return aggregate_weights(client_results)
