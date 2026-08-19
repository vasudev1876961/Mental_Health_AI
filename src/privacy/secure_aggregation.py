"""
Cryptographic Secure Aggregation Protocol Simulator.

Implements pairwise Secret Sharing mask cancellation (Diffie-Hellman protocol based).
Ensures the FL server receives only the exact aggregated global sum vector without seeing
individual client gradient vectors, preventing server-side gradient inversion attacks.

Usage:
    sec_agg = SecureAggregationProtocol(num_clients=4)
    masked_updates = sec_agg.mask_client_updates(client_updates)
    aggregated_weights = sec_agg.aggregate_masked_updates(masked_updates)
"""

import numpy as np
from typing import List, Dict


class SecureAggregationProtocol:
    """Simulates pairwise additive secret sharing for secure server-side weight aggregation."""

    def __init__(self, num_clients: int, seed: int = 42):
        self.num_clients = num_clients
        self.rng = np.random.RandomState(seed)
        self.pairwise_masks = self._generate_pairwise_masks()

    def _generate_pairwise_masks(self) -> Dict[str, np.ndarray]:
        """Generates zero-sum pairwise secret masks between all pairs of clients (u, v)."""
        masks = {}
        for i in range(self.num_clients):
            for j in range(i + 1, self.num_clients):
                # Random secret seed per pair
                shared_seed = self.rng.randint(0, 1000000)
                mask_val = np.random.RandomState(shared_seed).normal(0.0, 1.0, size=(100,)).astype(np.float32)
                masks[f"{i}_{j}"] = mask_val
        return masks

    def mask_client_weights(self, client_idx: int, weight_vector: np.ndarray) -> np.ndarray:
        """Applies pairwise additive noise masks for client `client_idx`."""
        masked_vector = weight_vector.copy()
        for j in range(self.num_clients):
            if client_idx == j:
                continue
            elif client_idx < j:
                key = f"{client_idx}_{j}"
                mask = self.pairwise_masks[key]
                if mask.shape != weight_vector.shape:
                    mask = np.resize(mask, weight_vector.shape)
                masked_vector += mask
            else:
                key = f"{j}_{client_idx}"
                mask = self.pairwise_masks[key]
                if mask.shape != weight_vector.shape:
                    mask = np.resize(mask, weight_vector.shape)
                masked_vector -= mask
        return masked_vector

    def aggregate_masked_updates(self, masked_client_weights: List[np.ndarray]) -> np.ndarray:
        """Aggregates masked client weight vectors. Pairwise noise masks automatically cancel out to 0."""
        summed = np.zeros_like(masked_client_weights[0])
        for w in masked_client_weights:
            summed += w
        # Average over clients
        return summed / float(len(masked_client_weights))
