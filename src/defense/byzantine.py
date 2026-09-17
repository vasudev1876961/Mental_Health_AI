"""
Byzantine-Robust Aggregation and Adversarial Poisoning Attack Defense.

Provides defense mechanisms against malicious or corrupted federated edge clients:
- Multi-Krum Aggregation (Blanchard et al., NeurIPS)
- Coordinate-wise Trimmed Mean
- Coordinate-wise Median
- Adversarial Attack Simulation & Anomaly Detection
"""

from typing import List, Tuple, Dict, Optional
import numpy as np


class ByzantineRobustAggregator:
    """Byzantine-robust parameter aggregation algorithms."""

    def __init__(self, num_byzantine: int = 1):
        self.num_byzantine = max(0, num_byzantine)

    @staticmethod
    def _flatten_weights(weights: List[np.ndarray]) -> np.ndarray:
        """Flattens a list of parameter tensors into a 1D vector."""
        return np.concatenate([w.ravel() for w in weights])

    @staticmethod
    def _unflatten_weights(flat: np.ndarray, shapes: List[Tuple[int, ...]]) -> List[np.ndarray]:
        """Restores a 1D flat vector back into original parameter tensor shapes."""
        restored = []
        offset = 0
        for s in shapes:
            size = int(np.prod(s)) if len(s) > 0 else 1
            restored.append(flat[offset : offset + size].reshape(s).astype(np.float32))
            offset += size
        return restored

    def multi_krum(
        self,
        client_updates: List[List[np.ndarray]],
        num_byzantine: Optional[int] = None,
        num_selected: Optional[int] = None,
    ) -> Tuple[List[np.ndarray], List[int]]:
        """Multi-Krum aggregation algorithm.
        
        Args:
            client_updates: List of client parameter updates (each a list of np.ndarray)
            num_byzantine: Assumed number of malicious clients f (default: self.num_byzantine)
            num_selected: Number of benign candidates to average (default: m - 2f)

        Returns:
            Tuple of (aggregated_weights, selected_client_indices)
        """
        m = len(client_updates)
        if m <= 2:
            # Fallback to standard average if too few clients
            flat_all = [self._flatten_weights(u) for u in client_updates]
            avg_flat = np.mean(flat_all, axis=0)
            shapes = [w.shape for w in client_updates[0]]
            return self._unflatten_weights(avg_flat, shapes), list(range(m))

        f = self.num_byzantine if num_byzantine is None else num_byzantine
        # Constraint: 2f + 2 < m for theoretical guarantee, but tolerate edge bounds
        f = min(f, max(0, (m - 3) // 2))

        # Flatten updates
        flat_vectors = np.array([self._flatten_weights(u) for u in client_updates], dtype=np.float64)
        shapes = [w.shape for w in client_updates[0]]

        # Compute pairwise squared Euclidean distances
        dist_matrix = np.zeros((m, m), dtype=np.float64)
        for i in range(m):
            for j in range(i + 1, m):
                d = np.sum((flat_vectors[i] - flat_vectors[j]) ** 2)
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d

        # For each client i, sum the distances to its (m - f - 2) closest neighbors
        k_closest = max(1, m - f - 2)
        scores = np.zeros(m, dtype=np.float64)
        for i in range(m):
            sorted_dists = np.sort(dist_matrix[i])
            # sorted_dists[0] is dist to self (0.0), take the next k_closest
            scores[i] = np.sum(sorted_dists[1 : k_closest + 1])

        # Select top candidates with lowest scores
        k_select = num_selected if num_selected is not None else max(1, m - 2 * f)
        k_select = min(m, max(1, k_select))
        selected_indices = list(np.argsort(scores)[:k_select])

        avg_flat = np.mean(flat_vectors[selected_indices], axis=0)
        return self._unflatten_weights(avg_flat, shapes), selected_indices

    def trimmed_mean(
        self,
        client_updates: List[List[np.ndarray]],
        trim_ratio: float = 0.2,
    ) -> List[np.ndarray]:
        """Coordinate-wise Trimmed Mean aggregation.
        
        Args:
            client_updates: List of client parameter updates
            trim_ratio: Proportion of extreme values to trim from both ends (0.0 to 0.4)
        """
        m = len(client_updates)
        flat_vectors = np.array([self._flatten_weights(u) for u in client_updates], dtype=np.float64)
        shapes = [w.shape for w in client_updates[0]]

        k_trim = int(np.round(m * trim_ratio))
        if k_trim == 0 and trim_ratio > 0 and m >= 3:
            k_trim = 1

        if k_trim == 0 or 2 * k_trim >= m:
            avg_flat = np.mean(flat_vectors, axis=0)
            return self._unflatten_weights(avg_flat, shapes)

        # Sort coordinates across clients
        sorted_coords = np.sort(flat_vectors, axis=0)
        trimmed_coords = sorted_coords[k_trim : m - k_trim]
        avg_flat = np.mean(trimmed_coords, axis=0)
        return self._unflatten_weights(avg_flat, shapes)

    def coordinate_median(
        self,
        client_updates: List[List[np.ndarray]],
    ) -> List[np.ndarray]:
        """Coordinate-wise Median aggregation. Extremely robust against arbitrary magnitude attacks."""
        flat_vectors = np.array([self._flatten_weights(u) for u in client_updates], dtype=np.float64)
        shapes = [w.shape for w in client_updates[0]]
        median_flat = np.median(flat_vectors, axis=0)
        return self._unflatten_weights(median_flat, shapes)


class AdversarialAttackSimulator:
    """Simulates adversarial client poisoning attacks for security evaluation."""

    @staticmethod
    def sign_flip_attack(
        weights: List[np.ndarray], scale: float = 2.5
    ) -> List[np.ndarray]:
        """Inverts gradient/parameter signs with a scaling factor to derail convergence."""
        return [(-scale * w).astype(np.float32) for w in weights]

    @staticmethod
    def gaussian_noise_attack(
        weights: List[np.ndarray], mean: float = 0.0, std: float = 5.0, seed: Optional[int] = None
    ) -> List[np.ndarray]:
        """Injects large Gaussian noise to mask meaningful physiological patterns."""
        rng = np.random.RandomState(seed)
        poisoned = []
        for w in weights:
            noise = rng.normal(loc=mean, scale=std, size=w.shape).astype(np.float32)
            poisoned.append(w + noise)
        return poisoned

    @staticmethod
    def constant_offset_attack(
        weights: List[np.ndarray], offset: float = 10.0
    ) -> List[np.ndarray]:
        """Adds extreme constant offset to shift model outputs toward arbitrary bias."""
        return [(w + offset).astype(np.float32) for w in weights]

    @staticmethod
    def detect_anomalous_clients(
        client_updates: List[List[np.ndarray]], threshold_z: float = 2.5
    ) -> Dict[str, any]:
        """Detects and flags anomalous client updates using robust modified Z-scores based on median absolute deviation (MAD)."""
        flat_vectors = np.array(
            [ByzantineRobustAggregator._flatten_weights(u) for u in client_updates], dtype=np.float64
        )
        centroid = np.median(flat_vectors, axis=0)
        distances = np.linalg.norm(flat_vectors - centroid, axis=1)

        med_dist = float(np.median(distances))
        abs_devs = np.abs(distances - med_dist)
        mad = float(np.median(abs_devs))

        if mad > 1e-8:
            # Standard Iglewicz-Hoaglin modified Z-score
            z_scores = 0.6745 * (distances - med_dist) / mad
        else:
            # Fallback when benign clients have nearly identical updates
            mean_dist = float(np.mean(distances))
            std_dist = float(np.std(distances))
            if std_dist > 1e-8:
                z_scores = (distances - mean_dist) / std_dist
            else:
                z_scores = np.zeros_like(distances)
                for i, d in enumerate(distances):
                    if d > med_dist + 1e-3:
                        z_scores[i] = threshold_z + 1.0

        flagged = [int(i) for i, z in enumerate(z_scores) if z > threshold_z or (distances[i] > med_dist * 3.0 and distances[i] > 1.0)]

        return {
            "distances": [float(d) for d in distances],
            "z_scores": [float(z) for z in z_scores],
            "flagged_client_indices": flagged,
            "has_adversaries": len(flagged) > 0,
        }
