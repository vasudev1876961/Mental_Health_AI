"""
Sliding Window Sequence Buffer.

Maintains a sliding temporal window of 18-dim frame features [30, 18] for real-time inference.
"""

from collections import deque
import numpy as np
import torch


class SlidingWindowBuffer:
    """Fixed-size FIFO queue for temporal sequence frames."""

    def __init__(self, window_size: int = 30, feature_dim: int = 18):
        self.window_size = window_size
        self.feature_dim = feature_dim
        self.buffer = deque(maxlen=window_size)

        # Initialize with default neutral features
        default_feat = np.zeros(feature_dim, dtype=np.float32)
        default_feat[0] = 0.25 # ear
        default_feat[1] = 0.15 # mar
        default_feat[8] = 0.60 # eye openness
        default_feat[11] = 0.70 # neutral emotion default

        for _ in range(window_size):
            self.buffer.append(default_feat)

    def add_frame(self, feature_vector: np.ndarray):
        """Pushes new frame feature vector into queue."""
        if len(feature_vector) < self.feature_dim:
            feature_vector = np.pad(feature_vector, (0, self.feature_dim - len(feature_vector)))
        self.buffer.append(feature_vector[: self.feature_dim])

    def get_sequence_tensor(self) -> torch.Tensor:
        """Returns sequence tensor of shape [1, window_size, feature_dim]."""
        arr = np.array(self.buffer, dtype=np.float32)
        tensor = torch.tensor(arr, dtype=torch.float32).unsqueeze(0)
        return tensor

    def is_full(self) -> bool:
        return len(self.buffer) == self.window_size
