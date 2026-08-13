"""
Missing Modality Masker Utility.

Handles missing camera/mic/text streams dynamically.
"""

import torch
import numpy as np
from typing import Tuple


class MissingModalityMasker:
    """Simulates and applies missing modality masking vectors."""

    def __init__(self, drop_prob: float = 0.15):
        self.drop_prob = drop_prob

    def generate_mask(self, batch_size: int, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        """Generates random modality mask tensor of shape [B, 3]."""
        mask = torch.ones(batch_size, 3, dtype=torch.float32, device=device)
        for i in range(batch_size):
            if np.random.rand() < self.drop_prob:
                drop_modality = np.random.randint(0, 3)
                mask[i, drop_modality] = 0.0
        return mask
