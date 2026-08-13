"""
Text NLP Encoder Module.

Projects speech transcripts and semantic embeddings into a compact [B, 128] representation space.
"""

import torch
import torch.nn as nn
from typing import List, Union


class TextNLPEncoder(nn.Module):
    """Encodes spoken text transcripts into semantic embedding space."""

    def __init__(self, input_dim: int = 128, output_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, text_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            text_features: Tensor of shape [B, input_dim] or [input_dim]

        Returns:
            Tensor of shape [B, output_dim]
        """
        if text_features.dim() == 1:
            text_features = text_features.unsqueeze(0)
        return self.projection(text_features)
