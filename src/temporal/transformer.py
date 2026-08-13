"""
Temporal Transformer Encoder Module.

Models long-range temporal micro-expression dynamics using multi-head self-attention.
"""

import math
import torch
import torch.nn as nn
from typing import Tuple, Optional


class PositionalEncoding(nn.Module):
    """Sinusoidal Positional Encoding for temporal sequence frames."""

    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x shape: [B, T, d_model]"""
        return x + self.pe[:, : x.size(1)]


class TemporalTransformerEncoder(nn.Module):
    """Transformer Encoder for temporal behavioral sequence features."""

    def __init__(
        self,
        input_dim: int = 18,
        hidden_dim: int = 128,
        num_layers: int = 2,
        nhead: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Tensor of shape [B, T, input_dim]

        Returns:
            Tuple of:
            - Out tensor of shape [B, hidden_dim]
            - Temporal attention features [B, T, hidden_dim]
        """
        # Linear projection + Positional Encoding
        h = self.input_proj(x)
        h = self.pos_encoder(h)

        # Transformer encoding: [B, T, hidden_dim]
        tf_out = self.transformer_encoder(h)
        tf_out = self.norm(tf_out)

        # Mean pooling across sequence dimension
        pooled = torch.mean(tf_out, dim=1)
        return pooled, tf_out
