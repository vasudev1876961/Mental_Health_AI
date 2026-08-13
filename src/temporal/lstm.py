"""
BiLSTM Sequence Encoder Module.

Processes sliding window frame behavioral features [B, T, D_in] into sequence representation [B, D_out].
"""

import torch
import torch.nn as nn


class BiLSTMSequenceEncoder(nn.Module):
    """Bi-directional LSTM sequence encoder."""

    def __init__(
        self,
        input_dim: int = 18,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * self.num_directions, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [B, T, input_dim]
        Returns:
            Tensor of shape [B, hidden_dim]
        """
        # lstm_out: [B, T, hidden_dim * num_directions]
        lstm_out, (hn, cn) = self.lstm(x)

        # Global average pooling over temporal dimension
        pooled = torch.mean(lstm_out, dim=1)
        out = self.fc(pooled)
        return out
