"""
Multi-Task Risk Prediction Heads.

Primary: Stress Risk Regression (0-100) & Stress Classification (Low/Med/High).
Secondary: Fatigue & Attention indicators.
"""

import torch
import torch.nn as nn
from typing import Dict


class RiskPredictionHeads(nn.Module):
    """Multi-task output heads for continuous risk scores and categorical indicators."""

    def __init__(self, fused_dim: int = 128, dropout: float = 0.1):
        super().__init__()

        # Primary: Continuous Stress Score (0.0 to 100.0)
        self.stress_reg_head = nn.Sequential(
            nn.Linear(fused_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # Primary: Categorical Stress Level (Low=0, Medium=1, High=2)
        self.stress_cls_head = nn.Sequential(
            nn.Linear(fused_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 3),
        )

        # Secondary: Fatigue (0.0 to 1.0)
        self.fatigue_head = nn.Sequential(
            nn.Linear(fused_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        # Secondary: Attention (0.0 to 1.0)
        self.attention_head = nn.Sequential(
            nn.Linear(fused_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, fused_embedding: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            fused_embedding: Tensor of shape [B, fused_dim]

        Returns:
            Dict containing:
            - 'stress_score': Continuous stress tensor [B] (0.0 to 100.0)
            - 'stress_logits': Stress classification logits [B, 3]
            - 'fatigue': Fatigue tensor [B] (0.0 to 1.0)
            - 'attention': Attention tensor [B] (0.0 to 1.0)
        """
        stress_score = self.stress_reg_head(fused_embedding).squeeze(-1) * 100.0
        stress_logits = self.stress_cls_head(fused_embedding)
        fatigue = self.fatigue_head(fused_embedding).squeeze(-1)
        attention = self.attention_head(fused_embedding).squeeze(-1)

        return {
            "stress_score": stress_score,
            "stress_logits": stress_logits,
            "fatigue": fatigue,
            "attention": attention,
        }
