"""
Early Fusion Module.

Concatenates vision, audio, and text embeddings into a single unified vector representation.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class EarlyFusion(nn.Module):
    """Early fusion via feature concatenation and non-linear projection."""

    def __init__(
        self,
        vision_dim: int = 128,
        audio_dim: int = 16,
        text_dim: int = 128,
        fused_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Audio feature projection
        self.audio_proj = nn.Linear(audio_dim, 64)
        total_dim = vision_dim + 64 + text_dim

        self.fusion_network = nn.Sequential(
            nn.Linear(total_dim, fused_dim * 2),
            nn.BatchNorm1d(fused_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fused_dim * 2, fused_dim),
            nn.LayerNorm(fused_dim),
        )

    def forward(
        self,
        v_feat: torch.Tensor,
        a_feat: torch.Tensor,
        t_feat: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            v_feat: Vision tensor [B, 128]
            a_feat: Audio tensor [B, 16]
            t_feat: Text tensor [B, 128]
            mask: Modality mask [B, 3] (1.0 present, 0.0 missing)

        Returns:
            Tuple of:
            - Fused representation tensor [B, fused_dim]
            - Modality contribution weights [B, 3]
        """
        a_proj = self.audio_proj(a_feat)

        if mask is not None:
            v_feat = v_feat * mask[:, 0:1]
            a_proj = a_proj * mask[:, 1:2]
            t_feat = t_feat * mask[:, 2:3]

        cat_feat = torch.cat([v_feat, a_proj, t_feat], dim=1)
        fused = self.fusion_network(cat_feat)

        # Approximate modality attribution weights
        if mask is not None:
            weights = mask / (torch.sum(mask, dim=1, keepdim=True) + 1e-5)
        else:
            weights = torch.full((v_feat.size(0), 3), 1.0 / 3.0, device=v_feat.device)

        return fused, weights
