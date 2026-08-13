"""
Late Fusion Module.

Combines unimodal representation vectors using learnable modality attention weights.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class LateFusion(nn.Module):
    """Late fusion with learnable adaptive modality weighting."""

    def __init__(
        self,
        vision_dim: int = 128,
        audio_dim: int = 16,
        text_dim: int = 128,
        fused_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.v_proj = nn.Linear(vision_dim, fused_dim)
        self.a_proj = nn.Linear(audio_dim, fused_dim)
        self.t_proj = nn.Linear(text_dim, fused_dim)

        self.modality_attention = nn.Sequential(
            nn.Linear(fused_dim * 3, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )

        self.out_norm = nn.LayerNorm(fused_dim)

    def forward(
        self,
        v_feat: torch.Tensor,
        a_feat: torch.Tensor,
        t_feat: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            v_feat: [B, vision_dim]
            a_feat: [B, audio_dim]
            t_feat: [B, text_dim]
            mask: [B, 3]

        Returns:
            Tuple of fused tensor [B, fused_dim] and modality attribution weights [B, 3]
        """
        v_h = torch.relu(self.v_proj(v_feat))
        a_h = torch.relu(self.a_proj(a_feat))
        t_h = torch.relu(self.t_proj(t_feat))

        concat_h = torch.cat([v_h, a_h, t_h], dim=1)
        attn_logits = self.modality_attention(concat_h)

        if mask is not None:
            # Mask out missing modalities with large negative value before softmax
            attn_logits = attn_logits + (1.0 - mask) * -1e9

        attn_weights = torch.softmax(attn_logits, dim=1) # [B, 3]

        # Weighted combination of unimodal embeddings
        v_w = attn_weights[:, 0:1] * v_h
        a_w = attn_weights[:, 1:2] * a_h
        t_w = attn_weights[:, 2:3] * t_h

        fused = self.out_norm(v_w + a_w + t_w)
        return fused, attn_weights
