"""
Cross-Modal Attention Fusion Module.

Exchanges context representations dynamically between Vision, Audio, and Text streams.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class CrossModalAttentionFusion(nn.Module):
    """Multi-head Cross-Modal Attention Fusion."""

    def __init__(
        self,
        vision_dim: int = 128,
        audio_dim: int = 16,
        text_dim: int = 128,
        fused_dim: int = 128,
        nhead: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.v_proj = nn.Linear(vision_dim, fused_dim)
        self.a_proj = nn.Linear(audio_dim, fused_dim)
        self.t_proj = nn.Linear(text_dim, fused_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=fused_dim, num_heads=nhead, dropout=dropout, batch_first=True
        )

        self.norm = nn.LayerNorm(fused_dim)
        self.fc = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.GELU(),
            nn.Dropout(dropout),
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
            v_feat: [B, vision_dim]
            a_feat: [B, audio_dim]
            t_feat: [B, text_dim]
            mask: [B, 3]

        Returns:
            Tuple of fused tensor [B, fused_dim] and modality attribution weights [B, 3]
        """
        v_h = self.v_proj(v_feat).unsqueeze(1) # [B, 1, fused_dim]
        a_h = self.a_proj(a_feat).unsqueeze(1) # [B, 1, fused_dim]
        t_h = self.t_proj(t_feat).unsqueeze(1) # [B, 1, fused_dim]

        # Stack into sequence of 3 modality tokens: [B, 3, fused_dim]
        modal_tokens = torch.cat([v_h, a_h, t_h], dim=1)

        # Cross-modal self-attention across 3 modality tokens
        attn_out, attn_weights = self.cross_attn(modal_tokens, modal_tokens, modal_tokens)
        # attn_out: [B, 3, fused_dim], attn_weights: [B, 3, 3]

        if mask is not None:
            mask_3d = mask.unsqueeze(-1) # [B, 3, 1]
            attn_out = attn_out * mask_3d

        # Mean pool across present modalities
        if mask is not None:
            pooled = torch.sum(attn_out, dim=1) / (torch.sum(mask, dim=1, keepdim=True) + 1e-5)
            attr_weights = mask / (torch.sum(mask, dim=1, keepdim=True) + 1e-5)
        else:
            pooled = torch.mean(attn_out, dim=1)
            attr_weights = torch.full((v_feat.size(0), 3), 1.0 / 3.0, device=v_feat.device)

        out = self.norm(pooled + self.fc(pooled))
        return out, attr_weights
