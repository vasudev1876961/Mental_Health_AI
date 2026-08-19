"""
Self-Supervised Multimodal Contrastive Alignment Engine (InfoNCE Loss).

Aligns Vision, Audio, and Text representation spaces into a shared joint embedding space
using Normalized Temperature-scaled Cross-Entropy (InfoNCE) loss.

Usage:
    loss = multimodal_contrastive_loss(vision_emb, audio_emb, text_emb, temperature=0.07)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class MultimodalContrastiveHead(nn.Module):
    """Projects vision, audio, and text embeddings into a shared normalized projection space."""

    def __init__(self, in_dim: int = 128, proj_dim: int = 64, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.vision_proj = nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        )
        self.audio_proj = nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        )
        self.text_proj = nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        )

    def forward(
        self, v_emb: torch.Tensor, a_emb: torch.Tensor, t_emb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Projects and normalizes input embeddings."""
        v_proj = F.normalize(self.vision_proj(v_emb), dim=-1)
        a_proj = F.normalize(self.audio_proj(a_emb), dim=-1)
        t_proj = F.normalize(self.text_proj(t_emb), dim=-1)
        return v_proj, a_proj, t_proj

    def compute_infonce_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Computes pairwise InfoNCE contrastive loss between two normalized representations."""
        batch_size = z1.size(0)
        sim_matrix = torch.matmul(z1, z2.T) / self.temperature  # [B, B]
        labels = torch.arange(batch_size, device=z1.device)
        loss = F.cross_entropy(sim_matrix, labels)
        return loss

    def compute_multimodal_loss(
        self, v_emb: torch.Tensor, a_emb: torch.Tensor, t_emb: torch.Tensor
    ) -> torch.Tensor:
        """Computes symmetric 3-way multimodal contrastive loss (Vision-Audio, Vision-Text, Audio-Text)."""
        zv, za, zt = self.forward(v_emb, a_emb, t_emb)
        loss_va = self.compute_infonce_loss(zv, za)
        loss_vt = self.compute_infonce_loss(zv, zt)
        loss_at = self.compute_infonce_loss(za, zt)
        return (loss_va + loss_vt + loss_at) / 3.0
