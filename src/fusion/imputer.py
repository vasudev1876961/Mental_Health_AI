"""
Dynamic Cross-Modal Representation Imputation Engine.

Reconstructs missing modality embeddings (e.g., when camera is occluded or microphone is muted)
from available sensor channels using cross-modal projection heads, replacing naive zero-masking.
"""

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalImputer(nn.Module):
    """Reconstructs missing modality representations using cross-attention and dense mappings."""

    def __init__(self, vision_dim: int = 128, audio_dim: int = 16, text_dim: int = 128, hidden_dim: int = 128):
        super().__init__()
        self.vision_dim = vision_dim
        self.audio_dim = audio_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim

        # Modality standardizers
        self.proj_v = nn.Linear(vision_dim, hidden_dim)
        self.proj_a = nn.Linear(audio_dim, hidden_dim)
        self.proj_t = nn.Linear(text_dim, hidden_dim)

        # Imputation generators (generate modality X from remaining modalities Y + Z)
        # Vision from Audio + Text
        self.impute_vision_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, vision_dim),
        )

        # Audio from Vision + Text
        self.impute_audio_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, audio_dim),
        )

        # Text from Vision + Audio
        self.impute_text_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, text_dim),
        )

    def forward(
        self,
        vision_feat: torch.Tensor,
        audio_feat: torch.Tensor,
        text_feat: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
        """Imputes missing modalities if mask indicates absences.

        Args:
            vision_feat: [B, vision_dim]
            audio_feat: [B, audio_dim]
            text_feat: [B, text_dim]
            mask: Optional [B, 3] indicating [vision_present, audio_present, text_present]

        Returns:
            Tuple of (imputed_vision, imputed_audio, imputed_text, imputation_metadata)
        """
        B = vision_feat.size(0)
        device = vision_feat.device

        if mask is None:
            mask = torch.ones(B, 3, device=device)

        v_norm = self.proj_v(vision_feat)
        a_norm = self.proj_a(audio_feat)
        t_norm = self.proj_t(text_feat)

        out_v = vision_feat.clone()
        out_a = audio_feat.clone()
        out_t = text_feat.clone()

        imputed_counts = {"vision": 0, "audio": 0, "text": 0}

        for i in range(B):
            m_v, m_a, m_t = mask[i, 0].item(), mask[i, 1].item(), mask[i, 2].item()

            # Impute Vision if absent and Audio/Text are present
            if m_v == 0.0 and (m_a > 0 or m_t > 0):
                at_context = torch.cat([a_norm[i : i + 1], t_norm[i : i + 1]], dim=-1)
                synth_v = self.impute_vision_head(at_context)
                out_v[i] = synth_v.squeeze(0)
                imputed_counts["vision"] += 1

            # Impute Audio if absent and Vision/Text are present
            if m_a == 0.0 and (m_v > 0 or m_t > 0):
                vt_context = torch.cat([v_norm[i : i + 1], t_norm[i : i + 1]], dim=-1)
                synth_a = self.impute_audio_head(vt_context)
                out_a[i] = synth_a.squeeze(0)
                imputed_counts["audio"] += 1

            # Impute Text if absent and Vision/Audio are present
            if m_t == 0.0 and (m_v > 0 or m_a > 0):
                va_context = torch.cat([v_norm[i : i + 1], a_norm[i : i + 1]], dim=-1)
                synth_t = self.impute_text_head(va_context)
                out_t[i] = synth_t.squeeze(0)
                imputed_counts["text"] += 1

        metadata = {
            "imputed_vision_ratio": round(imputed_counts["vision"] / B, 2),
            "imputed_audio_ratio": round(imputed_counts["audio"] / B, 2),
            "imputed_text_ratio": round(imputed_counts["text"] / B, 2),
        }

        return out_v, out_a, out_t, metadata

    def compute_reconstruction_loss(
        self,
        vision_feat: torch.Tensor,
        audio_feat: torch.Tensor,
        text_feat: torch.Tensor,
    ) -> torch.Tensor:
        """Self-supervised cross-modal reconstruction loss on complete multimodal data."""
        v_norm = self.proj_v(vision_feat)
        a_norm = self.proj_a(audio_feat)
        t_norm = self.proj_t(text_feat)

        synth_v = self.impute_vision_head(torch.cat([a_norm, t_norm], dim=-1))
        synth_a = self.impute_audio_head(torch.cat([v_norm, t_norm], dim=-1))
        synth_t = self.impute_text_head(torch.cat([v_norm, a_norm], dim=-1))

        loss_v = F.mse_loss(synth_v, vision_feat)
        loss_a = F.mse_loss(synth_a, audio_feat)
        loss_t = F.mse_loss(synth_t, text_feat)

        return (loss_v + loss_a + loss_t) / 3.0
