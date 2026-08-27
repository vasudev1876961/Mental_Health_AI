"""
Unified Multimodal Mental Health Risk Assessment Model.

Integrates Vision Sequence Encoders, Audio Prosody, Text NLP, Cross-Modal Fusion,
Missing-Modality Masking, Multi-Task Prediction Heads, and Confidence Scoring.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional

from src.temporal.transformer import TemporalTransformerEncoder
from src.temporal.lstm import BiLSTMSequenceEncoder
from src.fusion.cross_attention import CrossModalAttentionFusion
from src.fusion.early import EarlyFusion
from src.fusion.late import LateFusion
from src.fusion.contrastive import MultimodalContrastiveHead
from src.text.encoder import TextNLPEncoder
from .heads import RiskPredictionHeads


class MultimodalMentalHealthRiskModel(nn.Module):
    """Unified Deep Learning Model for Mental Health Behavioral Risk Assessment."""

    def __init__(
        self,
        vision_dim: int = 18,
        audio_dim: int = 16,
        text_dim: int = 128,
        hidden_dim: int = 128,
        temporal_type: str = "transformer",
        fusion_strategy: str = "cross_attention",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vision_dim = vision_dim
        self.audio_dim = audio_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim

        # 1. Temporal Vision Encoder
        if temporal_type == "transformer":
            self.temporal_encoder = TemporalTransformerEncoder(
                input_dim=vision_dim, hidden_dim=hidden_dim, dropout=dropout
            )
        else:
            self.temporal_encoder = BiLSTMSequenceEncoder(
                input_dim=vision_dim, hidden_dim=hidden_dim, dropout=dropout
            )

        # 2. Text NLP Encoder
        self.text_encoder = TextNLPEncoder(input_dim=text_dim, output_dim=hidden_dim, dropout=dropout)

        # 3. Multimodal Fusion Engine
        if fusion_strategy == "early":
            self.fusion_engine = EarlyFusion(
                vision_dim=hidden_dim, audio_dim=audio_dim, text_dim=hidden_dim, fused_dim=hidden_dim
            )
        elif fusion_strategy == "late":
            self.fusion_engine = LateFusion(
                vision_dim=hidden_dim, audio_dim=audio_dim, text_dim=hidden_dim, fused_dim=hidden_dim
            )
        else:
            self.fusion_engine = CrossModalAttentionFusion(
                vision_dim=hidden_dim, audio_dim=audio_dim, text_dim=hidden_dim, fused_dim=hidden_dim
            )

        # 4. Prediction Heads
        self.heads = RiskPredictionHeads(fused_dim=hidden_dim, dropout=dropout)

        # 5. Multimodal Contrastive Alignment Head (InfoNCE)
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.contrastive_head = MultimodalContrastiveHead(in_dim=hidden_dim, proj_dim=64, temperature=0.07)

    def extract_modality_embeddings(
        self, vision_seq: torch.Tensor, audio_feat: torch.Tensor, text_feat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extracts aligned feature representations for Vision, Audio, and Text streams."""
        if isinstance(self.temporal_encoder, TemporalTransformerEncoder):
            v_emb, _ = self.temporal_encoder(vision_seq)
        else:
            v_emb = self.temporal_encoder(vision_seq)
        a_emb = self.audio_proj(audio_feat)
        t_emb = self.text_encoder(text_feat)
        return v_emb, a_emb, t_emb

    def compute_contrastive_loss(
        self, vision_seq: torch.Tensor, audio_feat: torch.Tensor, text_feat: torch.Tensor
    ) -> torch.Tensor:
        """Computes InfoNCE 3-way multimodal contrastive alignment loss."""
        v_emb, a_emb, t_emb = self.extract_modality_embeddings(vision_seq, audio_feat, text_feat)
        return self.contrastive_head.compute_multimodal_loss(v_emb, a_emb, t_emb)

    def forward(
        self,
        vision_seq: torch.Tensor,
        audio_feat: torch.Tensor,
        text_feat: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_contrastive: bool = False,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Args:
            vision_seq: Tensor of shape [B, T, vision_dim]
            audio_feat: Tensor of shape [B, audio_dim]
            text_feat: Tensor of shape [B, text_dim]
            mask: Tensor of shape [B, 3] indicating modality availability
            return_contrastive: Whether to compute contrastive loss

        Returns:
            Tuple of:
            - Prediction dict containing 'stress_score', 'stress_logits', 'fatigue', 'attention'
              (and optionally 'contrastive_loss' if return_contrastive is True)
            - Modality attribution weights tensor [B, 3]
            - Confidence score tensor [B] (0.0 to 1.0)
        """
        # Encode temporal vision
        if isinstance(self.temporal_encoder, TemporalTransformerEncoder):
            v_emb, _ = self.temporal_encoder(vision_seq)
        else:
            v_emb = self.temporal_encoder(vision_seq)

        # Encode text
        t_emb = self.text_encoder(text_feat)

        # Apply Multimodal Fusion
        fused_emb, modality_weights = self.fusion_engine(v_emb, audio_feat, t_emb, mask=mask)

        # Risk Predictions
        predictions = self.heads(fused_emb)

        # Optional Contrastive Loss calculation
        if return_contrastive:
            a_emb = self.audio_proj(audio_feat)
            cl_loss = self.contrastive_head.compute_multimodal_loss(v_emb, a_emb, t_emb)
            predictions["contrastive_loss"] = cl_loss

        # Confidence Estimation based on modality presence and quality
        if mask is not None:
            confidence = torch.sum(mask, dim=1) / 3.0
        else:
            confidence = torch.ones(vision_seq.size(0), device=vision_seq.device)

        return predictions, modality_weights, confidence
