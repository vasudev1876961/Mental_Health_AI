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

    def forward(
        self,
        vision_seq: torch.Tensor,
        audio_feat: torch.Tensor,
        text_feat: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Args:
            vision_seq: Tensor of shape [B, T, vision_dim]
            audio_feat: Tensor of shape [B, audio_dim]
            text_feat: Tensor of shape [B, text_dim]
            mask: Tensor of shape [B, 3] indicating modality availability

        Returns:
            Tuple of:
            - Prediction dict containing 'stress_score', 'stress_logits', 'fatigue', 'attention'
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

        # Confidence Estimation based on modality presence and quality
        if mask is not None:
            confidence = torch.sum(mask, dim=1) / 3.0
        else:
            confidence = torch.ones(vision_seq.size(0), device=vision_seq.device)

        return predictions, modality_weights, confidence
