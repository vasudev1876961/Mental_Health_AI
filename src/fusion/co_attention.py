"""
Bi-Directional Cross-Modal Co-Attention and Dynamic Gated Fusion.

Computes dense affinity-based co-attention between facial video, audio prosody,
and verbal text streams with dynamic gating, producing fine-grained interpretable
cross-modal co-saliency maps.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any


class BiDirectionalCoAttention(nn.Module):
    """
    Bi-Directional Co-Attention with Dynamic Gated Fusion (DGF).

    Computes pairwise affinity matrices across modalities:
    - Vision-to-Audio (facial gestures conditioned on vocal pitch/tempo)
    - Audio-to-Vision (vocal tremor conditioned on micro-facial tension)
    - Dynamic Gated Fusion (DGF) unit regularized by verbal text context.
    """

    def __init__(
        self,
        vision_dim: int = 128,
        audio_dim: int = 16,
        text_dim: int = 128,
        fused_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vision_dim = vision_dim
        self.audio_dim = audio_dim
        self.text_dim = text_dim
        self.fused_dim = fused_dim

        # Modality projection heads to shared fused_dim
        self.v_proj = nn.Sequential(
            nn.Linear(vision_dim, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.GELU(),
        )
        self.a_proj = nn.Sequential(
            nn.Linear(audio_dim, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.GELU(),
        )
        self.t_proj = nn.Sequential(
            nn.Linear(text_dim, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.GELU(),
        )

        # Bilinear affinity weight matrix for Vision-Audio co-attention
        self.affinity_weight = nn.Parameter(torch.Tensor(fused_dim, fused_dim))
        nn.init.xavier_uniform_(self.affinity_weight)

        # Dynamic Gated Fusion (DGF) gate generator
        self.gate_fc = nn.Sequential(
            nn.Linear(fused_dim * 3, fused_dim),
            nn.Sigmoid(),
        )

        # Output projection and normalization
        self.output_fc = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fused_dim, fused_dim),
        )
        self.norm = nn.LayerNorm(fused_dim)

    def forward(
        self,
        v_feat: torch.Tensor,
        a_feat: torch.Tensor,
        t_feat: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for Bi-Directional Co-Attention.

        Args:
            v_feat: Vision features [B, V_dim] or temporal sequence [B, T_v, V_dim]
            a_feat: Audio features [B, A_dim] or temporal sequence [B, T_a, A_dim]
            t_feat: Text features [B, T_dim]
            mask: Optional modality presence indicator [B, 3]

        Returns:
            Tuple of:
            - Fused representation tensor [B, fused_dim]
            - Co-saliency affinity matrix [B, T_v, T_a] (or [B, 1, 1] if static)
            - Modality attribution weights [B, 3]
        """
        # Ensure temporal dimension exists: [B, T, D]
        is_v_2d = (v_feat.dim() == 2)
        is_a_2d = (a_feat.dim() == 2)

        v_seq = v_feat.unsqueeze(1) if is_v_2d else v_feat
        a_seq = a_feat.unsqueeze(1) if is_a_2d else a_feat
        t_seq = t_feat.unsqueeze(1) if t_feat.dim() == 2 else t_feat

        # 1. Project to shared dimension
        v_h = self.v_proj(v_seq)  # [B, T_v, fused_dim]
        a_h = self.a_proj(a_seq)  # [B, T_a, fused_dim]
        t_h = self.t_proj(t_seq).mean(dim=1)  # [B, fused_dim]

        # 2. Compute Bilinear Affinity Matrix C = tanh(V * W * A^T)
        # v_h: [B, T_v, D], affinity_weight: [D, D] -> [B, T_v, D]
        v_w = torch.matmul(v_h, self.affinity_weight)
        # [B, T_v, D] x [B, D, T_a] -> [B, T_v, T_a]
        affinity = torch.tanh(torch.bmm(v_w, a_h.transpose(1, 2)))

        # 3. Softmax normalized co-attention distributions
        # Vision-attended audio context
        attn_v2a = F.softmax(affinity, dim=-1)  # [B, T_v, T_a]
        h_v = torch.bmm(attn_v2a, a_h).mean(dim=1)  # [B, fused_dim]

        # Audio-attended vision context
        attn_a2v = F.softmax(affinity.transpose(1, 2), dim=-1)  # [B, T_a, T_v]
        h_a = torch.bmm(attn_a2v, v_h).mean(dim=1)  # [B, fused_dim]

        # 4. Modality Masking (Zero out missing channels)
        if mask is not None:
            # mask: [B, 3] -> vision, audio, text
            v_mask = mask[:, 0].unsqueeze(1)
            a_mask = mask[:, 1].unsqueeze(1)
            t_mask = mask[:, 2].unsqueeze(1)

            h_v = h_v * v_mask
            h_a = h_a * a_mask
            t_h = t_h * t_mask

        # 5. Dynamic Gated Fusion (DGF)
        gate_input = torch.cat([h_v, h_a, t_h], dim=-1)  # [B, fused_dim * 3]
        g = self.gate_fc(gate_input)  # [B, fused_dim] in (0, 1)

        # Gated combination: gate balances cross-modal sensor dynamics vs verbal semantics
        fused = g * (h_v + h_a) + (1.0 - g) * t_h

        # Residual connection + LayerNorm + FeedForward
        fused = self.norm(fused + self.output_fc(fused))

        # Modality attribution weights
        norm_v = torch.norm(h_v, p=2, dim=-1, keepdim=True)
        norm_a = torch.norm(h_a, p=2, dim=-1, keepdim=True)
        norm_t = torch.norm(t_h, p=2, dim=-1, keepdim=True)
        total = norm_v + norm_a + norm_t + 1e-6
        modality_weights = torch.cat([norm_v / total, norm_a / total, norm_t / total], dim=-1)

        return fused, affinity, modality_weights

    def compute_saliency_heatmap(
        self,
        v_feat: torch.Tensor,
        a_feat: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Extracts human-interpretable cross-modal co-saliency alignment metrics.
        Identifies highest-affinity cross-modal correlation peaks.
        """
        self.eval()
        with torch.no_grad():
            is_v_2d = (v_feat.dim() == 2)
            is_a_2d = (a_feat.dim() == 2)

            v_seq = v_feat.unsqueeze(1) if is_v_2d else v_feat
            a_seq = a_feat.unsqueeze(1) if is_a_2d else a_feat

            v_h = self.v_proj(v_seq)
            a_h = self.a_proj(a_seq)

            v_w = torch.matmul(v_h, self.affinity_weight)
            affinity = torch.tanh(torch.bmm(v_w, a_h.transpose(1, 2)))
            aff_np = affinity.squeeze(0).cpu().numpy()

            # Cross-modal correlation index: mean absolute affinity
            alignment_score = float(np.mean(np.abs(aff_np)))

            # Peak alignment coordinates
            max_idx = np.unravel_index(np.argmax(aff_np), aff_np.shape)

            return {
                "affinity_matrix": aff_np.tolist(),
                "cross_modal_alignment_score": round(alignment_score, 4),
                "peak_alignment_coords": [int(max_idx[0]), int(max_idx[1])],
                "peak_affinity_value": round(float(aff_np[max_idx]), 4),
                "status": "High Cross-Modal Coherence" if alignment_score > 0.25 else "Moderate Alignment",
            }
