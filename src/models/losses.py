"""
Joint Multi-Task Loss Module.

Combines Huber Loss for continuous stress regression, CrossEntropy for classification,
and MSE for secondary fatigue and attention indicators.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple


class MultiTaskRiskLoss(nn.Module):
    """Joint Multi-Task Loss function."""

    def __init__(
        self,
        weight_stress_reg: float = 1.0,
        weight_stress_cls: float = 0.5,
        weight_fatigue: float = 0.2,
        weight_attention: float = 0.2,
        weight_contrastive: float = 0.1,
    ):
        super().__init__()
        self.w_reg = weight_stress_reg
        self.w_cls = weight_stress_cls
        self.w_fatigue = weight_fatigue
        self.w_attention = weight_attention
        self.w_contrastive = weight_contrastive

        self.huber = nn.HuberLoss(delta=5.0)
        self.cross_entropy = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()

    def forward(
        self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            predictions: Output dict from RiskPredictionHeads (and optional contrastive_loss)
            targets: Target dict from DataLoader batch

        Returns:
            Tuple of total_loss tensor and component loss dict
        """
        loss_reg = self.huber(predictions["stress_score"], targets["stress_score"])
        loss_cls = self.cross_entropy(predictions["stress_logits"], targets["stress_class"])
        loss_fatigue = self.mse(predictions["fatigue"], targets["fatigue"])
        loss_attention = self.mse(predictions["attention"], targets["attention"])

        total_loss = (
            self.w_reg * loss_reg
            + self.w_cls * loss_cls
            + self.w_fatigue * loss_fatigue
            + self.w_attention * loss_attention
        )

        loss_components = {
            "total_loss": float(total_loss.item()),
            "loss_reg": float(loss_reg.item()),
            "loss_cls": float(loss_cls.item()),
            "loss_fatigue": float(loss_fatigue.item()),
            "loss_attention": float(loss_attention.item()),
        }

        if "contrastive_loss" in predictions and predictions["contrastive_loss"] is not None:
            loss_cl = predictions["contrastive_loss"]
            total_loss = total_loss + self.w_contrastive * loss_cl
            loss_components["loss_contrastive"] = float(loss_cl.item())
            loss_components["total_loss"] = float(total_loss.item())

        return total_loss, loss_components
