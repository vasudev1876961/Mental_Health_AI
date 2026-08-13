"""
Unit Tests for Stage 5 & 8 Risk Prediction Models, Prediction Heads, and Multi-Task Loss.
"""

import unittest
import torch

from src.models.heads import RiskPredictionHeads
from src.models.losses import MultiTaskRiskLoss
from src.models.risk_model import MultimodalMentalHealthRiskModel


class TestStage5ModelArchitecture(unittest.TestCase):

    def setUp(self):
        self.batch_size = 4
        self.seq_len = 30
        self.v_seq = torch.randn(self.batch_size, self.seq_len, 18)
        self.a_feat = torch.randn(self.batch_size, 16)
        self.t_feat = torch.randn(self.batch_size, 128)
        self.mask = torch.ones(self.batch_size, 3)

    def test_risk_heads(self):
        """Verify prediction heads output dimensions and value ranges."""
        heads = RiskPredictionHeads(fused_dim=128)
        fused = torch.randn(self.batch_size, 128)

        preds = heads(fused)
        self.assertEqual(preds["stress_score"].shape, (self.batch_size,))
        self.assertEqual(preds["stress_logits"].shape, (self.batch_size, 3))
        self.assertEqual(preds["fatigue"].shape, (self.batch_size,))
        self.assertEqual(preds["attention"].shape, (self.batch_size,))

        # Verify output value constraints
        self.assertTrue(torch.all(preds["stress_score"] >= 0.0) and torch.all(preds["stress_score"] <= 100.0))
        self.assertTrue(torch.all(preds["fatigue"] >= 0.0) and torch.all(preds["fatigue"] <= 1.0))
        self.assertTrue(torch.all(preds["attention"] >= 0.0) and torch.all(preds["attention"] <= 1.0))

    def test_multitask_loss(self):
        """Verify joint multi-task loss calculation."""
        loss_fn = MultiTaskRiskLoss()
        heads = RiskPredictionHeads(fused_dim=128)
        fused = torch.randn(self.batch_size, 128)
        preds = heads(fused)

        targets = {
            "stress_score": torch.tensor([25.0, 45.0, 75.0, 10.0]),
            "stress_class": torch.tensor([0, 1, 2, 0], dtype=torch.long),
            "fatigue": torch.tensor([0.2, 0.4, 0.8, 0.1]),
            "attention": torch.tensor([0.9, 0.7, 0.3, 0.95]),
        }

        total_loss, loss_dict = loss_fn(preds, targets)
        self.assertGreater(float(total_loss.item()), 0.0)
        self.assertIn("total_loss", loss_dict)
        self.assertIn("loss_reg", loss_dict)

    def test_unified_risk_model(self):
        """Verify unified MultimodalMentalHealthRiskModel forward pass."""
        model = MultimodalMentalHealthRiskModel()
        preds, modality_weights, confidence = model(self.v_seq, self.a_feat, self.t_feat, mask=self.mask)

        self.assertEqual(preds["stress_score"].shape, (self.batch_size,))
        self.assertEqual(modality_weights.shape, (self.batch_size, 3))
        self.assertEqual(confidence.shape, (self.batch_size,))


if __name__ == "__main__":
    unittest.main()
