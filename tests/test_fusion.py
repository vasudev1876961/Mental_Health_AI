"""
Unit Tests for Stage 7 Multimodal Fusion & Missing Modality Masking.
"""

import unittest
import torch

from src.fusion.early import EarlyFusion
from src.fusion.late import LateFusion
from src.fusion.cross_attention import CrossModalAttentionFusion
from src.fusion.modality_mask import MissingModalityMasker


class TestStage7Fusion(unittest.TestCase):

    def setUp(self):
        self.batch_size = 4
        self.v_feat = torch.randn(self.batch_size, 128)
        self.a_feat = torch.randn(self.batch_size, 16)
        self.t_feat = torch.randn(self.batch_size, 128)
        self.mask = torch.ones(self.batch_size, 3)
        self.mask[0, 1] = 0.0 # Audio missing for sample 0

    def test_early_fusion(self):
        """Verify Early Fusion shape and modality attribution weights."""
        fusion = EarlyFusion()
        out, weights = fusion(self.v_feat, self.a_feat, self.t_feat, mask=self.mask)

        self.assertEqual(out.shape, (self.batch_size, 128))
        self.assertEqual(weights.shape, (self.batch_size, 3))

    def test_late_fusion(self):
        """Verify Late Fusion shape and adaptive modality attention weights."""
        fusion = LateFusion()
        out, weights = fusion(self.v_feat, self.a_feat, self.t_feat, mask=self.mask)

        self.assertEqual(out.shape, (self.batch_size, 128))
        self.assertEqual(weights.shape, (self.batch_size, 3))
        # Missing audio for sample 0 should have ~0.0 weight
        self.assertLess(float(weights[0, 1].item()), 0.05)

    def test_cross_attention_fusion(self):
        """Verify Cross-Modal Attention Fusion output shape."""
        fusion = CrossModalAttentionFusion()
        out, weights = fusion(self.v_feat, self.a_feat, self.t_feat, mask=self.mask)

        self.assertEqual(out.shape, (self.batch_size, 128))
        self.assertEqual(weights.shape, (self.batch_size, 3))

    def test_modality_masker(self):
        """Verify MissingModalityMasker batch shape."""
        masker = MissingModalityMasker(drop_prob=0.5)
        mask = masker.generate_mask(10)
        self.assertEqual(mask.shape, (10, 3))


if __name__ == "__main__":
    unittest.main()
