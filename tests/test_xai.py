"""
Unit Tests for Stage 11 Explainable AI & Uncertainty Estimator.
"""

import unittest
import numpy as np
import torch

from src.explainability.shap_explainer import SHAPBehaviorExplainer
from src.explainability.gradcam import GradCAMExplainer
from src.explainability.modality_attribution import ModalityAttributionAnalyzer
from src.explainability.uncertainty import UncertaintyEstimator


class TestStage11XAI(unittest.TestCase):

    def test_shap_explainer(self):
        """Verify SHAP behavioral feature ranking output."""
        explainer = SHAPBehaviorExplainer()
        dummy_feat = np.random.randn(18).astype(np.float32)

        attributions = explainer.explain_instance(dummy_feat, stress_score=72.0)
        self.assertEqual(len(attributions), 18)
        self.assertIn("feature", attributions[0])
        self.assertIn("attribution", attributions[0])

    def test_gradcam_heatmap(self):
        """Verify Grad-CAM heatmap generation."""
        explainer = GradCAMExplainer(model=torch.nn.Identity())
        dummy_face = np.full((224, 224, 3), 120, dtype=np.uint8)

        heatmap = explainer.generate_heatmap(dummy_face)
        self.assertEqual(heatmap.shape, (224, 224, 3))

    def test_modality_attribution(self):
        """Verify percentage breakdown across modalities."""
        analyzer = ModalityAttributionAnalyzer()
        weights = torch.tensor([0.5, 0.3, 0.2])

        pcts = analyzer.compute_attribution_percentage(weights)
        self.assertEqual(pcts["vision_pct"], 50.0)
        self.assertEqual(pcts["audio_pct"], 30.0)
        self.assertEqual(pcts["text_pct"], 20.0)

    def test_uncertainty_estimator(self):
        """Verify confidence and quality warning calculation."""
        estimator = UncertaintyEstimator()
        eval_res = estimator.evaluate_quality(face_confidence=0.9, audio_rms=0.15, modality_mask=np.array([1, 1, 1]))

        self.assertIn("confidence_score", eval_res)
        self.assertGreater(eval_res["confidence_score"], 0.5)
        self.assertTrue(eval_res["is_reliable"])


if __name__ == "__main__":
    unittest.main()
