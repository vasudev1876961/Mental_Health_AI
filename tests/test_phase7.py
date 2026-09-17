"""
Unit Tests for Phase 7 Advanced Modules:
- Byzantine-Robust Defense (Multi-Krum, Trimmed Mean, Median, Poisoning Attacks)
- Distribution-Free Conformal Prediction (Residuals, Quantiles, Coverage)
- Elastic Weight Consolidation (EWC Fisher Information & Penalty)
- Dynamic Cross-Modal Imputer (Generative Synthesis & Reconstruction Loss)
- End-to-End Real-Time Pipeline & FastAPI Phase 7 Endpoints
"""

import unittest
import torch
import numpy as np
from fastapi.testclient import TestClient

from src.defense.byzantine import ByzantineRobustAggregator, AdversarialAttackSimulator
from src.uncertainty.conformal import ConformalRiskPredictor
from src.continual.ewc import ElasticWeightConsolidation
from src.fusion.imputer import CrossModalImputer
from src.models.risk_model import MultimodalMentalHealthRiskModel
from src.inference.realtime import RealtimeInferenceEngine
from src.api.server import app


class TestPhase7Modules(unittest.TestCase):

    def test_byzantine_multi_krum_defense(self):
        """Verify Multi-Krum rejects poisoned clients and recovers benign updates."""
        aggregator = ByzantineRobustAggregator(num_byzantine=1)
        # 5 clients: 4 benign centered at 3.0, 1 poisoned with sign flip
        rng = np.random.RandomState(42)
        benign = [rng.normal(3.0, 0.1, size=50).astype(np.float32) for _ in range(4)]
        poisoned = AdversarialAttackSimulator.sign_flip_attack([benign[0]], scale=5.0)[0]
        all_clients = [[benign[0]], [benign[1]], [benign[2]], [benign[3]], [poisoned]]

        agg_weights, selected = aggregator.multi_krum(all_clients, num_byzantine=1, num_selected=3)
        # Poisoned client (index 4) must NOT be among selected
        self.assertNotIn(4, selected)
        # Result must be close to benign 3.0
        self.assertAlmostEqual(float(np.mean(agg_weights[0])), 3.0, delta=0.25)

    def test_byzantine_trimmed_mean_and_median(self):
        """Verify Trimmed Mean and Coordinate Median resist extreme offset attack."""
        aggregator = ByzantineRobustAggregator()
        benign = [np.ones((20,), dtype=np.float32) * 2.0 for _ in range(5)]
        poisoned = AdversarialAttackSimulator.constant_offset_attack([benign[0]], offset=100.0)[0]
        client_updates = [[b] for b in benign] + [[poisoned]]

        med_weights = aggregator.coordinate_median(client_updates)
        self.assertAlmostEqual(float(np.mean(med_weights[0])), 2.0, delta=0.1)

        trimmed_weights = aggregator.trimmed_mean(client_updates, trim_ratio=0.15)
        self.assertAlmostEqual(float(np.mean(trimmed_weights[0])), 2.0, delta=0.5)

    def test_adversarial_anomaly_detection(self):
        """Verify detect_anomalous_clients flags outlier updates."""
        benign = [np.ones((30,), dtype=np.float32) * 1.5 for _ in range(4)]
        poisoned = [np.ones((30,), dtype=np.float32) * 50.0]
        report = AdversarialAttackSimulator.detect_anomalous_clients([[b] for b in benign] + [poisoned])
        self.assertTrue(report["has_adversaries"])
        self.assertIn(4, report["flagged_client_indices"])

    def test_conformal_prediction_intervals(self):
        """Verify ConformalRiskPredictor calculates finite-sample guaranteed intervals."""
        conformal = ConformalRiskPredictor(alpha=0.10)
        y_true = np.array([20.0, 30.0, 45.0, 60.0, 75.0, 90.0])
        y_pred = np.array([22.0, 28.0, 47.0, 58.0, 78.0, 88.0])

        q = conformal.calibrate(y_true, y_pred)
        self.assertGreater(q, 0.0)
        self.assertTrue(conformal.is_calibrated)

        bounds = conformal.predict_interval(65.0)
        self.assertLessEqual(bounds["lower_bound"], 65.0)
        self.assertGreaterEqual(bounds["upper_bound"], 65.0)
        self.assertEqual(bounds["confidence_level"], 0.90)

        # Coverage evaluation
        eval_metrics = conformal.evaluate_coverage(y_true, y_pred)
        self.assertGreaterEqual(eval_metrics["empirical_coverage"], 0.80)

    def test_conformal_classification_set(self):
        """Verify conformal prediction set construction."""
        conformal = ConformalRiskPredictor(alpha=0.10)
        probs = [0.70, 0.25, 0.05]
        res = conformal.predict_classification_set(probs)
        self.assertIn("Low", res["prediction_set"])
        self.assertIn("Medium", res["prediction_set"])
        self.assertGreaterEqual(res["cumulative_mass"], 0.90)

    def test_elastic_weight_consolidation(self):
        """Verify EWC penalty computes quadratic loss based on weight shift."""
        model = MultimodalMentalHealthRiskModel()
        ewc = ElasticWeightConsolidation(model, ewc_lambda=100.0)

        # Mock initial optimal parameters theta* and Fisher matrix
        ewc.optimal_params = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}
        ewc.fisher_matrix = {n: torch.ones_like(p) * 0.5 for n, p in model.named_parameters() if p.requires_grad}
        ewc.has_prior_task = True

        # Penalty before shift must be 0
        self.assertEqual(ewc.compute_penalty_value(), 0.0)

        # Shift weights
        with torch.no_grad():
            for p in model.parameters():
                p.add_(0.1)

        pen = ewc.compute_penalty_value()
        self.assertGreater(pen, 0.0)

    def test_cross_modal_imputer(self):
        """Verify CrossModalImputer reconstructs missing sensor embeddings."""
        imputer = CrossModalImputer(vision_dim=128, audio_dim=16, text_dim=128, hidden_dim=128)
        v = torch.randn(4, 128)
        a = torch.randn(4, 16)
        t = torch.randn(4, 128)

        # Mask: Vision missing for client 0 [0, 1, 1], Audio missing for client 1 [1, 0, 1]
        mask = torch.tensor([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])

        out_v, out_a, out_t, meta = imputer(v, a, t, mask=mask)
        self.assertEqual(out_v.shape, (4, 128))
        self.assertEqual(out_a.shape, (4, 16))
        self.assertEqual(out_t.shape, (4, 128))
        self.assertEqual(meta["imputed_vision_ratio"], 0.25)
        self.assertEqual(meta["imputed_audio_ratio"], 0.25)

        # Reconstruction loss
        rec_loss = imputer.compute_reconstruction_loss(v, a, t)
        self.assertGreater(rec_loss.item(), 0.0)

    def test_model_and_realtime_engine_integration(self):
        """Verify MultimodalMentalHealthRiskModel and RealtimeInferenceEngine with Phase 7 features."""
        model = MultimodalMentalHealthRiskModel()
        v_seq = torch.randn(2, 30, 18)
        a_feat = torch.randn(2, 16)
        t_feat = torch.randn(2, 128)
        mask = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])

        # Forward with dynamic imputation
        preds, modality_weights, confidence = model(v_seq, a_feat, t_feat, mask=mask, impute_missing=True)
        self.assertIn("imputation_metadata", preds)
        self.assertEqual(modality_weights.shape, (2, 3))
        self.assertGreater(confidence[0].item(), 0.0)

        # Real-time inference engine
        engine = RealtimeInferenceEngine(window_size=10)
        dummy_frame = np.full((480, 640, 3), 160, dtype=np.uint8)
        res = engine.process_frame(image=dummy_frame, impute_missing=True, conformal_alpha=0.10)
        self.assertIn("conformal_bounds", res)
        self.assertIn("lower_bound", res["conformal_bounds"])
        self.assertIn("upper_bound", res["conformal_bounds"])

    def test_fastapi_phase7_endpoints(self):
        """Verify new Phase 7 REST API endpoints."""
        client = TestClient(app)

        # 1. Root features
        r_root = client.get("/")
        self.assertEqual(r_root.status_code, 200)
        self.assertEqual(r_root.json()["version"], "2.1.0")

        # 2. Byzantine defense endpoint
        r_byz = client.post("/defense/byzantine/aggregate", json={
            "num_clients": 5,
            "num_byzantine": 1,
            "defense_method": "multi_krum",
            "vector_dim": 20,
            "attack_type": "sign_flip"
        })
        self.assertEqual(r_byz.status_code, 200)
        self.assertIn("selected_clients", r_byz.json())

        # 3. Conformal prediction endpoint
        r_conf = client.post("/uncertainty/conformal/predict", json={
            "predicted_stress": 64.0,
            "confidence_level": 0.90
        })
        self.assertEqual(r_conf.status_code, 200)
        data_conf = r_conf.json()
        self.assertLessEqual(data_conf["lower_bound"], 64.0)
        self.assertGreaterEqual(data_conf["upper_bound"], 64.0)

        # 4. Impute endpoint
        r_imp = client.post("/fusion/impute", json={
            "vision_present": False,
            "audio_present": True,
            "text_present": True,
            "latent_dim": 128
        })
        self.assertEqual(r_imp.status_code, 200)
        self.assertIn("Vision (Face/Pose)", r_imp.json()["imputed_modalities"])


if __name__ == "__main__":
    unittest.main()
