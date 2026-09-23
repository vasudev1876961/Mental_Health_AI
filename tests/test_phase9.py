"""
Unit Tests for Phase 9 Frontier Modules:
- Hierarchical Clustered Federated Learning (FedCluster / Clinical CFL)
- Bi-Directional Cross-Modal Co-Attention & Dynamic Gated Fusion
- Federated Semi-Supervised Active Learning (FedActive)
- Clinical Pareto-Optimal Risk Calibration
- Dynamic Quantization Profiler
- End-to-End Real-Time Pipeline Integration
- FastAPI Phase 9 REST API Endpoints
"""

import unittest
import numpy as np
import torch
import torch.nn as nn
from fastapi.testclient import TestClient

from src.federated.clustered_fl import (
    ClinicalClusterManager,
    ClusteredFLServer,
    simulate_clustered_fl_session,
)
from src.fusion.co_attention import BiDirectionalCoAttention
from src.continual.active_learning import (
    FederatedActiveLearner,
    simulate_active_learning_curve,
)
from src.uncertainty.pareto_calibration import (
    ClinicalParetoCalibrator,
    generate_synthetic_pareto_evaluation,
)
from src.optimization.dynamic_quant import DynamicQuantizationProfiler
from src.inference.realtime import RealtimeInferenceEngine
from src.models.risk_model import MultimodalMentalHealthRiskModel
from src.api.server import app


class TestPhase9Modules(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

    # ---------------------------------------------------------
    # 1. Clustered Federated Learning Tests
    # ---------------------------------------------------------
    def test_cosine_similarity_computation(self):
        """Verify parameter cosine similarity and edge cases."""
        v1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        v2 = np.array([2.0, 4.0, 6.0], dtype=np.float32)
        v3 = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
        v_zero = np.zeros(3, dtype=np.float32)

        # Identical direction
        sim_collinear = ClinicalClusterManager.compute_cosine_similarity(v1, v2)
        self.assertAlmostEqual(sim_collinear, 1.0, places=4)

        # Opposite direction
        sim_opposite = ClinicalClusterManager.compute_cosine_similarity(v1, v3)
        self.assertAlmostEqual(sim_opposite, -1.0, places=4)

        # Zero vector handling
        sim_zero = ClinicalClusterManager.compute_cosine_similarity(v1, v_zero)
        self.assertEqual(sim_zero, 0.0)

    def test_clustered_fl_server_aggregation(self):
        """Verify ClusteredFLServer partitions clients into phenotypes and aggregates."""
        server = ClusteredFLServer(num_clusters=3, vector_dim=20)
        rng = np.random.RandomState(42)

        # 6 clients: 2 panic, 2 depressive, 2 neurotypical
        updates = {
            "c_panic_1": rng.normal(1.5, 0.1, size=20).astype(np.float32),
            "c_panic_2": rng.normal(1.5, 0.1, size=20).astype(np.float32),
            "c_depr_1": rng.normal(-1.5, 0.1, size=20).astype(np.float32),
            "c_depr_2": rng.normal(-1.5, 0.1, size=20).astype(np.float32),
            "c_norm_1": rng.normal(0.0, 0.1, size=20).astype(np.float32),
            "c_norm_2": rng.normal(0.0, 0.1, size=20).astype(np.float32),
        }

        report = server.aggregate_cluster_updates(updates)
        self.assertIn("active_clusters", report)
        self.assertGreaterEqual(report["active_clusters"], 2)
        self.assertEqual(len(report["client_cluster_map"]), 6)

        # Check that panic clients share the same cluster
        self.assertEqual(
            report["client_cluster_map"]["c_panic_1"],
            report["client_cluster_map"]["c_panic_2"],
        )
        # Check that depressive clients share the same cluster
        self.assertEqual(
            report["client_cluster_map"]["c_depr_1"],
            report["client_cluster_map"]["c_depr_2"],
        )

        # Verify cluster model retrieval
        model_panic = server.get_cluster_model("c_panic_1")
        self.assertEqual(len(model_panic), 20)

    def test_simulate_clustered_fl_session(self):
        """Verify clustered FL session simulation demonstrates personalization gain."""
        sim = simulate_clustered_fl_session(num_clients=6, rounds=3)
        self.assertIn("clustered_fl_mae", sim)
        self.assertIn("standard_fedavg_mae", sim)
        self.assertLess(sim["clustered_fl_mae"], sim["standard_fedavg_mae"])
        self.assertGreater(sim["personalization_gain_pct"], 10.0)

    # ---------------------------------------------------------
    # 2. Bi-Directional Cross-Modal Co-Attention Tests
    # ---------------------------------------------------------
    def test_coattention_forward_and_shapes(self):
        """Verify BiDirectionalCoAttention forward pass, affinity matrix, and dynamic gating."""
        co_attn = BiDirectionalCoAttention(
            vision_dim=18, audio_dim=16, text_dim=128, fused_dim=64
        )

        v = torch.randn(4, 18)
        a = torch.randn(4, 16)
        t = torch.randn(4, 128)
        mask = torch.tensor([[1.0, 1.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 0.0]])

        fused, affinity, mod_weights = co_attn(v, a, t, mask=mask)
        self.assertEqual(fused.shape, (4, 64))
        self.assertEqual(affinity.dim(), 3)  # [B, T_v, T_a]
        self.assertEqual(mod_weights.shape, (4, 3))

        # Modality weights must sum to approximately 1.0
        sums = mod_weights.sum(dim=-1).detach().numpy()
        np.testing.assert_allclose(sums, np.ones(4), atol=1e-3)

    def test_coattention_saliency_heatmap(self):
        """Verify cross-modal co-saliency heatmap extraction."""
        co_attn = BiDirectionalCoAttention(vision_dim=18, audio_dim=16, text_dim=128, fused_dim=64)
        v = torch.randn(1, 18)
        a = torch.randn(1, 16)

        saliency = co_attn.compute_saliency_heatmap(v, a)
        self.assertIn("cross_modal_alignment_score", saliency)
        self.assertIn("peak_alignment_coords", saliency)
        self.assertIn("peak_affinity_value", saliency)
        self.assertIn("status", saliency)
        self.assertGreaterEqual(saliency["cross_modal_alignment_score"], 0.0)

    # ---------------------------------------------------------
    # 3. Federated Active Learning Tests
    # ---------------------------------------------------------
    def test_active_learning_uncertainty_score(self):
        """Verify Conformal-Entropy sample uncertainty ranking."""
        learner = FederatedActiveLearner()

        # High ambiguity case (uniform probs, wide conformal interval)
        ambiguous = learner.compute_sample_uncertainty(
            probs=[0.33, 0.33, 0.34],
            conformal_lower=20.0,
            conformal_upper=85.0,
            confidence_score=0.70,
        )

        # Certain case (sharp probs, narrow interval)
        certain = learner.compute_sample_uncertainty(
            probs=[0.95, 0.03, 0.02],
            conformal_lower=22.0,
            conformal_upper=34.0,
            confidence_score=1.0,
        )

        self.assertGreater(ambiguous["acquisition_score"], certain["acquisition_score"])
        self.assertGreater(ambiguous["normalized_entropy"], certain["normalized_entropy"])
        self.assertGreater(ambiguous["conformal_interval_width"], certain["conformal_interval_width"])

    def test_active_learning_query_and_pseudolabeling(self):
        """Verify query ranking budget selection and temporal pseudo-label generation."""
        learner = FederatedActiveLearner()
        candidates = [
            {"id": "c1", "probs": [0.33, 0.33, 0.34], "conformal_lower": 20.0, "conformal_upper": 85.0, "confidence_score": 0.50},
            {"id": "c2", "probs": [0.92, 0.05, 0.03], "conformal_lower": 22.0, "conformal_upper": 32.0, "confidence_score": 0.95},
            {"id": "c3", "probs": [0.48, 0.48, 0.04], "conformal_lower": 40.0, "conformal_upper": 75.0, "confidence_score": 0.80},
            {"id": "c4", "probs": [0.02, 0.94, 0.04], "conformal_lower": 48.0, "conformal_upper": 58.0, "confidence_score": 0.95},
        ]

        report = learner.query_informative_samples(candidates, budget_fraction=0.50)
        self.assertEqual(report["clinician_queried_count"], 2)
        self.assertIn("query_reason", report["queried_samples"][0])

        # c2 or c4 should qualify for pseudo-labeling
        self.assertGreaterEqual(report["pseudo_labeled_count"], 1)
        pseudo_ids = [p["id"] for p in report["pseudo_labeled_samples"]]
        self.assertTrue("c2" in pseudo_ids or "c4" in pseudo_ids)

    def test_simulate_active_learning_curve(self):
        """Verify active learning budget curve simulation."""
        sim = simulate_active_learning_curve()
        self.assertIn("mae_curve", sim)
        self.assertIn("f1_curve", sim)
        self.assertGreaterEqual(sim["efficiency_gain_pct"], 80.0)

    # ---------------------------------------------------------
    # 4. Clinical Pareto-Optimal Risk Calibration Tests
    # ---------------------------------------------------------
    def test_pareto_calibrator_optimization(self):
        """Verify asymmetric cost calibration guarantees high sensitivity."""
        calibrator = ClinicalParetoCalibrator(cost_fn=10.0, cost_fp=1.0, min_sensitivity=0.95)
        rng = np.random.RandomState(42)

        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_scores = np.array([20.0, 32.0, 45.0, 52.0, 58.0, 68.0, 75.0, 88.0])

        stats = calibrator.calibrate(y_true, y_scores)
        self.assertTrue(calibrator.is_calibrated)
        self.assertGreaterEqual(stats["clinical_sensitivity"], 0.95)
        self.assertLessEqual(calibrator.optimal_threshold, 60.0)

        # Test triage classification
        high_risk_triage = calibrator.triage_risk(75.0)
        self.assertIn("Crisis", high_risk_triage["triage_level"])
        self.assertIn("Tier-1", high_risk_triage["urgency_tier"])

        healthy_triage = calibrator.triage_risk(25.0)
        self.assertIn("Normative", healthy_triage["triage_level"])

    # ---------------------------------------------------------
    # 5. Dynamic Quantization Profiler Tests
    # ---------------------------------------------------------
    def test_dynamic_quantization_profiler(self):
        """Verify INT8 dynamic quantization and precision profiling."""
        profiler = DynamicQuantizationProfiler()
        model = MultimodalMentalHealthRiskModel()

        quant_model = profiler.apply_dynamic_quantization(model)
        self.assertIsNotNone(quant_model)

        size_mb = profiler.measure_model_size_mb(model)
        self.assertGreater(size_mb, 0.0)

        bench = profiler.benchmark_precision_profiles(model, num_warmup=2, num_iters=5)
        self.assertIn("compression_ratio", bench)
        self.assertIn("memory_reduction_pct", bench)
        self.assertIn("inference_speedup", bench)
        self.assertGreater(bench["compression_ratio"], 1.0)

    # ---------------------------------------------------------
    # 6. Real-Time Pipeline Integration
    # ---------------------------------------------------------
    def test_realtime_inference_engine_phase9_payload(self):
        """Verify RealtimeInferenceEngine returns Phase 9 co-saliency, pareto, and active uncertainty."""
        engine = RealtimeInferenceEngine(window_size=15)
        dummy_frame = np.full((120, 160, 3), 160, dtype=np.uint8)

        res = engine.process_frame(image=dummy_frame, transcript_text="Feeling overwhelmed")
        self.assertIn("pareto_triage", res)
        self.assertIn("co_saliency", res)
        self.assertIn("active_learning_uncertainty", res)
        self.assertIn("triage_level", res["pareto_triage"])
        self.assertIn("cross_modal_alignment_score", res["co_saliency"])
        self.assertIn("acquisition_score", res["active_learning_uncertainty"])

    # ---------------------------------------------------------
    # 7. FastAPI Phase 9 REST API Endpoints
    # ---------------------------------------------------------
    def test_fastapi_phase9_endpoints(self):
        """Verify FastAPI v2.3.0 endpoints for Clustered FL, Co-Attention, Active Learning, and Pareto."""
        # 1. Root version check
        r_root = self.client.get("/")
        self.assertEqual(r_root.status_code, 200)
        self.assertEqual(r_root.json()["version"], "2.3.0")

        # 2. Clustered FL assign endpoint
        r_clust = self.client.post("/federated/cluster/assign", json={
            "num_clients": 6,
            "vector_dim": 20
        })
        self.assertEqual(r_clust.status_code, 200)
        data_clust = r_clust.json()
        self.assertEqual(data_clust["num_clients"], 6)
        self.assertIn("cluster_assignments", data_clust)
        self.assertIn("phenotype_labels", data_clust)

        # 3. Co-Attention Saliency endpoint
        r_co = self.client.post("/fusion/coattention/saliency", json={
            "vision_features": [0.5] * 18,
            "audio_features": [0.2] * 16,
        })
        self.assertEqual(r_co.status_code, 200)
        data_co = r_co.json()
        self.assertIn("cross_modal_alignment_score", data_co)
        self.assertIn("peak_affinity_value", data_co)

        # 4. Active learning query endpoint
        r_act = self.client.post("/active/query/sample", json={
            "num_candidates": 6,
            "budget_fraction": 0.33
        })
        self.assertEqual(r_act.status_code, 200)
        data_act = r_act.json()
        self.assertEqual(data_act["total_candidates"], 6)
        self.assertGreaterEqual(data_act["clinician_queried_count"], 1)

        # 5. Pareto calibration endpoint
        r_par = self.client.post("/calibration/pareto/threshold", json={
            "stress_score": 78.5,
            "cost_fn_ratio": 10.0
        })
        self.assertEqual(r_par.status_code, 200)
        data_par = r_par.json()
        self.assertEqual(data_par["input_stress_score"], 78.5)
        self.assertIn("Crisis Alert", data_par["triage_level"])
        self.assertIn("Tier-1", data_par["urgency_tier"])


if __name__ == "__main__":
    unittest.main()
