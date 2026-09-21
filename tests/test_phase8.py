"""
Unit Tests for Phase 8 Frontier Modules:
- Contactless Physiological rPPG & Autonomic HRV Biomarkers (POS algorithm, RMSSD, SDNN, Baevsky SI)
- Causal Multimodal Counterfactual Recourse & Actionable Clinical Prescriptions
- Asynchronous Federated Learning (FedAsync) with Dynamic Staleness Decay
- ONNX Runtime Edge Acceleration & Magnitude Weight Pruning
- FastAPI Phase 8 REST Endpoints
"""

import unittest
import numpy as np
import torch
import torch.nn as nn
from fastapi.testclient import TestClient

from src.physiological.rppg import RemotePPGExtractor, HRVMetrics
from src.explainability.counterfactual import CounterfactualRecourseEngine
from src.federated.async_fl import (
    AsyncFLServer,
    AsyncFLClient,
    StalenessFunction,
    simulate_heterogeneous_async_session,
)
from src.optimization.onnx_exporter import ONNXEdgeInferenceEngine
from src.optimization.pruning import MultimodalWeightPruner
from src.models.risk_model import MultimodalMentalHealthRiskModel
from src.api.server import app


class TestPhase8Modules(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

    # 1. Contactless Physiological rPPG Tests
    def test_rppg_buffer_and_pos_extraction(self):
        """Verify rPPG extracts pulse waveform and computes HRV metrics."""
        extractor = RemotePPGExtractor(fps=30.0, buffer_size=60)

        # Feed 50 synthetic frames
        dummy_frame = np.full((120, 160, 3), 150, dtype=np.uint8)
        for i in range(50):
            # Modulate green channel with synthetic 1.2 Hz pulse (~72 BPM)
            frame = dummy_frame.copy()
            frame[:, :, 1] = int(150 + 8.0 * np.sin(2 * np.pi * 1.2 * (i / 30.0)))
            r, g, b = extractor.add_frame(frame)
            self.assertGreater(g, 0)

        bvp = extractor.compute_bvp_pos()
        self.assertEqual(len(bvp), 50)

        hrv = extractor.extract_hrv()
        self.assertIsInstance(hrv, HRVMetrics)
        self.assertGreaterEqual(hrv.heart_rate_bpm, 45.0)
        self.assertLessEqual(hrv.heart_rate_bpm, 190.0)
        self.assertGreater(hrv.rmssd_ms, 0.0)
        self.assertGreater(hrv.sdnn_ms, 0.0)
        self.assertIn(hrv.vagal_tone_status, ["Optimal", "Moderate", "Suppressed"])

    def test_rppg_physiological_simulation(self):
        """Verify simulation generates physiologically distinct states."""
        extractor = RemotePPGExtractor()
        low_stress_hrv = extractor.simulate_physiological_sample(target_stress_level="Low")
        high_stress_hrv = extractor.simulate_physiological_sample(target_stress_level="High")

        # Parasympathetic RMSSD must be higher during Low stress than High stress
        self.assertGreater(low_stress_hrv.rmssd_ms, high_stress_hrv.rmssd_ms)
        # Sympathetic Baevsky Stress Index must be higher during High stress
        self.assertGreater(high_stress_hrv.baevsky_stress_index, low_stress_hrv.baevsky_stress_index)
        self.assertEqual(low_stress_hrv.vagal_tone_status, "Optimal")
        self.assertEqual(high_stress_hrv.vagal_tone_status, "Suppressed")

    # 2. Causal Multimodal Counterfactual Recourse Tests
    def test_counterfactual_recourse_generation(self):
        """Verify counterfactual engine computes minimal plausible shifts to reach healthy target."""
        engine = CounterfactualRecourseEngine()
        res = engine.generate_counterfactual(
            current_stress_score=82.0,
            target_stress_score=28.0,
            max_interventions=4,
        )

        self.assertEqual(res.original_stress_score, 82.0)
        self.assertEqual(res.target_stress_score, 28.0)
        self.assertLess(res.achieved_stress_score, 82.0)
        self.assertGreaterEqual(res.sparsity_count, 1)
        self.assertGreaterEqual(res.plausibility_score, 0.50)
        self.assertIsInstance(res.recourse_items, list)
        self.assertGreater(len(res.recourse_items), 0)

        # Check item structure
        first_item = res.recourse_items[0]
        self.assertIn("feature_name", first_item)
        self.assertIn("modality", first_item)
        self.assertIn("percentage_change", first_item)
        self.assertIn("clinical_rationale", first_item)

    def test_counterfactual_recourse_already_low_stress(self):
        """Verify no intervention is prescribed if stress is already below target."""
        engine = CounterfactualRecourseEngine()
        res = engine.generate_counterfactual(
            current_stress_score=26.0,
            target_stress_score=28.0,
        )
        self.assertEqual(res.sparsity_count, 0)
        self.assertEqual(len(res.recourse_items), 0)
        self.assertEqual(res.plausibility_score, 1.0)

    # 3. Asynchronous Federated Learning (FedAsync) Tests
    def test_async_staleness_functions(self):
        """Verify polynomial and exponential staleness decay functions."""
        # Fresh update (tau = 0) has weight 1.0
        self.assertEqual(StalenessFunction.polynomial(0, a=0.5), 1.0)
        self.assertEqual(StalenessFunction.exponential(0, a=0.1), 1.0)

        # Stale update (tau = 4) has discounted weight < 1.0
        w_poly = StalenessFunction.polynomial(4, a=0.5)
        self.assertLess(w_poly, 1.0)
        self.assertGreater(w_poly, 0.0)

    def test_async_fl_server_aggregation(self):
        """Verify AsyncFLServer aggregates client updates non-blockingly."""
        server = AsyncFLServer(base_alpha=0.5, staleness_mode="polynomial", staleness_param=0.5)
        w_init, step0 = server.get_global_weights()
        self.assertEqual(step0, 0)

        # Simulate update from client pulled at step 0 arriving at server step 0 (tau = 0)
        client_w = [w_init[0] + 1.0]
        rec1 = server.update_from_client(client_id="ClientA", client_weights=client_w, pulled_step=0)
        self.assertEqual(rec1["server_step"], 1)
        self.assertEqual(rec1["staleness_tau"], 0)

        # Simulate stale update from client pulled at step 0 arriving at server step 1 (tau = 1)
        rec2 = server.update_from_client(client_id="ClientB", client_weights=client_w, pulled_step=0)
        self.assertEqual(rec2["server_step"], 2)
        self.assertEqual(rec2["staleness_tau"], 1)
        # Stale alpha must be lower than fresh alpha
        self.assertLess(rec2["staleness_alpha"], rec1["staleness_alpha"])

    def test_heterogeneous_async_session_simulation(self):
        """Verify simulate_heterogeneous_async_session yields speedup over synchronous lockstep."""
        sim = simulate_heterogeneous_async_session(num_clients=4, total_events=12)
        self.assertGreater(sim["total_updates"], 0)
        self.assertGreater(sim["wall_clock_speedup"], 1.0)
        self.assertGreater(len(sim["events_log"]), 0)

    # 4. ONNX Runtime Edge Acceleration & Weight Pruning Tests
    def test_magnitude_weight_pruner(self):
        """Verify MultimodalWeightPruner zeroes out low-magnitude weights."""
        model = nn.Sequential(
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 5),
        )
        pruner = MultimodalWeightPruner()
        pruned_model = pruner.apply_magnitude_pruning(model, sparsity=0.40)

        sparsity_info = pruner.compute_model_sparsity(pruned_model)
        self.assertGreater(sparsity_info["zero_parameters"], 0)
        self.assertAlmostEqual(sparsity_info["global_sparsity_pct"], 40.0, delta=5.0)

    def test_pruning_curve_simulation(self):
        """Verify simulate_pruning_curve outputs valid tradeoffs."""
        curve = MultimodalWeightPruner.simulate_pruning_curve(base_mae=22.0)
        self.assertEqual(len(curve), 6)
        self.assertEqual(curve[0]["sparsity_pct"], 0.0)
        self.assertGreater(curve[-1]["payload_reduction_pct"], 60.0)

    def test_onnx_benchmark_comparison(self):
        """Verify ONNX Edge Engine runs benchmark comparison cleanly."""
        model = MultimodalMentalHealthRiskModel()
        engine = ONNXEdgeInferenceEngine()
        bench = engine.benchmark_comparison(model, num_iters=5, warmup=2)

        self.assertIn("pytorch", bench)
        self.assertIn("onnx_runtime", bench)
        self.assertIn("speedup_factor", bench)
        self.assertGreater(bench["speedup_factor"], 1.0)
        self.assertGreater(bench["pytorch"]["mean_latency_ms"], 0.0)

    # 5. FastAPI Phase 8 Endpoints Tests
    def test_fastapi_root_version_and_features(self):
        """Verify API root endpoint advertises version 2.2.0 and Phase 8 capabilities."""
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["version"], "2.2.0")
        feature_text = " ".join(data["features"])
        self.assertIn("rPPG", feature_text)
        self.assertIn("Counterfactual", feature_text)
        self.assertIn("FedAsync", feature_text)
        self.assertIn("ONNX", feature_text)

    def test_fastapi_rppg_extract_endpoint(self):
        """Verify POST /physiological/rppg/extract."""
        resp = self.client.post("/physiological/rppg/extract", json={
            "stress_level_context": "High",
            "simulated": True
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertGreater(data["heart_rate_bpm"], 40.0)
        self.assertGreater(data["rmssd_ms"], 0.0)
        self.assertIn("vagal_tone_status", data)

    def test_fastapi_counterfactual_recourse_endpoint(self):
        """Verify POST /explainability/counterfactual/recourse."""
        resp = self.client.post("/explainability/counterfactual/recourse", json={
            "current_stress_score": 80.0,
            "target_stress_score": 30.0
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["original_stress_score"], 80.0)
        self.assertLess(data["achieved_stress_score"], 80.0)
        self.assertGreater(data["sparsity_count"], 0)
        self.assertIsInstance(data["recourse_items"], list)

    def test_fastapi_async_fl_update_endpoint(self):
        """Verify POST /federated/async/update."""
        resp = self.client.post("/federated/async/update", json={
            "client_id": "EdgeClient_Test",
            "pulled_step": 0,
            "staleness_mode": "polynomial",
            "vector_dim": 30
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("server_step", data)
        self.assertIn("staleness_alpha", data)

    def test_fastapi_onnx_benchmark_endpoint(self):
        """Verify POST /optimization/onnx/benchmark."""
        resp = self.client.post("/optimization/onnx/benchmark", json={"num_iters": 5})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertGreater(data["speedup_factor"], 1.0)
        self.assertGreater(data["pytorch_mean_ms"], 0.0)
        self.assertGreater(data["onnx_runtime_mean_ms"], 0.0)


if __name__ == "__main__":
    unittest.main()
