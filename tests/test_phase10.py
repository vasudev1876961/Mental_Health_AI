"""
Unit Tests for Phase 10 Frontier & Production Platform:
- Clinician-AI Copilot Engine (PHQ-9, GAD-7, DSM-5, SOAP Notes, FHIR JSON, SHA-256 Sign-off)
- Real-Time Crisis Safety Guardrails & Emergency Escalation Protocol
- Federated Domain Generalization (FedDG) with Kernel MMD Invariant Aggregation
- Edge Adaptive Power & Runtime Scheduling
- FastAPI Phase 10 REST API Endpoints
"""

import unittest
import numpy as np
import torch
import torch.nn as nn
from fastapi.testclient import TestClient

from src.clinical.copilot import ClinicalCopilotEngine
from src.clinical.safety_guardrails import CrisisSafetyGuardrails
from src.federated.domain_generalization import (
    DomainInvariantAggregator,
    InvariantFeatureProjector,
    simulate_feddg_cross_domain_adaptation,
)
from src.optimization.adaptive_scheduler import (
    EdgeAdaptiveScheduler,
    simulate_adaptive_edge_profile_run,
)
from src.api.server import app


class TestPhase10Modules(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)
        self.copilot = ClinicalCopilotEngine()
        self.safety = CrisisSafetyGuardrails()
        self.aggregator = DomainInvariantAggregator(gamma=0.5, mmd_weight=2.0)
        self.scheduler = EdgeAdaptiveScheduler()

    # ---------------------------------------------------------
    # 1. Clinician Copilot Tests
    # ---------------------------------------------------------
    def test_phq9_estimation(self):
        """Verify PHQ-9 estimation, bounds, and severity bands."""
        res_low = self.copilot.estimate_phq9_score(stress_score=10.0, fatigue_score=10.0, blunted_affect_score=0.1)
        self.assertLessEqual(res_low["estimated_phq9"], 4)
        self.assertEqual(res_low["severity_band"], "Minimal Depression")

        res_high = self.copilot.estimate_phq9_score(stress_score=90.0, fatigue_score=85.0, blunted_affect_score=0.9)
        self.assertGreaterEqual(res_high["estimated_phq9"], 20)
        self.assertEqual(res_high["severity_band"], "Severe Depression")
        self.assertLessEqual(res_high["estimated_phq9"], 27)

    def test_gad7_estimation(self):
        """Verify GAD-7 estimation, bounds, and severity bands."""
        res_low = self.copilot.estimate_gad7_score(stress_score=15.0, attention_score=90.0, autonomic_arousal=0.1)
        self.assertLessEqual(res_low["estimated_gad7"], 4)
        self.assertEqual(res_low["severity_band"], "Minimal Anxiety")

        res_severe = self.copilot.estimate_gad7_score(stress_score=88.0, attention_score=25.0, autonomic_arousal=0.9)
        self.assertGreaterEqual(res_severe["estimated_gad7"], 15)
        self.assertEqual(res_severe["severity_band"], "Severe Anxiety")
        self.assertLessEqual(res_severe["estimated_gad7"], 21)

    def test_dsm5_criteria_evaluation(self):
        """Verify DSM-5 syndromal criteria breakdown."""
        dsm5_high = self.copilot.evaluate_dsm5_criteria(
            stress_score=82.0,
            fatigue_score=75.0,
            attention_score=35.0,
            hrv_rmssd=16.0,
            hrv_lf_hf=3.2,
            facial_affect={"blunted_affect": 0.6, "blinks_per_min": 35.0, "pitch_sd": 10.0},
        )
        self.assertGreaterEqual(dsm5_high["positive_criteria_count"], 3)
        self.assertIn("affective_blunting", dsm5_high["criteria_breakdown"])
        self.assertTrue(dsm5_high["criteria_breakdown"]["fatigue_energy_loss"]["present"])
        self.assertTrue(dsm5_high["criteria_breakdown"]["cognitive_distractibility"]["present"])

    def test_soap_note_and_fhir_export(self):
        """Verify SOAP note generation, FHIR format, and SHA-256 sign-off."""
        soap = self.copilot.generate_soap_note(
            patient_id="PT-9941",
            stress_score=72.0,
            stress_level="HIGH",
            conformal_interval=(65.0, 80.0),
            fatigue_score=60.0,
            attention_score=40.0,
            hrv_metrics={"mean_hr": 84.0, "rmssd": 21.0, "lf_hf_ratio": 2.4, "snr_db": 15.0},
            transcript_text="I feel constantly stressed and tired.",
        )
        self.assertEqual(soap["patient_id"], "PT-9941")
        self.assertIn("subjective", soap)
        self.assertIn("objective", soap)
        self.assertIn("assessment", soap)
        self.assertIn("plan", soap)

        # FHIR Export
        fhir = self.copilot.export_fhir_diagnostic_report(soap, clinician_id="Dr. Alice Smith, MD")
        self.assertEqual(fhir["resourceType"], "DiagnosticReport")
        self.assertEqual(fhir["subject"]["reference"], "Patient/PT-9941")
        self.assertEqual(len(fhir["contained"]), 2)

        # Clinician Sign-Off Hash
        signoff = self.copilot.sign_off_report(soap, clinician_id="Dr. Alice Smith, MD", clinician_signature="ASMITH-MD-SIG")
        self.assertTrue(signoff["tamper_proof_verified"])
        self.assertEqual(len(signoff["cryptographic_audit_hash"]), 64)

    # ---------------------------------------------------------
    # 2. Crisis Safety Guardrails Tests
    # ---------------------------------------------------------
    def test_crisis_text_scanning(self):
        """Verify text scanning for crisis keywords and neutral speech."""
        res_crisis = self.safety.scan_text_for_crisis("I can't go on anymore, I want to end my life.")
        self.assertTrue(res_crisis["crisis_keywords_detected"])
        self.assertEqual(res_crisis["text_risk_level"], "EMERGENCY_CRISIS")

        res_panic = self.safety.scan_text_for_crisis("I'm having a severe panic attack and my heart racing.")
        self.assertTrue(res_panic["panic_keywords_detected"])
        self.assertEqual(res_panic["text_risk_level"], "ACUTE_PANIC")

        res_routine = self.safety.scan_text_for_crisis("I had lunch and went for a brisk walk.")
        self.assertFalse(res_routine["crisis_keywords_detected"])
        self.assertFalse(res_routine["panic_keywords_detected"])
        self.assertEqual(res_routine["text_risk_level"], "ROUTINE")

    def test_physiology_crisis_scanning(self):
        """Verify autonomic collapse detection via rPPG HRV."""
        res_collapse = self.safety.scan_physiology_for_crisis(
            stress_score=88.0,
            hrv_metrics={"rmssd": 9.5, "lf_hf_ratio": 4.5, "mean_hr": 130.0},
        )
        self.assertTrue(res_collapse["autonomic_crisis"])
        self.assertTrue(res_collapse["rmssd_critical"])
        self.assertTrue(res_collapse["sympathetic_overdrive"])

        res_normative = self.safety.scan_physiology_for_crisis(
            stress_score=35.0,
            hrv_metrics={"rmssd": 45.0, "lf_hf_ratio": 1.2, "mean_hr": 70.0},
        )
        self.assertFalse(res_collapse_normative := res_normative["autonomic_crisis"])

    def test_safety_escalation_tiers(self):
        """Verify safety escalation tier determination and 988 emergency dispatch."""
        eval_tier4 = self.safety.evaluate_safety_escalation(
            stress_score=92.0,
            transcript_text="I want to die, kill myself.",
        )
        self.assertEqual(eval_tier4["escalation_tier"], "EMERGENCY_CRISIS_ESCALATION")
        self.assertTrue(eval_tier4["requires_immediate_dispatch"])
        self.assertIn("crisis_lifeline", eval_tier4["emergency_resources"])

        eval_tier1 = self.safety.evaluate_safety_escalation(
            stress_score=25.0,
            transcript_text="Feeling calm and productive.",
        )
        self.assertEqual(eval_tier1["escalation_tier"], "ROUTINE_MONITORING")
        self.assertFalse(eval_tier1["requires_immediate_dispatch"])

    def test_output_sanitization(self):
        """Verify medical disclaimer is enforced and improper diagnoses replaced."""
        raw_text = "I diagnose you with severe anxiety and you definitely have depression."
        sanitized = self.safety.sanitize_model_output(raw_text)
        self.assertNotIn("I diagnose you with", sanitized)
        self.assertNotIn("you definitely have", sanitized)
        self.assertIn("AI-generated screening indicator", sanitized)

    # ---------------------------------------------------------
    # 3. Federated Domain Generalization (FedDG) Tests
    # ---------------------------------------------------------
    def test_invariant_feature_projector_forward(self):
        """Verify InvariantFeatureProjector produces invariant latent and domain logits."""
        projector = InvariantFeatureProjector(input_dim=32, invariant_dim=16)
        x = torch.randn(8, 32)
        inv_z, domain_logits = projector(x)
        self.assertEqual(inv_z.shape, (8, 16))
        self.assertEqual(domain_logits.shape, (8, 4))

    def test_mmd_computation(self):
        """Verify RBF kernel MMD divergence calculation."""
        rng = np.random.RandomState(42)
        x = rng.normal(0, 1, size=(30, 8)).astype(np.float32)
        y_shifted = rng.normal(3.0, 1, size=(30, 8)).astype(np.float32)

        mmd_self = self.aggregator.compute_mmd(x, x)
        self.assertAlmostEqual(mmd_self, 0.0, places=4)

        mmd_shifted = self.aggregator.compute_mmd(x, y_shifted)
        self.assertGreater(mmd_shifted, 0.1)

    def test_feddg_invariant_aggregation(self):
        """Verify DomainInvariantAggregator weights clients inversely to domain shift."""
        rng = np.random.RandomState(42)
        client_updates = {
            "c1": rng.normal(1.0, 0.1, 10).astype(np.float32),
            "c2": rng.normal(1.0, 0.1, 10).astype(np.float32),
            "c3": rng.normal(1.0, 0.1, 10).astype(np.float32),
        }
        client_features = {
            "c1": rng.normal(0, 1, size=(20, 8)).astype(np.float32),
            "c2": rng.normal(0, 1, size=(20, 8)).astype(np.float32),
            "c3": rng.normal(4, 1, size=(20, 8)).astype(np.float32),  # Heavy domain shift
        }
        client_tags = {"c1": "Hospital_A", "c2": "Hospital_A", "c3": "Outlier_Clinic"}

        res = self.aggregator.aggregate_invariant_updates(client_updates, client_features, client_tags)
        self.assertIn("aggregated_weights", res)
        self.assertEqual(res["aggregated_weights"].shape, (10,))
        # Outlier clinic c3 should receive lower weight than c1 and c2
        weights = res["client_domain_weights"]
        self.assertLess(weights["c3"], weights["c1"])

    def test_simulate_feddg_cross_domain_adaptation(self):
        """Verify cross-domain adaptation simulation produces valid metrics."""
        sim = simulate_feddg_cross_domain_adaptation(num_clients=6, feature_dim=16)
        self.assertEqual(sim["num_clients"], 6)
        self.assertGreater(sim["generalization_gain_pct"], 0.0)
        self.assertLess(sim["ood_mae_feddg"], sim["ood_mae_fedavg"])

    # ---------------------------------------------------------
    # 4. Edge Adaptive Scheduler Tests
    # ---------------------------------------------------------
    def test_adaptive_fidelity_tier_selection(self):
        """Verify dynamic fidelity tier selection based on edge constraints."""
        # Charging -> ULTRA_HIGH
        tier_plugged = self.scheduler.determine_fidelity_tier(battery_pct=50.0, is_charging=True)
        self.assertEqual(tier_plugged, "ULTRA_HIGH_FIDELITY")

        # Low battery -> LOW_POWER_SAVER
        tier_low_bat = self.scheduler.determine_fidelity_tier(battery_pct=10.0, is_charging=False)
        self.assertEqual(tier_low_bat, "LOW_POWER_SAVER")

        # Critical thermal -> LOW_POWER_SAVER
        tier_thermal = self.scheduler.determine_fidelity_tier(battery_pct=80.0, is_charging=False, thermal_state="CRITICAL")
        self.assertEqual(tier_thermal, "LOW_POWER_SAVER")

        # Offline network -> OFFLINE_STORE_AND_FORWARD
        tier_offline = self.scheduler.determine_fidelity_tier(battery_pct=80.0, network_state="OFFLINE")
        self.assertEqual(tier_offline, "OFFLINE_STORE_AND_FORWARD")

    def test_adaptive_scheduler_profile_and_queue(self):
        """Verify schedule profile calculation, offline buffer enqueue, and flush."""
        profile = self.scheduler.schedule_edge_profile(
            battery_pct=30.0,
            is_charging=False,
            thermal_state="NOMINAL",
            network_state="CELLULAR",
        )
        self.assertEqual(profile["selected_tier"], "BALANCED_EDGE")
        self.assertEqual(profile["pipeline_configuration"]["fps_target"], 15)
        self.assertGreater(profile["estimated_battery_hours_remaining"], 0.0)

        # Enqueue offline updates
        dummy_vec = np.ones(20, dtype=np.float32)
        q_len = self.scheduler.enqueue_offline_delta("client_edge_1", dummy_vec, 123456.0)
        self.assertEqual(q_len, 1)

        # Flush
        flushed = self.scheduler.flush_offline_queue()
        self.assertEqual(len(flushed), 1)
        self.assertEqual(len(self.scheduler.offline_queue), 0)

    def test_simulate_adaptive_edge_profile_run(self):
        """Verify adaptive battery depletion simulation."""
        sim = simulate_adaptive_edge_profile_run()
        self.assertIn("simulation_profiles", sim)
        self.assertGreaterEqual(len(sim["simulation_profiles"]), 5)
        self.assertEqual(sim["simulation_profiles"][0]["tier"], "ULTRA_HIGH_FIDELITY")
        self.assertEqual(sim["simulation_profiles"][-1]["tier"], "LOW_POWER_SAVER")

    # ---------------------------------------------------------
    # 5. FastAPI Phase 10 REST Endpoints Tests
    # ---------------------------------------------------------
    def test_api_copilot_soap_report(self):
        """Verify POST /copilot/generate_soap_report endpoint."""
        payload = {
            "patient_id": "PT-7701",
            "stress_score": 76.0,
            "stress_level": "HIGH",
            "conformal_lower": 68.0,
            "conformal_upper": 84.0,
            "fatigue_score": 62.0,
            "attention_score": 38.0,
            "mean_hr": 92.0,
            "rmssd": 17.5,
            "lf_hf_ratio": 3.1,
            "transcript_text": "Experiencing prolonged fatigue and persistent worry.",
            "clinician_notes": "First evaluation session.",
        }
        res = self.client.post("/copilot/generate_soap_report", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["patient_id"], "PT-7701")
        self.assertIn("estimated_phq9", data)
        self.assertIn("estimated_gad7", data)
        self.assertIn("fhir_report_id", data)
        self.assertEqual(len(data["audit_hash"]), 64)

    def test_api_safety_crisis_triage(self):
        """Verify POST /safety/crisis_triage endpoint."""
        payload = {
            "stress_score": 88.0,
            "transcript_text": "I feel so hopeless, want to end my life.",
            "mean_hr": 105.0,
            "rmssd": 11.0,
            "lf_hf_ratio": 4.2,
            "conformal_upper_bound": 94.0,
        }
        res = self.client.post("/safety/crisis_triage", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["escalation_tier"], "EMERGENCY_CRISIS_ESCALATION")
        self.assertTrue(data["requires_immediate_dispatch"])
        self.assertTrue(data["crisis_text_detected"])
        self.assertIn("988", str(data["emergency_resources"]))

    def test_api_domain_generalization(self):
        """Verify POST /federated/domain_generalization endpoint."""
        payload = {"num_clients": 6, "feature_dim": 16}
        res = self.client.post("/federated/domain_generalization", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["num_clients"], 6)
        self.assertGreater(data["generalization_gain_pct"], 0.0)

    def test_api_edge_adaptive_profile(self):
        """Verify POST /edge/adaptive_profile endpoint."""
        payload = {
            "battery_pct": 12.0,
            "is_charging": False,
            "thermal_state": "FAIR",
            "network_state": "CELLULAR",
        }
        res = self.client.post("/edge/adaptive_profile", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["selected_tier"], "LOW_POWER_SAVER")
        self.assertEqual(data["fps_target"], 5)
        self.assertGreater(data["estimated_battery_hours_remaining"], 0.0)


if __name__ == "__main__":
    unittest.main()
