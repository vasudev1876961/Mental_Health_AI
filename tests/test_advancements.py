"""
Unit Tests for Phase 6 Advanced Modules:
- Multimodal Contrastive Alignment (InfoNCE Loss & Model Integration)
- Personalized Federated Learning (FedPer Client & Parameter Preservation)
- Cryptographic Secure Aggregation (SecAgg for Arbitrary Client Sizes)
- API Advanced Endpoints (SecAgg and Contrastive Verification)
"""

import unittest
import torch
import numpy as np
from collections import OrderedDict
from fastapi.testclient import TestClient

from src.fusion.contrastive import MultimodalContrastiveHead
from src.models.risk_model import MultimodalMentalHealthRiskModel
from src.models.losses import MultiTaskRiskLoss
from src.training.trainer import ModelTrainer
from src.federated.fedper import FedPerManager
from src.federated.client import MentalHealthFlowerClient
from src.privacy.secure_aggregation import SecureAggregationProtocol
from src.api.server import app


class TestAdvancedModules(unittest.TestCase):

    def test_contrastive_infonce_loss(self):
        """Verify 3-way multimodal InfoNCE contrastive loss computation."""
        head = MultimodalContrastiveHead(in_dim=128, proj_dim=64)
        v = torch.randn(8, 128)
        a = torch.randn(8, 128)
        t = torch.randn(8, 128)

        loss = head.compute_multimodal_loss(v, a, t)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertGreater(loss.item(), 0.0)

    def test_model_contrastive_integration(self):
        """Verify MultimodalMentalHealthRiskModel returns contrastive representations and loss."""
        model = MultimodalMentalHealthRiskModel()
        v_seq = torch.randn(4, 30, 18)
        a_feat = torch.randn(4, 16)
        t_feat = torch.randn(4, 128)

        # Modality embedding extraction
        v_emb, a_emb, t_emb = model.extract_modality_embeddings(v_seq, a_feat, t_feat)
        self.assertEqual(v_emb.shape, (4, 128))
        self.assertEqual(a_emb.shape, (4, 128))
        self.assertEqual(t_emb.shape, (4, 128))

        # Forward with return_contrastive=True
        preds, modality_weights, confidence = model(v_seq, a_feat, t_feat, return_contrastive=True)
        self.assertIn("contrastive_loss", preds)
        self.assertGreater(preds["contrastive_loss"].item(), 0.0)

        # MultiTaskRiskLoss with contrastive loss
        loss_fn = MultiTaskRiskLoss(weight_contrastive=0.1)
        targets = {
            "stress_score": torch.tensor([25.0, 45.0, 65.0, 80.0]),
            "stress_class": torch.tensor([0, 1, 2, 2]),
            "fatigue": torch.tensor([0.2, 0.4, 0.6, 0.8]),
            "attention": torch.tensor([0.8, 0.6, 0.4, 0.2]),
        }
        total_loss, loss_dict = loss_fn(preds, targets)
        self.assertIn("loss_contrastive", loss_dict)
        self.assertGreater(loss_dict["total_loss"], 0.0)

    def test_fedper_parameter_splitting(self):
        """Verify FedPer global backbone extraction and local personal head merging."""
        manager = FedPerManager(personal_layer_prefixes=["risk_head"])

        state_dict = OrderedDict([
            ("encoder.weight", torch.ones(4, 4)),
            ("risk_head.weight", torch.zeros(2, 2)),
        ])

        global_dict = manager.extract_global_parameters(state_dict)
        self.assertIn("encoder.weight", global_dict)
        self.assertNotIn("risk_head.weight", global_dict)

        merged = manager.merge_global_and_local(global_dict, state_dict)
        self.assertIn("risk_head.weight", merged)

    def test_fedper_flower_client(self):
        """Verify MentalHealthFlowerClient retains personal head parameters across set_parameters."""
        model = MultimodalMentalHealthRiskModel()
        personal_prefixes = ["heads.stress_reg_head", "heads.stress_cls_head"]
        client = MentalHealthFlowerClient(
            client_id=1,
            model=model,
            train_loader=None,
            val_loader=None,
            loss_fn=MultiTaskRiskLoss(),
            is_fedper=True,
            personal_layer_prefixes=personal_prefixes,
        )

        # Modify personal head weights locally
        with torch.no_grad():
            model.heads.stress_reg_head[0].weight.fill_(99.0)

        # Client extracts global parameters (must not contain heads.stress_reg_head)
        global_params = client.get_parameters()
        self.assertTrue(len(global_params) > 0)

        # Simulate receiving new global parameters from server
        new_global_params = [p + 0.1 for p in global_params]
        client.set_parameters(new_global_params)

        # Personal head weights must be preserved!
        self.assertTrue(torch.allclose(model.heads.stress_reg_head[0].weight, torch.tensor(99.0)))

    def test_secure_aggregation_zero_sum(self):
        """Verify that pairwise noise masks cancel out to zero during Secure Aggregation."""
        for num_clients in [3, 4, 6]:
            sec_agg = SecureAggregationProtocol(num_clients=num_clients, seed=123)

            # Original unmasked weights
            raw_weights = [np.ones((100,), dtype=np.float32) * float(i + 1) for i in range(num_clients)]

            # Mask weights locally on each client
            masked_weights = [sec_agg.mask_client_weights(i, raw_weights[i]) for i in range(num_clients)]

            # Verify individual masked vector is NOT equal to raw vector
            self.assertFalse(np.allclose(masked_weights[0], raw_weights[0]))

            # Aggregate at central server
            sec_result = sec_agg.aggregate_masked_updates(masked_weights)

            # Server-side unmasked average
            raw_avg = np.mean(raw_weights, axis=0)

            # Secure aggregation result must EXACTLY match the raw average!
            np.testing.assert_allclose(sec_result, raw_avg, atol=1e-5)

    def test_api_secagg_endpoint(self):
        """Verify /privacy/secagg/verify API endpoint."""
        client = TestClient(app)
        response = client.post("/privacy/secagg/verify", json={"num_clients": 4, "vector_dim": 20})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["exact_match"])
        self.assertLess(data["cancellation_residual"], 1e-4)

    def test_api_contrastive_endpoint(self):
        """Verify /fusion/contrastive/similarity API endpoint."""
        client = TestClient(app)
        response = client.post("/fusion/contrastive/similarity", json={"vision_dim": 128})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("vision_audio_similarity", data)
        self.assertIn("infonce_alignment_loss", data)


if __name__ == "__main__":
    unittest.main()
