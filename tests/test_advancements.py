"""
Unit Tests for Phase 5 Advanced Modules:
- Multimodal Contrastive Alignment (InfoNCE Loss)
- Personalized Federated Learning (FedPer)
- Cryptographic Secure Aggregation (SecAgg)
"""

import unittest
import torch
import numpy as np
from collections import OrderedDict

from src.fusion.contrastive import MultimodalContrastiveHead
from src.federated.fedper import FedPerManager
from src.privacy.secure_aggregation import SecureAggregationProtocol


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

    def test_secure_aggregation_zero_sum(self):
        """Verify that pairwise noise masks cancel out to zero during Secure Aggregation."""
        num_clients = 4
        sec_agg = SecureAggregationProtocol(num_clients=num_clients, seed=123)

        # Original unmasked weights (e.g. vector of ones)
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


if __name__ == "__main__":
    unittest.main()
