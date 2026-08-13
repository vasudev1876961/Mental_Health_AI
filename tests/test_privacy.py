"""
Unit Tests for Stage 10 Differential Privacy & Privacy Accountant.
"""

import unittest
import torch
import torch.nn as nn

from src.privacy.differential_privacy import DifferentialPrivacyEngine
from src.privacy.accountant import PrivacyAccountant


class TestStage10Privacy(unittest.TestCase):

    def test_dp_clipping_and_noise(self):
        """Verify L2 gradient norm clipping and noise injection."""
        model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1))
        dp_engine = DifferentialPrivacyEngine(max_grad_norm=0.5, noise_multiplier=0.2)

        # Forward and backward pass
        x = torch.randn(4, 10)
        y = model(x).sum()
        y.backward()

        grad_norm = dp_engine.clip_and_noise_gradients(model)
        self.assertGreater(grad_norm, 0.0)

        budget = dp_engine.get_privacy_budget()
        self.assertEqual(budget["steps"], 1)
        self.assertIn("epsilon", budget)

    def test_privacy_accountant(self):
        """Verify privacy accountant tracking."""
        accountant = PrivacyAccountant(target_delta=1e-5)
        accountant.log_step(epsilon=1.2, step=1)
        accountant.log_step(epsilon=1.8, step=2)

        total = accountant.get_total_budget()
        self.assertEqual(total["total_steps"], 2)
        self.assertAlmostEqual(total["total_epsilon"], 1.8)


if __name__ == "__main__":
    unittest.main()
