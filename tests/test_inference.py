"""
Unit Tests for Stage 12 Real-Time Inference Manager & Buffer.
"""

import unittest
import numpy as np
import torch

from src.inference.buffer import SlidingWindowBuffer
from src.inference.realtime import RealtimeInferenceEngine


class TestStage12InferenceEngine(unittest.TestCase):

    def test_sliding_window_buffer(self):
        """Verify queue buffer size and sequence tensor conversion."""
        buffer = SlidingWindowBuffer(window_size=30, feature_dim=18)
        self.assertTrue(buffer.is_full())

        tensor = buffer.get_sequence_tensor()
        self.assertEqual(tensor.shape, (1, 30, 18))

        # Add 5 new frames
        for _ in range(5):
            buffer.add_frame(np.random.randn(18).astype(np.float32))

        tensor_after = buffer.get_sequence_tensor()
        self.assertEqual(tensor_after.shape, (1, 30, 18))

    def test_realtime_inference_engine(self):
        """Verify process_frame payload output."""
        engine = RealtimeInferenceEngine(window_size=30)
        dummy_frame = np.full((480, 640, 3), 150, dtype=np.uint8)

        payload = engine.process_frame(image=dummy_frame)

        self.assertIn("stress_score", payload)
        self.assertIn("stress_level", payload)
        self.assertIn("fatigue_score", payload)
        self.assertIn("attention_score", payload)
        self.assertIn("shap_ranks", payload)
        self.assertIn("modality_pcts", payload)
        self.assertIn("quality", payload)

        self.assertGreaterEqual(payload["stress_score"], 0.0)
        self.assertLessEqual(payload["stress_score"], 100.0)


if __name__ == "__main__":
    unittest.main()
