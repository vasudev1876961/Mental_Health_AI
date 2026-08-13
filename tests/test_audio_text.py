"""
Unit Tests for Stage 6 Audio Prosody & Text NLP Processors.
"""

import unittest
import numpy as np
import torch

from src.audio.features import AudioFeatureExtractor
from src.text.encoder import TextNLPEncoder


class TestStage6AudioText(unittest.TestCase):

    def test_audio_feature_extractor(self):
        """Verify audio prosody and MFCC extraction shape."""
        extractor = AudioFeatureExtractor()
        # Synthetic audio signal: 1 second at 16kHz
        t = np.linspace(0, 1.0, 16000, dtype=np.float32)
        audio_signal = np.sin(2 * np.pi * 440 * t) + 0.1 * np.random.randn(16000)

        feat = extractor.extract_features(audio_signal, sr=16000)
        self.assertEqual(feat.shape, (16,))
        # Pitch F0 should be positive
        self.assertGreater(feat[0], 0.0)

    def test_text_nlp_encoder(self):
        """Verify text NLP embedding projection shape."""
        encoder = TextNLPEncoder(input_dim=128, output_dim=128)
        dummy_text_emb = torch.randn(4, 128)

        out = encoder(dummy_text_emb)
        self.assertEqual(out.shape, (4, 128))


if __name__ == "__main__":
    unittest.main()
