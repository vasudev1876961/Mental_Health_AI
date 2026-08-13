"""
Unit Tests for Stage 3 & 5 Temporal Sequence Encoders.
"""

import unittest
import torch

from src.temporal.lstm import BiLSTMSequenceEncoder
from src.temporal.gru import GRUSequenceEncoder
from src.temporal.transformer import TemporalTransformerEncoder


class TestStage3TemporalEncoders(unittest.TestCase):

    def setUp(self):
        self.batch_size = 4
        self.seq_len = 30
        self.input_dim = 18
        self.hidden_dim = 128

        self.dummy_seq = torch.randn(self.batch_size, self.seq_len, self.input_dim)

    def test_bilstm_encoder(self):
        """Verify BiLSTM sequence encoding shape."""
        encoder = BiLSTMSequenceEncoder(input_dim=self.input_dim, hidden_dim=self.hidden_dim)
        out = encoder(self.dummy_seq)
        self.assertEqual(out.shape, (self.batch_size, self.hidden_dim))

    def test_gru_encoder(self):
        """Verify GRU sequence encoding shape."""
        encoder = GRUSequenceEncoder(input_dim=self.input_dim, hidden_dim=self.hidden_dim)
        out = encoder(self.dummy_seq)
        self.assertEqual(out.shape, (self.batch_size, self.hidden_dim))

    def test_transformer_encoder(self):
        """Verify Temporal Transformer encoding and attention output shapes."""
        encoder = TemporalTransformerEncoder(input_dim=self.input_dim, hidden_dim=self.hidden_dim)
        out, tf_features = encoder(self.dummy_seq)

        self.assertEqual(out.shape, (self.batch_size, self.hidden_dim))
        self.assertEqual(tf_features.shape, (self.batch_size, self.seq_len, self.hidden_dim))


if __name__ == "__main__":
    unittest.main()
