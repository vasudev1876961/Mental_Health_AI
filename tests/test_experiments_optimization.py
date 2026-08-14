"""
Unit Tests for Model Trainer, Experiment Runner, Quantization, and Latency Benchmark.
"""

import os
import shutil
import tempfile
import unittest
import torch

from src.data.dataset_generator import SyntheticMultimodalDatasetGenerator
from src.data.dataset import MultimodalMentalHealthDataset
from src.models.risk_model import MultimodalMentalHealthRiskModel
from src.models.losses import MultiTaskRiskLoss
from src.training.trainer import ModelTrainer
from src.optimization.quantize import DynamicQuantizer
from src.optimization.benchmark import LatencyBenchmark


class TestExperimentsAndOptimization(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.generator = SyntheticMultimodalDatasetGenerator(seed=42)
        self.generator.generate_dataset(num_subjects=6, output_dir=self.test_dir, num_sequences_per_subject=4)

        ds = MultimodalMentalHealthDataset(self.test_dir)
        self.loader = torch.utils.data.DataLoader(ds, batch_size=4)

        self.model = MultimodalMentalHealthRiskModel()
        self.loss_fn = MultiTaskRiskLoss()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_model_trainer(self):
        """Verify model training epoch and evaluation metrics."""
        trainer = ModelTrainer(self.model, self.loss_fn, checkpoint_dir=self.test_dir)
        train_res = trainer.train_epoch(self.loader)
        self.assertIn("train_loss", train_res)

        eval_res = trainer.evaluate(self.loader)
        self.assertIn("mae", eval_res)
        self.assertIn("rmse", eval_res)
        self.assertIn("f1_score", eval_res)
        self.assertIn("accuracy", eval_res)

    def test_dynamic_quantization(self):
        """Verify model dynamic INT8 quantization."""
        quantizer = DynamicQuantizer()
        q_model = quantizer.quantize(self.model)
        self.assertIsNotNone(q_model)

    def test_latency_benchmark(self):
        """Verify latency (ms) and FPS benchmark calculation."""
        benchmarker = LatencyBenchmark()
        bench = benchmarker.benchmark_model(self.model, num_warmup=1, num_iters=3)
        self.assertIn("mean_latency_ms", bench)
        self.assertIn("fps", bench)
        self.assertGreater(bench["fps"], 0.0)


if __name__ == "__main__":
    unittest.main()
