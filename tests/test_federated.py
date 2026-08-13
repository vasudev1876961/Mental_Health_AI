"""
Unit Tests for Stage 9 Federated Learning Engine (Flower Client, FedAvg/FedProx, Metrics).
"""

import os
import shutil
import tempfile
import unittest
import torch

from src.data.dataset_generator import SyntheticMultimodalDatasetGenerator
from src.data.partition import SubjectPartitioner
from src.data.dataset import MultimodalMentalHealthDataset, create_dataloaders
from src.models.risk_model import MultimodalMentalHealthRiskModel
from src.models.losses import MultiTaskRiskLoss
from src.federated.client import MentalHealthFlowerClient
from src.federated.fedprox import aggregate_weights, FedProxStrategy
from src.federated.metrics import FLMetricsTracker


class TestStage9FederatedEngine(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.generator = SyntheticMultimodalDatasetGenerator(seed=42)
        self.generator.generate_dataset(num_subjects=8, output_dir=self.test_dir, num_sequences_per_subject=4)

        self.partitioner = SubjectPartitioner(seed=42)
        subject_ids = [f"subject_{i:03d}" for i in range(1, 9)]
        self.client_partitions = self.partitioner.partition_for_federated_clients(
            [{"subject_id": s, "baseline_stress": idx * 10} for idx, s in enumerate(subject_ids)],
            num_clients=2,
            mode="non_iid",
        )

        self.model = MultimodalMentalHealthRiskModel()
        self.loss_fn = MultiTaskRiskLoss()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_flower_client_fit_and_evaluate(self):
        """Verify client local training update and parameter extraction."""
        client_0_subjs = self.client_partitions[0]
        ds_train = MultimodalMentalHealthDataset(self.test_dir, subject_ids=client_0_subjs)
        loader_train = torch.utils.data.DataLoader(ds_train, batch_size=4)

        client = MentalHealthFlowerClient(
            client_id=0,
            model=self.model,
            train_loader=loader_train,
            val_loader=loader_train,
            loss_fn=self.loss_fn,
            fedprox_mu=0.01,
        )

        params = client.get_parameters()
        self.assertGreater(len(params), 0)

        # Execute local fit round
        new_params, num_samples, metrics = client.fit(params, config={"local_epochs": 1})
        self.assertEqual(len(new_params), len(params))
        self.assertGreater(num_samples, 0)
        self.assertIn("train_loss", metrics)

        # Execute evaluation round
        loss, val_samples, eval_metrics = client.evaluate(new_params, config={})
        self.assertGreater(val_samples, 0)
        self.assertIn("mae", eval_metrics)

    def test_federated_weight_aggregation(self):
        """Verify FedAvg / FedProx weighted parameter aggregation."""
        client_1_weights = client_2_weights = [param.detach().cpu().numpy() for param in self.model.state_dict().values()]
        client_results = [(client_1_weights, 10), (client_2_weights, 30)]

        aggregated = aggregate_weights(client_results)
        self.assertEqual(len(aggregated), len(client_1_weights))
        self.assertEqual(aggregated[0].shape, client_1_weights[0].shape)

    def test_fl_metrics_tracker(self):
        """Verify metrics tracker history."""
        tracker = FLMetricsTracker()
        tracker.record_round(round_num=1, loss=12.5, mae=4.2)
        tracker.record_round(round_num=2, loss=10.1, mae=3.8)

        summary = tracker.get_summary()
        self.assertEqual(summary["total_rounds"], 2)
        self.assertAlmostEqual(summary["final_mae"], 3.8)


if __name__ == "__main__":
    unittest.main()
