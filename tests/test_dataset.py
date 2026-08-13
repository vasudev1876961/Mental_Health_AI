"""
Unit Tests for Stage 1 Data Pipeline, Generator, Partitioner, and DataLoaders.
"""

import os
import shutil
import tempfile
import unittest
import torch

from src.data.dataset_generator import SyntheticMultimodalDatasetGenerator
from src.data.partition import SubjectPartitioner
from src.data.dataset import MultimodalMentalHealthDataset, create_dataloaders


class TestStage1DataPipeline(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.num_subjects = 12
        self.generator = SyntheticMultimodalDatasetGenerator(seed=42)
        self.subject_files = self.generator.generate_dataset(
            num_subjects=self.num_subjects,
            output_dir=self.test_dir,
            num_sequences_per_subject=5,
            window_size=30,
        )

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_synthetic_data_generation(self):
        """Verify file generation and metadata structure."""
        self.assertEqual(len(self.subject_files), self.num_subjects)
        metadata_file = os.path.join(self.test_dir, "dataset_metadata.json")
        self.assertTrue(os.path.exists(metadata_file))

    def test_subject_split_zero_leakage(self):
        """Verify train/val/test splits have zero subject overlap."""
        partitioner = SubjectPartitioner(seed=42)
        subject_ids = [f"subject_{i:03d}" for i in range(1, self.num_subjects + 1)]

        train_s, val_s, test_s = partitioner.train_val_test_split(
            subject_ids, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2
        )

        train_set, val_set, test_set = set(train_s), set(val_s), set(test_s)
        self.assertEqual(len(train_set.intersection(val_set)), 0)
        self.assertEqual(len(train_set.intersection(test_set)), 0)
        self.assertEqual(len(val_set.intersection(test_set)), 0)

        # Check total count matches
        self.assertEqual(len(train_s) + len(val_s) + len(test_s), self.num_subjects)

    def test_fl_client_partitioning(self):
        """Verify IID and Non-IID partitioning for FL clients."""
        partitioner = SubjectPartitioner(seed=42)
        subject_meta = [{"subject_id": f"subject_{i:03d}", "baseline_stress": i * 5.0} for i in range(1, 13)]

        # Test IID mode
        iid_parts = partitioner.partition_for_federated_clients(subject_meta, num_clients=3, mode="iid")
        self.assertEqual(len(iid_parts), 3)

        # Test Non-IID mode
        non_iid_parts = partitioner.partition_for_federated_clients(subject_meta, num_clients=3, mode="non_iid")
        self.assertEqual(len(non_iid_parts), 3)
        # Client 0 should have lowest stress subjects, Client 2 should have highest
        client_0_subjs = non_iid_parts[0]
        client_2_subjs = non_iid_parts[2]
        self.assertNotEqual(client_0_subjs, client_2_subjs)

    def test_pytorch_dataset_and_dataloaders(self):
        """Verify PyTorch dataset and dataloader batch shapes."""
        partitioner = SubjectPartitioner(seed=42)
        subject_ids = [f"subject_{i:03d}" for i in range(1, self.num_subjects + 1)]
        train_s, val_s, test_s = partitioner.train_val_test_split(subject_ids)

        train_loader, val_loader, test_loader = create_dataloaders(
            data_dir=self.test_dir,
            train_subjects=train_s,
            val_subjects=val_s,
            test_subjects=test_s,
            batch_size=4,
        )

        batch = next(iter(train_loader))

        # Check Tensor shapes
        self.assertEqual(batch["vision"].shape, (4, 30, 18))  # [B, window_size, vision_dim]
        self.assertEqual(batch["audio"].shape, (4, 16))      # [B, audio_dim]
        self.assertEqual(batch["text"].shape, (4, 128))      # [B, text_dim]
        self.assertEqual(batch["mask"].shape, (4, 3))        # [B, 3] (vision, audio, text presence)
        self.assertEqual(batch["stress_score"].shape, (4,))   # [B]
        self.assertEqual(batch["stress_class"].shape, (4,))   # [B]

    def test_missing_modality_masking(self):
        """Verify that missing_modality_prob zeroes out tensors and updates mask."""
        ds = MultimodalMentalHealthDataset(self.test_dir, missing_modality_prob=1.0)
        sample = ds[0]
        mask = sample["mask"]
        # Exactly one modality should be zeroed out
        self.assertEqual(float(mask.sum().item()), 2.0)


if __name__ == "__main__":
    unittest.main()
