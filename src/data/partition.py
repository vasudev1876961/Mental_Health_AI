"""
Subject-Level Data Partitioner & Federated Client Splitter.

Enforces zero subject leakage between train/val/test splits and provides
IID and Non-IID subject assignment routines for Federated Learning simulation.
"""

import random
from typing import List, Dict, Tuple


class SubjectPartitioner:
    """Manages leakage-free subject splitting and FL client dataset partitioning."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)

    def train_val_test_split(
        self,
        subject_ids: List[str],
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> Tuple[List[str], List[str], List[str]]:
        """Splits subject IDs into train, validation, and test lists with zero subject overlap."""
        assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Ratios must sum to 1.0"

        shuffled = list(subject_ids)
        random.shuffle(shuffled)

        n_total = len(shuffled)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        train_subjects = shuffled[:n_train]
        val_subjects = shuffled[n_train : n_train + n_val]
        test_subjects = shuffled[n_train + n_val :]

        # Verify zero leakage
        train_set, val_set, test_set = set(train_subjects), set(val_subjects), set(test_subjects)
        assert len(train_set.intersection(val_set)) == 0, "Leakage detected between train and val"
        assert len(train_set.intersection(test_set)) == 0, "Leakage detected between train and test"
        assert len(val_set.intersection(test_set)) == 0, "Leakage detected between val and test"

        return train_subjects, val_subjects, test_subjects

    def partition_for_federated_clients(
        self,
        subject_metadata: List[Dict],
        num_clients: int = 4,
        mode: str = "non_iid",
    ) -> Dict[int, List[str]]:
        """Distributes subjects across K federated clients in IID or Non-IID fashion.

        Args:
            subject_metadata: List of dicts containing 'subject_id' and 'baseline_stress'.
            num_clients: Number of FL clients.
            mode: 'iid' for uniform random distribution, 'non_iid' for label/stress skew distribution.

        Returns:
            Dict mapping client_id (0..num_clients-1) to list of subject_ids assigned to that client.
        """
        client_partitions: Dict[int, List[str]] = {c_idx: [] for c_idx in range(num_clients)}

        if mode == "iid":
            shuffled = list(subject_metadata)
            random.shuffle(shuffled)
            for idx, item in enumerate(shuffled):
                client_id = idx % num_clients
                client_partitions[client_id].append(item["subject_id"])

        elif mode == "non_iid":
            # Sort subjects by baseline stress level to create statistical heterogeneity across clients
            sorted_subjects = sorted(subject_metadata, key=lambda x: x.get("baseline_stress", 50.0))
            chunk_size = max(1, len(sorted_subjects) // num_clients)

            for c_idx in range(num_clients):
                start_idx = c_idx * chunk_size
                if c_idx == num_clients - 1:
                    end_idx = len(sorted_subjects)
                else:
                    end_idx = (c_idx + 1) * chunk_size

                client_subjects = [item["subject_id"] for item in sorted_subjects[start_idx:end_idx]]
                client_partitions[c_idx] = client_subjects

        else:
            raise ValueError(f"Unknown partition mode: '{mode}'. Choose 'iid' or 'non_iid'.")

        return client_partitions


if __name__ == "__main__":
    partitioner = SubjectPartitioner(seed=42)
    subjects = [f"subject_{i:03d}" for i in range(1, 21)]
    train_s, val_s, test_s = partitioner.train_val_test_split(subjects)
    print(f"Train subjects ({len(train_s)}):", train_s)
    print(f"Val subjects ({len(val_s)}):", val_s)
    print(f"Test subjects ({len(test_s)}):", test_s)
