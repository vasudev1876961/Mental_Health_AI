"""
PyTorch Multimodal Dataset and DataLoader Utilities.

Loads sequence data for specified subject subsets, handles PyTorch tensor conversion,
and supports missing-modality masking vectors for robustness evaluation.
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional, Tuple


class MultimodalMentalHealthDataset(Dataset):
    """PyTorch Dataset for Multimodal Mental Health Risk Assessment."""

    LABEL_TO_CLASS = {"Low": 0, "Medium": 1, "High": 2}

    def __init__(
        self,
        data_dir: str,
        subject_ids: Optional[List[str]] = None,
        missing_modality_prob: float = 0.0,
    ):
        """
        Args:
            data_dir: Path to directory containing subject JSON files.
            subject_ids: Optional list of allowed subject IDs to include. If None, uses all found files.
            missing_modality_prob: Probability of simulating missing modalities during training/eval.
        """
        self.data_dir = data_dir
        self.missing_modality_prob = missing_modality_prob
        self.samples = []

        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory '{data_dir}' does not exist.")

        # Find subject JSON files
        json_files = [f for f in os.listdir(data_dir) if f.endswith(".json") and f != "dataset_metadata.json"]
        
        for file_name in json_files:
            subj_id = file_name.replace(".json", "")
            if subject_ids is not None and subj_id not in subject_ids:
                continue

            file_path = os.path.join(data_dir, file_name)
            with open(file_path, "r", encoding="utf-8") as f:
                subj_data = json.load(f)

            for seq in subj_data.get("sequences", []):
                self.samples.append(seq)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        vision_tensor = torch.tensor(sample["vision"], dtype=torch.float32) # [window_size, vision_dim]
        audio_tensor = torch.tensor(sample["audio"], dtype=torch.float32)   # [audio_dim]
        text_tensor = torch.tensor(sample["text"], dtype=torch.float32)     # [text_dim]

        # Missing Modality Mask: [vision_present, audio_present, text_present]
        mask = torch.ones(3, dtype=torch.float32)

        if self.missing_modality_prob > 0.0:
            if np.random.rand() < self.missing_modality_prob:
                # Randomly drop one modality
                drop_idx = np.random.randint(0, 3)
                mask[drop_idx] = 0.0
                if drop_idx == 0:
                    vision_tensor = torch.zeros_like(vision_tensor)
                elif drop_idx == 1:
                    audio_tensor = torch.zeros_like(audio_tensor)
                else:
                    text_tensor = torch.zeros_like(text_tensor)

        targets = sample["targets"]
        stress_score = torch.tensor(targets["stress_score"], dtype=torch.float32)
        stress_class = torch.tensor(self.LABEL_TO_CLASS.get(targets["stress_label"], 1), dtype=torch.long)
        fatigue = torch.tensor(targets.get("fatigue", 0.5), dtype=torch.float32)
        attention = torch.tensor(targets.get("attention", 0.5), dtype=torch.float32)

        return {
            "sequence_id": sample["sequence_id"],
            "vision": vision_tensor,
            "audio": audio_tensor,
            "text": text_tensor,
            "mask": mask,
            "stress_score": stress_score,
            "stress_class": stress_class,
            "fatigue": fatigue,
            "attention": attention,
        }


def create_dataloaders(
    data_dir: str,
    train_subjects: List[str],
    val_subjects: List[str],
    test_subjects: List[str],
    batch_size: int = 16,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Creates train, validation, and test PyTorch DataLoaders enforcing subject splits."""

    train_ds = MultimodalMentalHealthDataset(data_dir, subject_ids=train_subjects, missing_modality_prob=0.1)
    val_ds = MultimodalMentalHealthDataset(data_dir, subject_ids=val_subjects, missing_modality_prob=0.0)
    test_ds = MultimodalMentalHealthDataset(data_dir, subject_ids=test_subjects, missing_modality_prob=0.0)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
