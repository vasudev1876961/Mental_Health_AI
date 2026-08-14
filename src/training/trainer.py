"""
PyTorch Model Trainer and Checkpoint Manager.

Handles model training, validation, multi-task metric evaluation (MAE, RMSE, R2, F1, Accuracy),
and checkpoint saving.
"""

import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Optional
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, mean_absolute_error, mean_squared_error


class ModelTrainer:
    """Trainer for Multimodal Mental Health Risk Models."""

    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        learning_rate: float = 0.001,
        device: torch.device = torch.device("cpu"),
        checkpoint_dir: str = "checkpoints",
    ):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        if optimizer is None:
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        else:
            self.optimizer = optimizer

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Runs a single training epoch."""
        self.model.train()
        total_loss = 0.0
        total_samples = 0

        for batch in train_loader:
            v_seq = batch["vision"].to(self.device)
            a_feat = batch["audio"].to(self.device)
            t_feat = batch["text"].to(self.device)
            mask = batch["mask"].to(self.device)

            targets = {
                "stress_score": batch["stress_score"].to(self.device),
                "stress_class": batch["stress_class"].to(self.device),
                "fatigue": batch["fatigue"].to(self.device),
                "attention": batch["attention"].to(self.device),
            }

            self.optimizer.zero_grad()
            preds, _, _ = self.model(v_seq, a_feat, t_feat, mask=mask)
            loss, _ = self.loss_fn(preds, targets)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            batch_size = v_seq.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

        return {"train_loss": float(total_loss / max(1, total_samples))}

    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Evaluates model performance on validation or test DataLoader."""
        self.model.eval()
        total_loss = 0.0
        total_samples = 0

        all_stress_true = []
        all_stress_pred = []
        all_cls_true = []
        all_cls_pred = []

        with torch.no_grad():
            for batch in val_loader:
                v_seq = batch["vision"].to(self.device)
                a_feat = batch["audio"].to(self.device)
                t_feat = batch["text"].to(self.device)
                mask = batch["mask"].to(self.device)

                targets = {
                    "stress_score": batch["stress_score"].to(self.device),
                    "stress_class": batch["stress_class"].to(self.device),
                    "fatigue": batch["fatigue"].to(self.device),
                    "attention": batch["attention"].to(self.device),
                }

                preds, _, _ = self.model(v_seq, a_feat, t_feat, mask=mask)
                loss, _ = self.loss_fn(preds, targets)

                batch_size = v_seq.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

                all_stress_true.extend(targets["stress_score"].cpu().numpy())
                all_stress_pred.extend(preds["stress_score"].cpu().numpy())
                all_cls_true.extend(targets["stress_class"].cpu().numpy())
                all_cls_pred.extend(torch.argmax(preds["stress_logits"], dim=1).cpu().numpy())

        all_stress_true = np.array(all_stress_true)
        all_stress_pred = np.array(all_stress_pred)
        all_cls_true = np.array(all_cls_true)
        all_cls_pred = np.array(all_cls_pred)

        # Calculate Regression Metrics
        mae = float(mean_absolute_error(all_stress_true, all_stress_pred))
        rmse = float(np.sqrt(mean_squared_error(all_stress_true, all_stress_pred)))
        
        if len(all_stress_true) > 1 and np.std(all_stress_true) > 0 and np.std(all_stress_pred) > 0:
            pearson_r, _ = pearsonr(all_stress_true, all_stress_pred)
            r_val = float(pearson_r)
        else:
            r_val = 0.0

        # Calculate Classification Metrics
        acc = float(accuracy_score(all_cls_true, all_cls_pred))
        f1 = float(f1_score(all_cls_true, all_cls_pred, average="macro", zero_division=0))

        return {
            "val_loss": float(total_loss / max(1, total_samples)),
            "mae": mae,
            "rmse": rmse,
            "pearson_r": r_val,
            "accuracy": acc,
            "f1_score": f1,
        }

    def train_full(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 5,
        save_name: str = "best_model.pt",
    ) -> Dict[str, float]:
        """Trains model for multiple epochs with best-checkpoint saving."""
        best_mae = float("inf")
        best_metrics = {}

        for epoch in range(1, epochs + 1):
            train_res = self.train_epoch(train_loader)
            eval_res = self.evaluate(val_loader)

            if eval_res["mae"] < best_mae:
                best_mae = eval_res["mae"]
                best_metrics = eval_res.copy()
                save_path = os.path.join(self.checkpoint_dir, save_name)
                torch.save(self.model.state_dict(), save_path)

        return best_metrics
