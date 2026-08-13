"""
Federated Learning Client Module.

Implements local edge training without sending raw video/audio data off device.
"""

from collections import OrderedDict
import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple

try:
    import flwr as fl
    NumPyClientBase = fl.client.NumPyClient
except Exception:
    NumPyClientBase = object


class MentalHealthFlowerClient(NumPyClientBase):
    """Flower Federated Learning Client for local privacy-preserving model updates."""

    def __init__(
        self,
        client_id: int,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: torch.nn.Module,
        learning_rate: float = 0.001,
        fedprox_mu: float = 0.0,
        device: torch.device = torch.device("cpu"),
    ):
        self.client_id = client_id
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.lr = learning_rate
        self.fedprox_mu = fedprox_mu
        self.device = device

    def get_parameters(self, config: Dict[str, str] = None) -> List[np.ndarray]:
        """Extracts model weights as numpy arrays."""
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters: List[np.ndarray]):
        """Sets model weights from numpy array parameter list."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v).to(self.device) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[List[np.ndarray], int, Dict[str, float]]:
        """Trains local model weights for local_epochs."""
        self.set_parameters(parameters)
        self.model.train()

        epochs = int(config.get("local_epochs", 2))
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)

        # Store initial global parameters for FedProx penalty
        if self.fedprox_mu > 0.0:
            global_params = [p.clone().detach() for p in self.model.parameters()]

        total_samples = 0
        total_loss_accum = 0.0

        for epoch in range(epochs):
            for batch in self.train_loader:
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

                optimizer.zero_grad()
                preds, _, _ = self.model(v_seq, a_feat, t_feat, mask=mask)
                loss, loss_dict = self.loss_fn(preds, targets)

                # Add FedProx proximal term penalty if mu > 0
                if self.fedprox_mu > 0.0:
                    proximal_term = 0.0
                    for p, g_p in zip(self.model.parameters(), global_params):
                        proximal_term += torch.sum((p - g_p) ** 2)
                    loss += (self.fedprox_mu / 2.0) * proximal_term

                loss.backward()
                optimizer.step()

                batch_size = v_seq.size(0)
                total_samples += batch_size
                total_loss_accum += loss.item() * batch_size

        avg_loss = total_loss_accum / max(1, total_samples)
        return self.get_parameters(), total_samples, {"train_loss": float(avg_loss)}

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[float, int, Dict[str, float]]:
        """Evaluates model performance on client validation set."""
        self.set_parameters(parameters)
        self.model.eval()

        total_samples = 0
        total_loss = 0.0
        mae_accum = 0.0

        with torch.no_grad():
            for batch in self.val_loader:
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
                total_samples += batch_size
                total_loss += loss.item() * batch_size

                mae = torch.abs(preds["stress_score"] - targets["stress_score"]).sum().item()
                mae_accum += mae

        avg_loss = total_loss / max(1, total_samples)
        avg_mae = mae_accum / max(1, total_samples)

        return float(avg_loss), total_samples, {"eval_loss": float(avg_loss), "mae": float(avg_mae)}
