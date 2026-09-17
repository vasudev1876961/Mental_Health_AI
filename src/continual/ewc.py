"""
Elastic Weight Consolidation (EWC) for Continual Edge Adaptation.

Prevents catastrophic forgetting of historical personal baseline emotional and
physiological patterns when the model continually adapts to shifting acute stress states.

Mathematical Formulation:
    L(theta) = L_new(theta) + (lambda / 2) * sum_i F_i * (theta_i - theta_i^*)^2
where F_i is the diagonal element of the empirical Fisher Information Matrix.
"""

from typing import Dict, Optional, List
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class ElasticWeightConsolidation:
    """Manages empirical Fisher Information and computes EWC quadratic penalty."""

    def __init__(self, model: nn.Module, ewc_lambda: float = 400.0, device: Optional[torch.device] = None):
        self.model = model
        self.ewc_lambda = float(ewc_lambda)
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.optimal_params: Dict[str, torch.Tensor] = {}
        self.fisher_matrix: Dict[str, torch.Tensor] = {}
        self.has_prior_task: bool = False

    def register_task(
        self,
        dataloader: DataLoader,
        loss_fn: nn.Module,
        max_batches: int = 20,
    ) -> Dict[str, float]:
        """Estimates diagonal Fisher Information matrix and saves current model parameters as theta*."""
        self.model.eval()

        # 1. Snapshot optimal parameters theta*
        self.optimal_params = {
            name: param.clone().detach().to(self.device)
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        # 2. Initialize Fisher accumulators to zero
        fisher = {
            name: torch.zeros_like(param, device=self.device)
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        total_samples = 0
        batch_count = 0

        for batch in dataloader:
            if batch_count >= max_batches:
                break

            self.model.zero_grad()

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
            loss, _ = loss_fn(preds, targets)
            loss.backward()

            batch_size = v_seq.size(0)
            total_samples += batch_size
            batch_count += 1

            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    # Accumulate squared empirical gradients
                    fisher[name] += (param.grad.detach() ** 2) * batch_size

        if total_samples > 0:
            for name in fisher:
                fisher[name] /= total_samples

        self.fisher_matrix = fisher
        self.has_prior_task = True

        # Calculate average Fisher magnitude across all parameters
        avg_fisher = float(torch.mean(torch.stack([torch.mean(f) for f in fisher.values()])).item())
        return {
            "total_samples_analyzed": total_samples,
            "mean_fisher_magnitude": round(avg_fisher, 6),
            "regularized_param_count": len(self.optimal_params),
        }

    def penalty(self) -> torch.Tensor:
        """Computes quadratic EWC penalty loss: (lambda / 2) * sum_i F_i (theta_i - theta_i^*)^2."""
        if not self.has_prior_task:
            return torch.tensor(0.0, device=self.device)

        loss = torch.tensor(0.0, device=self.device)
        for name, param in self.model.named_parameters():
            if name in self.fisher_matrix and name in self.optimal_params:
                f_i = self.fisher_matrix[name]
                theta_star = self.optimal_params[name]
                diff_sq = (param - theta_star) ** 2
                loss += torch.sum(f_i * diff_sq)

        return (self.ewc_lambda / 2.0) * loss

    def compute_penalty_value(self) -> float:
        """Returns the scalar EWC penalty value without gradient tracking."""
        with torch.no_grad():
            pen = self.penalty()
            return float(pen.item())
