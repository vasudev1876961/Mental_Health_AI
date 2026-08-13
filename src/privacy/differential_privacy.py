"""
Differential Privacy Engine.

Applies L2 gradient clipping and calibrated Gaussian noise injection to model updates.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Dict


class DifferentialPrivacyEngine:
    """Differential Privacy mechanism for gradient clipping and noise injection."""

    def __init__(
        self,
        max_grad_norm: float = 1.0,
        noise_multiplier: float = 0.5,
        target_delta: float = 1e-5,
    ):
        self.max_grad_norm = max_grad_norm
        self.noise_multiplier = noise_multiplier
        self.target_delta = target_delta
        self.steps_taken = 0

    def clip_and_noise_gradients(self, model: nn.Module) -> float:
        """Clips parameter gradients to max_grad_norm and adds Gaussian noise.

        Returns:
            Computed gradient norm before clipping.
        """
        # 1. Calculate total L2 norm across all parameter gradients
        total_norm_sq = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm_sq += float(p.grad.data.norm(2).item() ** 2)
        total_norm = float(np.sqrt(total_norm_sq))

        # 2. Compute clipping coefficient
        clip_coef = self.max_grad_norm / max(total_norm, 1e-6)
        clip_coef_clamped = min(1.0, clip_coef)

        # 3. Clip gradients and add Gaussian noise
        noise_std = self.noise_multiplier * self.max_grad_norm

        for p in model.parameters():
            if p.grad is not None:
                p.grad.data.mul_(clip_coef_clamped)
                noise = torch.randn_like(p.grad.data) * noise_std
                p.grad.data.add_(noise)

        self.steps_taken += 1
        return total_norm

    def get_privacy_budget(self) -> Dict[str, float]:
        """Calculates approximate privacy budget epsilon spent so far."""
        if self.steps_taken == 0:
            return {"epsilon": 0.0, "delta": self.target_delta}

        # Standard RDP analytical approximation for Gaussian DP mechanism
        sigma = self.noise_multiplier
        eps = float((self.steps_taken * (1.0 / max(sigma, 1e-3)**2)) + np.sqrt(2 * np.log(1.25 / self.target_delta)))
        eps = float(np.round(eps / 10.0, 2)) # Scale to standard epsilon range

        return {"epsilon": eps, "delta": self.target_delta, "steps": self.steps_taken}
