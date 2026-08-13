"""
Privacy Accountant Module.

Tracks cumulative differential privacy budget over training steps.
"""

from typing import Dict


class PrivacyAccountant:
    """Tracks cumulative epsilon (epsilon, delta) spending across training iterations."""

    def __init__(self, target_delta: float = 1e-5):
        self.target_delta = target_delta
        self.history = []

    def log_step(self, epsilon: float, step: int):
        self.history.append({"step": step, "epsilon": epsilon, "delta": self.target_delta})

    def get_total_budget(self) -> Dict[str, float]:
        if not self.history:
            return {"total_epsilon": 0.0, "delta": self.target_delta}
        return {
            "total_epsilon": self.history[-1]["epsilon"],
            "delta": self.target_delta,
            "total_steps": len(self.history),
        }
