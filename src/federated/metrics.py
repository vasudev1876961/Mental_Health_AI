"""
FL Metrics Tracker.

Records global loss, MAE, RMSE, F1-Score, and convergence rounds for FL experiments.
"""

from typing import Dict, List


class FLMetricsTracker:
    """Tracks metrics history across Federated Learning communication rounds."""

    def __init__(self):
        self.history = []

    def record_round(self, round_num: int, loss: float, mae: float, client_metrics: Dict = None):
        """Records metrics for a specific FL round."""
        entry = {
            "round": round_num,
            "global_loss": float(loss),
            "global_mae": float(mae),
            "client_metrics": client_metrics or {},
        }
        self.history.append(entry)

    def get_summary(self) -> Dict[str, float]:
        """Returns final summary metrics."""
        if not self.history:
            return {"final_loss": 0.0, "final_mae": 0.0}

        latest = self.history[-1]
        return {
            "final_loss": latest["global_loss"],
            "final_mae": latest["global_mae"],
            "total_rounds": len(self.history),
        }
