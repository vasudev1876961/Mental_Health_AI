"""
Distribution-Free Split Conformal Prediction and Calibrated Uncertainty Intervals.

Provides finite-sample statistical coverage guarantees:
    P( Y in [L(X), U(X)] ) >= 1 - alpha
Without making distributional or Gaussianity assumptions about user stress scores.
"""

from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import torch


class ConformalRiskPredictor:
    """Calibrates and generates distribution-free conformal prediction intervals."""

    def __init__(self, alpha: float = 0.10):
        """
        Args:
            alpha: Significance level (default 0.10 for 90% confidence coverage).
        """
        self.alpha = float(np.clip(alpha, 0.01, 0.50))
        self.calibration_scores: np.ndarray = np.array([], dtype=np.float64)
        self.calibrated_quantile: float = 7.5 # Sensible initial default (in points out of 100)
        self.is_calibrated: bool = False

    def calibrate(
        self,
        y_true: Union[np.ndarray, List[float], torch.Tensor],
        y_pred: Union[np.ndarray, List[float], torch.Tensor],
    ) -> float:
        """Calibrates non-conformity scores using holdout validation data.

        Non-conformity score R_i = | y_i - y_hat_i |

        Args:
            y_true: Ground truth stress scores (0 - 100)
            y_pred: Model predicted stress scores (0 - 100)

        Returns:
            The calibrated quantile q_hat.
        """
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()

        y_true = np.asarray(y_true, dtype=np.float64).ravel()
        y_pred = np.asarray(y_pred, dtype=np.float64).ravel()

        if len(y_true) == 0:
            return self.calibrated_quantile

        residuals = np.abs(y_true - y_pred)
        self.calibration_scores = np.sort(residuals)

        n = len(residuals)
        # Finite sample correction quantile: ceil((n + 1) * (1 - alpha)) / n
        p_val = min(1.0, np.ceil((n + 1) * (1.0 - self.alpha)) / n)
        self.calibrated_quantile = float(np.quantile(self.calibration_scores, p_val, method="higher"))
        self.is_calibrated = True

        return self.calibrated_quantile

    def predict_interval(
        self,
        y_pred: Union[float, np.ndarray, torch.Tensor],
        alpha_override: Optional[float] = None,
    ) -> Dict[str, Union[float, List[float]]]:
        """Produces calibrated lower and upper prediction bounds.

        Args:
            y_pred: Predicted stress score
            alpha_override: Optional custom significance level

        Returns:
            Dict containing 'pred', 'lower_bound', 'upper_bound', 'interval_width', 'confidence_level'.
        """
        if isinstance(y_pred, torch.Tensor):
            y_pred = float(y_pred.detach().cpu().item())
        elif isinstance(y_pred, np.ndarray):
            y_pred = float(y_pred.ravel()[0])
        else:
            y_pred = float(y_pred)

        alpha = self.alpha if alpha_override is None else float(alpha_override)

        if len(self.calibration_scores) > 0 and (alpha_override is not None and alpha_override != self.alpha):
            n = len(self.calibration_scores)
            p_val = min(1.0, np.ceil((n + 1) * (1.0 - alpha)) / n)
            q = float(np.quantile(self.calibration_scores, p_val, method="higher"))
        else:
            q = self.calibrated_quantile

        lower = float(np.clip(y_pred - q, 0.0, 100.0))
        upper = float(np.clip(y_pred + q, 0.0, 100.0))

        return {
            "prediction": round(y_pred, 2),
            "lower_bound": round(lower, 2),
            "upper_bound": round(upper, 2),
            "margin_q": round(q, 2),
            "interval_width": round(upper - lower, 2),
            "confidence_level": round(1.0 - alpha, 2),
        }

    def predict_classification_set(
        self,
        class_probs: Union[List[float], np.ndarray, torch.Tensor],
        class_labels: Optional[List[str]] = None,
        alpha_override: Optional[float] = None,
    ) -> Dict[str, any]:
        """Forms a conformal prediction set for discrete risk levels (Low, Medium, High)."""
        if class_labels is None:
            class_labels = ["Low", "Medium", "High"]

        if isinstance(class_probs, torch.Tensor):
            probs = class_probs.detach().cpu().numpy().ravel()
        else:
            probs = np.asarray(class_probs, dtype=np.float64).ravel()

        alpha = self.alpha if alpha_override is None else float(alpha_override)
        target_coverage = 1.0 - alpha

        # Sort classes descending by probability
        sorted_indices = np.argsort(probs)[::-1]
        cumulative = 0.0
        selected_classes = []

        for idx in sorted_indices:
            selected_classes.append(class_labels[idx])
            cumulative += probs[idx]
            if cumulative >= target_coverage:
                break

        return {
            "prediction_set": selected_classes,
            "set_size": len(selected_classes),
            "probabilities": {class_labels[i]: round(float(probs[i]), 3) for i in range(len(class_labels))},
            "cumulative_mass": round(cumulative, 3),
            "confidence_level": round(target_coverage, 2),
        }

    def evaluate_coverage(
        self,
        y_true: Union[np.ndarray, List[float]],
        y_pred: Union[np.ndarray, List[float]],
    ) -> Dict[str, float]:
        """Evaluates empirical coverage on an evaluation set: fraction of targets inside [L, U]."""
        y_true = np.asarray(y_true, dtype=np.float64).ravel()
        y_pred = np.asarray(y_pred, dtype=np.float64).ravel()

        q = self.calibrated_quantile
        lower_bounds = np.clip(y_pred - q, 0.0, 100.0)
        upper_bounds = np.clip(y_pred + q, 0.0, 100.0)

        inside = (y_true >= lower_bounds) & (y_true <= upper_bounds)
        empirical_coverage = float(np.mean(inside))
        avg_width = float(np.mean(upper_bounds - lower_bounds))

        return {
            "target_coverage": round(1.0 - self.alpha, 3),
            "empirical_coverage": round(empirical_coverage, 3),
            "coverage_gap": round(empirical_coverage - (1.0 - self.alpha), 3),
            "mean_interval_width": round(avg_width, 2),
        }
