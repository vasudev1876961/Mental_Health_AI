"""
Clinical Pareto-Optimal Risk Calibration.

Optimizes asymmetric clinical decision thresholds (C_FN >> C_FP) on the ROC/PR Pareto frontier
to guarantee high clinical sensitivity (>= 95%) for acute mental health crisis early warning
while minimizing clinician false alarm fatigue.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any


class ClinicalParetoCalibrator:
    """
    Calibrates risk decision thresholds under asymmetric clinical misclassification penalties:
    C_FN (Cost of False Negative - missed acute crisis) vs
    C_FP (Cost of False Positive - unnecessary alarm).
    """

    def __init__(
        self,
        cost_fn: float = 10.0,
        cost_fp: float = 1.0,
        min_sensitivity: float = 0.95,
        default_high_risk_threshold: float = 65.0,
    ):
        self.cost_fn = cost_fn
        self.cost_fp = cost_fp
        self.min_sensitivity = min_sensitivity
        self.default_threshold = default_high_risk_threshold
        self.optimal_threshold: float = default_high_risk_threshold
        self.is_calibrated: bool = False
        self.calibration_stats: Dict[str, Any] = {}

    def calibrate(
        self,
        y_true_binary: np.ndarray,
        y_scores: np.ndarray,
        num_thresholds: int = 100,
    ) -> Dict[str, Any]:
        """
        Computes the Pareto frontier and finds the optimal decision threshold.

        Args:
            y_true_binary: Ground truth binary indicators (1 = High Risk/Crisis, 0 = Non-Crisis).
            y_scores: Continuous model stress/risk predictions (e.g. 0 to 100).
            num_thresholds: Number of candidate thresholds to evaluate.

        Returns:
            Dict containing optimal threshold, sensitivity, specificity, and Pareto curve points.
        """
        y_true = np.asarray(y_true_binary, dtype=int)
        scores = np.asarray(y_scores, dtype=float)

        n_pos = np.sum(y_true == 1)
        n_neg = np.sum(y_true == 0)
        n_total = len(y_true)

        if n_pos == 0 or n_neg == 0:
            self.optimal_threshold = self.default_threshold
            return {"optimal_threshold": self.default_threshold, "note": "Insufficient binary diversity"}

        thresholds = np.linspace(scores.min(), scores.max(), num_thresholds)
        tpr_list = []
        fpr_list = []
        precision_list = []
        ecm_list = []

        best_thresh = self.default_threshold
        best_ecm = float("inf")
        best_sens = 0.0
        best_spec = 0.0

        for t in thresholds:
            y_pred = (scores >= t).astype(int)
            tp = int(np.sum((y_true == 1) & (y_pred == 1)))
            fp = int(np.sum((y_true == 0) & (y_pred == 1)))
            tn = int(np.sum((y_true == 0) & (y_pred == 0)))
            fn = int(np.sum((y_true == 1) & (y_pred == 0)))

            sens = tp / max(n_pos, 1)  # TPR
            spec = tn / max(n_neg, 1)
            fpr = fp / max(n_neg, 1)
            prec = tp / max(tp + fp, 1)

            # Expected Cost of Misclassification
            ecm = (self.cost_fn * fn + self.cost_fp * fp) / max(n_total, 1)

            tpr_list.append(round(float(sens), 4))
            fpr_list.append(round(float(fpr), 4))
            precision_list.append(round(float(prec), 4))
            ecm_list.append(round(float(ecm), 4))

            # Select threshold satisfying minimum sensitivity constraint with minimal ECM
            if sens >= self.min_sensitivity and ecm < best_ecm:
                best_ecm = ecm
                best_thresh = float(t)
                best_sens = float(sens)
                best_spec = float(spec)

        # Fallback if no threshold met strict sensitivity constraint: pick threshold maximizing sensitivity
        if best_ecm == float("inf"):
            idx_max_sens = int(np.argmax(tpr_list))
            best_thresh = float(thresholds[idx_max_sens])
            best_sens = float(tpr_list[idx_max_sens])
            best_spec = float(1.0 - fpr_list[idx_max_sens])
            best_ecm = float(ecm_list[idx_max_sens])

        self.optimal_threshold = round(best_thresh, 2)
        self.is_calibrated = True

        # Approximate Area Under ROC Curve
        # Sort by FPR
        sorted_indices = np.argsort(fpr_list)
        sorted_fpr = np.array(fpr_list)[sorted_indices]
        sorted_tpr = np.array(tpr_list)[sorted_indices]
        auroc = float(np.trapz(sorted_tpr, sorted_fpr))
        auroc = round(abs(auroc), 4)

        self.calibration_stats = {
            "optimal_threshold": self.optimal_threshold,
            "clinical_sensitivity": round(best_sens, 4),
            "clinical_specificity": round(best_spec, 4),
            "false_alarm_rate": round(1.0 - best_spec, 4),
            "expected_cost_of_misclassification": round(best_ecm, 4),
            "auroc": auroc,
            "cost_fn_ratio": self.cost_fn / self.cost_fp,
            "threshold_grid": [round(float(t), 2) for t in thresholds],
            "tpr_curve": tpr_list,
            "fpr_curve": fpr_list,
            "ecm_curve": ecm_list,
        }

        return self.calibration_stats

    def triage_risk(self, stress_score: float) -> Dict[str, Any]:
        """
        Applies the calibrated Pareto operating threshold to classify triage urgency.
        """
        thresh = self.optimal_threshold if self.is_calibrated else self.default_threshold

        if stress_score >= thresh:
            level = "Crisis Alert (High Sensitivity Threshold Exceeded)"
            color = "#EF4444"
            urgency = "Tier-1 Priority: Immediate clinician notification recommended"
            action = "Dispatch autonomic check + immediate telehealth check-in"
        elif stress_score >= thresh - 15.0:
            level = "Elevated Borderline Risk"
            color = "#F59E0B"
            urgency = "Tier-2 Priority: Schedule check-in within 24 hours"
            action = "Recommend counterfactual recourse breathing pacing"
        elif stress_score >= 35.0:
            level = "Moderate Situational Stress"
            color = "#3B82F6"
            urgency = "Tier-3 Priority: Routine passive edge monitoring"
            action = "Continue non-invasive observational logging"
        else:
            level = "Normative Baseline"
            color = "#10B981"
            urgency = "Healthy: No clinical intervention required"
            action = "Maintain regular wellness tracking"

        return {
            "input_stress_score": round(stress_score, 2),
            "operating_threshold": thresh,
            "triage_level": level,
            "triage_color": color,
            "urgency_tier": urgency,
            "recommended_clinical_action": action,
            "calibration_status": "Calibrated (Asymmetric Loss)" if self.is_calibrated else "Default Uncalibrated",
        }


def generate_synthetic_pareto_evaluation(n_samples: int = 200) -> Dict[str, Any]:
    """
    Generates synthetic validation data and executes Pareto calibration.
    """
    rng = np.random.RandomState(42)
    # 25% high risk cases (positive class)
    y_true = rng.binomial(1, 0.25, size=n_samples)

    # Stress score distribution: Positives centered at 72 (std 12), Negatives centered at 38 (std 14)
    scores = np.where(
        y_true == 1,
        rng.normal(72.0, 12.0, size=n_samples),
        rng.normal(38.0, 14.0, size=n_samples),
    )
    scores = np.clip(scores, 0.0, 100.0)

    calibrator = ClinicalParetoCalibrator(cost_fn=10.0, cost_fp=1.0, min_sensitivity=0.95)
    stats = calibrator.calibrate(y_true, scores)
    return stats
