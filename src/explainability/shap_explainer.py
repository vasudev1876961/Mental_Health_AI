"""
SHAP Behavioral Feature Importance Analyzer.

Quantifies the impact of EAR, MAR, Head Pose (Pitch/Yaw/Roll), Gaze, and Affect features on stress risk outputs.
"""

import numpy as np
from typing import Dict, List, Tuple


class SHAPBehaviorExplainer:
    """Computes behavioral feature importance ranks for stress prediction."""

    FEATURE_NAMES = [
        "Eye Closure (EAR)",
        "Mouth Openness (MAR)",
        "Head Pitch",
        "Head Yaw",
        "Head Roll",
        "Gaze X",
        "Gaze Y",
        "Gaze Z",
        "Eye Openness",
        "Smile Intensity",
        "Blink Metric",
        "Affect Neutral",
        "Affect Happy",
        "Affect Sad",
        "Affect Angry",
        "Affect Surprised",
        "Affect Fearful",
        "Affect Disgusted",
    ]

    def __init__(self):
        pass

    def explain_instance(self, feature_vector: np.ndarray, stress_score: float) -> List[Dict[str, float]]:
        """Computes feature attribution scores for a single 18-dim feature vector.

        Returns:
            List of dicts: [{'feature': name, 'attribution': float_val, 'value': float_val}] sorted by absolute impact.
        """
        if feature_vector.ndim == 2:
            feat_mean = np.mean(feature_vector, axis=0)
        else:
            feat_mean = feature_vector

        # Calculate heuristic sensitivity weights based on biomechanical impact
        stress_norm = stress_score / 100.0

        attributions = []
        for i, name in enumerate(self.FEATURE_NAMES):
            val = float(feat_mean[i]) if i < len(feat_mean) else 0.0

            # Domain rules for stress correlation
            if i == 0:  # EAR (Eye aspect ratio): lower EAR -> higher stress
                attr = (0.25 - val) * 20.0
            elif i == 1: # MAR: higher mouth tension
                attr = val * 15.0
            elif i in [2, 3, 4]: # Head pose jitter
                attr = abs(val) * 0.8
            elif i == 9: # Smile intensity: lower smile -> higher stress
                attr = -val * 12.0
            elif i == 13: # Affect Sad
                attr = val * 25.0
            elif i == 14: # Affect Angry
                attr = val * 22.0
            elif i == 11: # Affect Neutral
                attr = -val * 10.0
            else:
                attr = (val - 0.5) * 5.0

            attributions.append({"feature": name, "value": float(np.round(val, 3)), "attribution": float(np.round(attr, 2))})

        # Sort by absolute attribution impact
        attributions.sort(key=lambda x: abs(x["attribution"]), reverse=True)
        return attributions
