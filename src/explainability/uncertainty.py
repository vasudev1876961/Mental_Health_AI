"""
Uncertainty & Data Quality Estimator.

Calculates confidence score, face clarity, audio quality, and data warnings.
"""

import numpy as np
from typing import Dict, Optional, Tuple


class UncertaintyEstimator:
    """Estimates prediction confidence and warns on poor video/audio input quality."""

    def __init__(self, quality_threshold: float = 0.4):
        self.quality_threshold = quality_threshold

    def evaluate_quality(
        self,
        face_confidence: float = 0.9,
        audio_rms: float = 0.1,
        modality_mask: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Calculates composite confidence and data quality ratings.

        Returns:
            Dict containing 'confidence_score', 'face_quality', 'audio_quality', and 'warnings'.
        """
        if modality_mask is None:
            modality_mask = np.array([1.0, 1.0, 1.0])

        face_present = float(modality_mask[0])
        audio_present = float(modality_mask[1])
        text_present = float(modality_mask[2])

        face_quality = face_confidence if face_present > 0 else 0.0
        audio_quality = float(np.clip(audio_rms / 0.2, 0.0, 1.0)) if audio_present > 0 else 0.0

        # Weighted composition
        comp_score = (face_quality * 0.5 + audio_quality * 0.3 + text_present * 0.2)
        confidence_score = float(np.round(comp_score, 2))

        warnings = []
        if face_quality < self.quality_threshold:
            warnings.append("Low face landmark quality or poor lighting")
        if audio_present > 0 and audio_quality < 0.2:
            warnings.append("Low audio signal energy or background noise")
        if face_present == 0:
            warnings.append("Camera feed unavailable / face obscured")

        return {
            "confidence_score": confidence_score,
            "face_quality": float(np.round(face_quality, 2)),
            "audio_quality": float(np.round(audio_quality, 2)),
            "is_reliable": confidence_score >= self.quality_threshold,
            "warnings": warnings,
        }
