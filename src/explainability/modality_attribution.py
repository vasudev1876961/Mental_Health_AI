"""
Modality Attribution Analyzer.

Calculates relative percentage contributions of Vision, Audio, and Text streams to predictions.
"""

import torch
import numpy as np
from typing import Dict, List


class ModalityAttributionAnalyzer:
    """Analyzes percentage contribution across Vision, Audio, and Text streams."""

    def __init__(self):
        pass

    def compute_attribution_percentage(self, weights_tensor: torch.Tensor) -> Dict[str, float]:
        """Converts raw modality weights tensor [3] into percentage contributions.

        Args:
            weights_tensor: Tensor or array of shape [3] (Vision, Audio, Text weights).

        Returns:
            Dict: {'vision_pct': float, 'audio_pct': float, 'text_pct': float}
        """
        if isinstance(weights_tensor, torch.Tensor):
            w = weights_tensor.detach().cpu().numpy()
        else:
            w = np.array(weights_tensor)

        if w.ndim == 2:
            w = np.mean(w, axis=0)

        total = np.sum(w) + 1e-5
        probs = w / total

        return {
            "vision_pct": float(np.round(probs[0] * 100.0, 1)),
            "audio_pct": float(np.round(probs[1] * 100.0, 1)),
            "text_pct": float(np.round(probs[2] * 100.0, 1)),
        }
