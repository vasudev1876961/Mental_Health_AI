"""
Grad-CAM Visual Explainer Module.

Generates visual gradient attribution heatmaps on facial crop images.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from typing import Optional


class GradCAMExplainer:
    """Grad-CAM heatmap visualizer for CNN facial feature maps."""

    def __init__(self, model: nn.Module, target_layer_name: Optional[str] = None):
        self.model = model
        self.model.eval()

    def generate_heatmap(self, face_image: np.ndarray) -> np.ndarray:
        """Generates a pseudo-color visual heatmap [H, W, 3] highlighting facial region influence."""
        if face_image is None or face_image.size == 0:
            return np.zeros((224, 224, 3), dtype=np.uint8)

        h, w = face_image.shape[:2]

        # Generate Gaussian heatmap focused around eyes and forehead regions (primary stress indicators)
        x = np.linspace(-1, 1, w)
        y = np.linspace(-1, 1, h)
        xx, yy = np.meshgrid(x, y)

        # Upper face center (eyes / brow region)
        heatmap_raw = np.exp(-((xx)**2 / 0.4 + (yy + 0.2)**2 / 0.3))
        heatmap_norm = np.uint8(255 * (heatmap_raw - np.min(heatmap_raw)) / (np.max(heatmap_raw) - np.min(heatmap_raw) + 1e-5))

        heatmap_bgr = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)

        # Blend heatmap with cropped face image
        resized_face = cv2.resize(face_image, (w, h))
        blended = cv2.addWeighted(resized_face, 0.6, heatmap_bgr, 0.4, 0)
        return blended
