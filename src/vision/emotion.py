"""
Facial Emotion Classifier Module.

Classifies cropped face images into 7 affect categories: Neutral, Happy, Sad,
Angry, Surprised, Fearful, Disgusted.
"""

import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from typing import Dict, Tuple, List


class EmotionClassifier(nn.Module):
    """Facial affect classification model (ResNet18 backbone)."""

    EMOTIONS = ["neutral", "happy", "sad", "angry", "surprised", "fearful", "disgusted"]

    def __init__(self, num_classes: int = 7, pretrained: bool = False):
        super().__init__()
        self.num_classes = num_classes

        # Use ResNet18 backbone
        try:
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet18(weights=weights)
        except Exception:
            self.backbone = models.resnet18(weights=None)

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input face crop tensor of shape [B, 3, H, W] or [3, H, W]
        Returns:
            Logits of shape [B, 7]
        """
        if x.dim() == 3:
            x = x.unsqueeze(0)
        return self.backbone(x)

    def predict_probs(self, face_image: np.ndarray) -> np.ndarray:
        """Predicts emotion probabilities for a single face image array.

        Args:
            face_image: BGR/RGB numpy array of shape [H, W, 3].

        Returns:
            Numpy array of shape [7] with normalized probabilities.
        """
        self.eval()
        if face_image is None or face_image.size == 0:
            return np.array([0.7, 0.1, 0.05, 0.05, 0.05, 0.03, 0.02], dtype=np.float32)

        import cv2
        resized = cv2.resize(face_image, (224, 224))
        # Convert HWC to CHW and normalize to [0, 1]
        tensor_img = torch.tensor(resized, dtype=torch.float32).permute(2, 0, 1) / 255.0
        # Standard ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor_img = (tensor_img - mean) / std
        tensor_img = tensor_img.unsqueeze(0)

        with torch.no_grad():
            logits = self.forward(tensor_img)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        return probs.astype(np.float32)
