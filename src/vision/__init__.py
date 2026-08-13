"""
Vision Module: Face detection, 3D face mesh landmarking, behavioral feature extraction, and facial affect classification.
"""

from .detector import FaceDetector
from .landmarks import FaceLandmarkExtractor
from .behavior import FacialBehaviorEngine
from .emotion import EmotionClassifier

__all__ = [
    "FaceDetector",
    "FaceLandmarkExtractor",
    "FacialBehaviorEngine",
    "EmotionClassifier",
]
