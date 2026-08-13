"""
Unit Tests for Stage 2 Vision & Behavioral Feature Engine.
"""

import unittest
import numpy as np
import torch

from src.vision.detector import FaceDetector
from src.vision.landmarks import FaceLandmarkExtractor
from src.vision.behavior import FacialBehaviorEngine
from src.vision.emotion import EmotionClassifier


class TestStage2VisionEngine(unittest.TestCase):

    def setUp(self):
        # Create a synthetic RGB image [480, 640, 3]
        self.synthetic_frame = np.full((480, 640, 3), 128, dtype=np.uint8)

    def test_face_detector(self):
        """Verify face detection and cropping."""
        detector = FaceDetector()
        bbox = detector.detect(self.synthetic_frame)

        self.assertIsNotNone(bbox)
        self.assertIn("x", bbox)
        self.assertIn("bbox_norm", bbox)

        crop = detector.crop_face(self.synthetic_frame, bbox, target_size=(224, 224))
        self.assertEqual(crop.shape, (224, 224, 3))

    def test_landmark_extractor(self):
        """Verify 468 3D landmark extraction."""
        extractor = FaceLandmarkExtractor()
        landmarks = extractor.extract_landmarks(self.synthetic_frame)

        self.assertIsNotNone(landmarks)
        self.assertEqual(landmarks.shape, (468, 3))
        # Verify normalized coordinate range
        self.assertTrue(np.all(landmarks[:, 0] >= 0.0) and np.all(landmarks[:, 0] <= 1.0))

    def test_behavioral_engine(self):
        """Verify 18-dimensional feature vector extraction."""
        extractor = FaceLandmarkExtractor()
        landmarks = extractor.extract_landmarks(self.synthetic_frame)

        engine = FacialBehaviorEngine()
        features = engine.compute_features(landmarks, image_shape=(480, 640))

        self.assertEqual(features.shape, (18,))
        # EAR should be positive
        self.assertGreaterEqual(features[0], 0.0)
        # MAR should be positive
        self.assertGreaterEqual(features[1], 0.0)

    def test_emotion_classifier(self):
        """Verify facial emotion classifier output shape and probabilities."""
        classifier = EmotionClassifier()
        dummy_input = torch.randn(2, 3, 224, 224)
        logits = classifier(dummy_input)

        self.assertEqual(logits.shape, (2, 7))

        probs = classifier.predict_probs(self.synthetic_frame)
        self.assertEqual(probs.shape, (7,))
        self.assertAlmostEqual(float(np.sum(probs)), 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
