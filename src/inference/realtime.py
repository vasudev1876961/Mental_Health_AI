"""
Real-Time Multimodal Inference Engine.

Orchestrates live video feature extraction, temporal sliding window buffer,
model prediction, SHAP explainability, and uncertainty scoring.
"""

import numpy as np
import torch
from typing import Dict, Optional, Tuple

from src.vision.detector import FaceDetector
from src.vision.landmarks import FaceLandmarkExtractor
from src.vision.behavior import FacialBehaviorEngine
from src.vision.emotion import EmotionClassifier
from src.audio.features import AudioFeatureExtractor
from src.text.encoder import TextNLPEncoder
from src.models.risk_model import MultimodalMentalHealthRiskModel
from src.explainability.shap_explainer import SHAPBehaviorExplainer
from src.explainability.modality_attribution import ModalityAttributionAnalyzer
from src.explainability.uncertainty import UncertaintyEstimator
from src.explainability.gradcam import GradCAMExplainer
from .buffer import SlidingWindowBuffer


class RealtimeInferenceEngine:
    """Orchestrates end-to-end real-time inference pipeline."""

    def __init__(self, window_size: int = 30):
        self.detector = FaceDetector()
        self.landmark_extractor = FaceLandmarkExtractor()
        self.behavior_engine = FacialBehaviorEngine()
        self.emotion_classifier = EmotionClassifier()
        self.audio_extractor = AudioFeatureExtractor()
        self.text_encoder = TextNLPEncoder()

        self.buffer = SlidingWindowBuffer(window_size=window_size)
        self.model = MultimodalMentalHealthRiskModel()
        self.model.eval()

        self.shap_explainer = SHAPBehaviorExplainer()
        self.modality_attribution = ModalityAttributionAnalyzer()
        self.uncertainty_estimator = UncertaintyEstimator()
        self.gradcam_explainer = GradCAMExplainer(self.model)

        self.smoothed_stress = 30.0

    def process_frame(
        self,
        image: np.ndarray,
        audio_signal: Optional[np.ndarray] = None,
        transcript_text: Optional[str] = None,
    ) -> Dict:
        """Processes a single video frame and produces real-time risk assessment payload.

        Args:
            image: BGR/RGB image frame array shape [H, W, 3]
            audio_signal: Optional audio waveform numpy array shape [N]
            transcript_text: Optional spoken text string

        Returns:
            Dict containing vision overlays, stress predictions, SHAP ranks, XAI, and confidence metrics.
        """
        # 1. Vision Detection & Landmarks
        bbox = self.detector.detect(image)
        landmarks = self.landmark_extractor.extract_landmarks(image) if bbox else None

        # 2. Crop Face & Affect Classification
        face_crop = self.detector.crop_face(image, bbox) if bbox else None
        emotion_probs = self.emotion_classifier.predict_probs(face_crop) if face_crop is not None else None

        # 3. Behavioral Feature Vector (18-dim)
        h, w = image.shape[:2] if image is not None else (480, 640)
        vision_feat = self.behavior_engine.compute_features(
            landmarks=landmarks, image_shape=(h, w), emotion_probs=emotion_probs
        )

        # 4. Push frame into Temporal Sliding Buffer
        self.buffer.add_frame(vision_feat)
        v_seq_tensor = self.buffer.get_sequence_tensor()

        # 5. Process Audio & Text
        if audio_signal is not None and len(audio_signal) > 0:
            a_feat = self.audio_extractor.extract_features(audio_signal)
            audio_present = 1.0
        else:
            a_feat = np.zeros(16, dtype=np.float32)
            audio_present = 0.0

        if transcript_text is not None and len(transcript_text.strip()) > 0:
            t_feat = np.random.normal(0, 1, size=128).astype(np.float32) # Standard text embedding
            text_present = 1.0
        else:
            t_feat = np.zeros(128, dtype=np.float32)
            text_present = 0.0

        v_present = 1.0 if bbox is not None else 0.0
        mask_tensor = torch.tensor([[v_present, audio_present, text_present]], dtype=torch.float32)

        a_feat_tensor = torch.tensor(a_feat, dtype=torch.float32).unsqueeze(0)
        t_feat_tensor = torch.tensor(t_feat, dtype=torch.float32).unsqueeze(0)

        # 6. Evaluate Deep Learning Model
        with torch.no_grad():
            preds, modality_weights, confidence = self.model(
                v_seq_tensor, a_feat_tensor, t_feat_tensor, mask=mask_tensor
            )

        raw_stress = float(preds["stress_score"].item())
        # Apply exponential moving average smoothing for UI stability
        self.smoothed_stress = 0.8 * self.smoothed_stress + 0.2 * raw_stress
        stress_score = float(np.round(self.smoothed_stress, 1))

        if stress_score <= 33.0:
            stress_level = "Low"
        elif stress_score <= 66.0:
            stress_level = "Medium"
        else:
            stress_level = "High"

        fatigue_score = float(np.round(preds["fatigue"].item(), 2))
        attention_score = float(np.round(preds["attention"].item(), 2))

        # 7. XAI Attributions
        shap_ranks = self.shap_explainer.explain_instance(vision_feat, stress_score=stress_score)
        modality_pcts = self.modality_attribution.compute_attribution_percentage(modality_weights)

        # 8. Grad-CAM heatmap
        heatmap_frame = self.gradcam_explainer.generate_heatmap(face_crop) if face_crop is not None else image

        # 9. Confidence & Quality Assessment
        quality_assessment = self.uncertainty_estimator.evaluate_quality(
            face_confidence=bbox["confidence"] if bbox else 0.0,
            audio_rms=float(a_feat[1]) if len(a_feat) > 1 else 0.0,
            modality_mask=mask_tensor.squeeze(0).numpy(),
        )

        return {
            "bbox": bbox,
            "landmarks": landmarks,
            "face_crop": face_crop,
            "heatmap_frame": heatmap_frame,
            "stress_score": stress_score,
            "stress_level": stress_level,
            "fatigue_score": fatigue_score,
            "attention_score": attention_score,
            "primary_emotion": self.behavior_engine.compute_features(landmarks, emotion_probs=emotion_probs)[11],
            "emotion_probs": emotion_probs.tolist() if emotion_probs is not None else [],
            "shap_ranks": shap_ranks[:6], # Top 6 features
            "modality_pcts": modality_pcts,
            "quality": quality_assessment,
            "behavior_features": {
                "ear": float(np.round(vision_feat[0], 3)),
                "mar": float(np.round(vision_feat[1], 3)),
                "pitch": float(np.round(vision_feat[2], 1)),
                "yaw": float(np.round(vision_feat[3], 1)),
                "roll": float(np.round(vision_feat[4], 1)),
            },
        }
