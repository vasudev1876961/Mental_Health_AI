"""
Facial Behavioral Feature Engine.

Computes Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR), 3D Head Pose (Pitch/Yaw/Roll),
Gaze direction vectors, and facial affect indicators into an 18-dimensional frame feature vector.
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional


class FacialBehaviorEngine:
    """Calculates behavioral and biomechanical metrics from 3D face mesh landmarks."""

    # 3D Generic Model Points for Head Pose solvePnP
    MODEL_POINTS_3D = np.array(
        [
            (0.0, 0.0, 0.0),          # Nose tip (1)
            (0.0, -330.0, -65.0),     # Chin (152)
            (-225.0, 170.0, -135.0),  # Left eye left corner (33)
            (225.0, 170.0, -135.0),   # Right eye right corner (263)
            (-150.0, -150.0, -125.0), # Left Mouth corner (61)
            (150.0, -150.0, -125.0),  # Right Mouth corner (291)
        ],
        dtype=np.float64,
    )

    def __init__(self):
        pass

    def compute_ear(self, landmarks: np.ndarray, eye_indices: list) -> float:
        """Calculates Eye Aspect Ratio (EAR) for a single eye given 6 landmark points."""
        p1, p2, p3, p4, p5, p6 = landmarks[eye_indices[:6]]

        # Vertical distances
        v1 = np.linalg.norm(p2[:2] - p6[:2])
        v2 = np.linalg.norm(p3[:2] - p5[:2])
        # Horizontal distance
        h = np.linalg.norm(p1[:2] - p4[:2])

        if h == 0:
            return 0.25
        ear = (v1 + v2) / (2.0 * h)
        return float(ear)

    def compute_mar(self, landmarks: np.ndarray) -> float:
        """Calculates Mouth Aspect Ratio (MAR)."""
        # Outer mouth corners: 61, 291; upper/lower lip centers: 13, 14
        left_corner = landmarks[61][:2]
        right_corner = landmarks[291][:2]
        upper_lip = landmarks[13][:2]
        lower_lip = landmarks[14][:2]

        v = np.linalg.norm(upper_lip - lower_lip)
        h = np.linalg.norm(left_corner - right_corner)

        if h == 0:
            return 0.15
        return float(v / h)

    def estimate_head_pose(
        self, landmarks: np.ndarray, image_shape: Tuple[int, int]
    ) -> Tuple[float, float, float]:
        """Estimates 3D Head Pose (Pitch, Yaw, Roll) in degrees using OpenCV solvePnP."""
        h, w = image_shape[:2]

        # Extract 2D image points for corresponding key landmarks
        image_points = np.array(
            [
                landmarks[1][:2] * [w, h],     # Nose tip
                landmarks[152][:2] * [w, h],   # Chin
                landmarks[33][:2] * [w, h],    # Left eye corner
                landmarks[263][:2] * [w, h],   # Right eye corner
                landmarks[61][:2] * [w, h],    # Left mouth corner
                landmarks[291][:2] * [w, h],   # Right mouth corner
            ],
            dtype=np.float64,
        )

        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
            dtype=np.float64,
        )
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.MODEL_POINTS_3D,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            return 0.0, 0.0, 0.0

        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        proj_matrix = np.hstack((rotation_matrix, translation_vector))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)

        pitch = float(euler_angles[0, 0])
        yaw = float(euler_angles[1, 0])
        roll = float(euler_angles[2, 0])

        return pitch, yaw, roll

    def compute_features(
        self,
        landmarks: np.ndarray,
        image_shape: Tuple[int, int] = (480, 640),
        emotion_probs: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Extracts complete 18-dimensional frame-level feature vector.

        Returns:
            Numpy array of shape [18]
        """
        if landmarks is None or len(landmarks) < 300:
            # Fallback zero/neutral vector
            vec = np.zeros(18, dtype=np.float32)
            vec[0] = 0.25 # ear
            vec[1] = 0.15 # mar
            vec[8] = 0.60 # eye openness
            vec[11] = 0.70 # neutral emotion default
            return vec

        # Left Eye indices: 33, 160, 158, 133, 153, 144
        left_ear = self.compute_ear(landmarks, [33, 160, 158, 133, 153, 144])
        # Right Eye indices: 362, 385, 387, 263, 373, 380
        right_ear = self.compute_ear(landmarks, [362, 385, 387, 263, 373, 380])
        mean_ear = (left_ear + right_ear) / 2.0

        mar = self.compute_mar(landmarks)

        pitch, yaw, roll = self.estimate_head_pose(landmarks, image_shape)

        # Gaze vector approximation
        nose = landmarks[1]
        gaze_x = float(nose[0] - 0.5) * 2.0
        gaze_y = float(nose[1] - 0.5) * 2.0
        gaze_z = float(nose[2])

        eye_openness = float(np.clip(mean_ear / 0.35, 0.0, 1.0))
        smile_intensity = float(np.clip(mar / 0.50, 0.0, 1.0))
        blink_metric = 1.0 if mean_ear < 0.18 else 0.0

        if emotion_probs is None:
            # Default neutral emotion distribution (7 classes)
            emotion_probs = np.array([0.7, 0.1, 0.05, 0.05, 0.05, 0.03, 0.02], dtype=np.float32)

        feat_vector = np.array(
            [
                mean_ear,
                mar,
                pitch,
                yaw,
                roll,
                gaze_x,
                gaze_y,
                gaze_z,
                eye_openness,
                smile_intensity,
                blink_metric,
                *emotion_probs[:7],
            ],
            dtype=np.float32,
        )

        return feat_vector
