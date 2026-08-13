"""
Synthetic Multimodal Dataset Generator for Pipeline Testing and Headless Simulation.

Generating realistic synthetic multimodal sequences (Vision, Audio, Text) partitioned
strictly by subject ID to prevent data leakage across train/validation/test splits.
"""

import os
import json
import argparse
import numpy as np
import torch


class SyntheticMultimodalDatasetGenerator:
    """Generates synthetic multimodal mental health behavioral sequence data for testing."""

    EMOTIONS = ["neutral", "happy", "sad", "angry", "surprised", "fearful", "disgusted"]

    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)

    def generate_subject_data(
        self,
        subject_id: str,
        num_sequences: int = 10,
        window_size: int = 30,
        vision_dim: int = 18,
        audio_dim: int = 16,
        text_dim: int = 128,
        baseline_stress: float = None,
    ) -> dict:
        """Generates sequences for a single subject with consistent subject-level baseline traits."""
        if baseline_stress is None:
            baseline_stress = float(np.random.uniform(10.0, 90.0))

        sequences = []

        for seq_idx in range(num_sequences):
            # Introduce sequence-level variation around baseline stress
            seq_stress = np.clip(baseline_stress + np.random.normal(0, 8.0), 0.0, 100.0)

            # High stress correlates with higher pitch/energy, lower EAR (fatigue/squinting), and affect shifts
            stress_norm = seq_stress / 100.0

            # 1. Vision features: shape [window_size, vision_dim]
            vision_seq = np.zeros((window_size, vision_dim), dtype=np.float32)

            for t in range(window_size):
                # EAR (Eye Aspect Ratio): lower when tired/stressed (~0.15 - 0.35)
                ear = np.clip(0.30 - 0.10 * stress_norm + np.random.normal(0, 0.02), 0.12, 0.40)
                # MAR (Mouth Aspect Ratio): ~0.0 - 0.5
                mar = np.clip(0.15 + np.random.normal(0, 0.05), 0.0, 0.6)
                # Head pose: pitch, yaw, roll in degrees
                pitch = np.random.normal(0, 5.0 + 3.0 * stress_norm)
                yaw = np.random.normal(0, 6.0 + 4.0 * stress_norm)
                roll = np.random.normal(0, 3.0)
                # Gaze vector: x, y, z
                gaze_x = np.random.normal(0, 0.1)
                gaze_y = np.random.normal(0, 0.1)
                gaze_z = np.random.normal(1.0, 0.05)
                # Additional metrics
                eye_openness = ear / 0.4
                smile_intensity = np.clip(0.2 - 0.15 * stress_norm + np.random.normal(0, 0.05), 0, 1)
                blink_rate = np.random.poisson(3.0 if stress_norm > 0.5 else 1.5)

                # Affect vector (7 emotion probabilities)
                if stress_norm > 0.6:
                    affect_logits = np.array([0.2, 0.05, 0.35, 0.25, 0.05, 0.08, 0.02])
                elif stress_norm < 0.3:
                    affect_logits = np.array([0.5, 0.4, 0.03, 0.02, 0.03, 0.01, 0.01])
                else:
                    affect_logits = np.array([0.6, 0.15, 0.1, 0.05, 0.05, 0.03, 0.02])

                affect_probs = np.exp(affect_logits) / np.sum(np.exp(affect_logits))

                # Combine into 18-dim vector
                frame_feat = np.array(
                    [
                        ear,
                        mar,
                        pitch,
                        yaw,
                        roll,
                        gaze_x,
                        gaze_y,
                        gaze_z,
                        eye_openness,
                        smile_intensity,
                        float(blink_rate),
                        *affect_probs,
                    ],
                    dtype=np.float32,
                )

                # Pad or truncate if dimensions differ
                if len(frame_feat) < vision_dim:
                    frame_feat = np.pad(frame_feat, (0, vision_dim - len(frame_feat)))
                else:
                    frame_feat = frame_feat[:vision_dim]

                vision_seq[t] = frame_feat

            # 2. Audio features: shape [audio_dim]
            # Higher stress -> higher mean pitch (F0), higher energy
            pitch_f0 = 120.0 + 80.0 * stress_norm + np.random.normal(0, 10.0)
            energy = 0.3 + 0.5 * stress_norm + np.random.normal(0, 0.05)
            speech_rate = 3.0 + 2.0 * stress_norm + np.random.normal(0, 0.5)
            mfccs = np.random.normal(stress_norm, 1.0, size=13)

            audio_feat = np.array([pitch_f0, energy, speech_rate, *mfccs], dtype=np.float32)
            if len(audio_feat) < audio_dim:
                audio_feat = np.pad(audio_feat, (0, audio_dim - len(audio_feat)))
            else:
                audio_feat = audio_feat[:audio_dim]

            # 3. Text embedding: shape [text_dim]
            # Text embedding correlated with sentiment/stress direction
            text_emb = np.random.normal(stress_norm * 0.5, 1.0, size=text_dim).astype(np.float32)

            sample_transcripts = [
                "I feel somewhat overwhelmed by the workload today.",
                "Everything is going smooth and I feel relaxed.",
                "I am having trouble focusing and feeling fatigued.",
                "Today was a productive day overall.",
                "I am feeling anxious about the upcoming deadline.",
            ]
            transcript = sample_transcripts[int(stress_norm * (len(sample_transcripts) - 1))]

            # 4. Target indicators
            stress_score = float(np.round(seq_stress, 2))
            if stress_score <= 33.0:
                stress_label = "Low"
            elif stress_score <= 66.0:
                stress_label = "Medium"
            else:
                stress_label = "High"

            fatigue_score = float(np.clip(0.2 + 0.6 * stress_norm + np.random.normal(0, 0.1), 0.0, 1.0))
            attention_score = float(np.clip(0.9 - 0.5 * stress_norm + np.random.normal(0, 0.1), 0.0, 1.0))

            sequences.append(
                {
                    "sequence_id": f"{subject_id}_seq_{seq_idx:03d}",
                    "vision": vision_seq.tolist(),
                    "audio": audio_feat.tolist(),
                    "text": text_emb.tolist(),
                    "transcript": transcript,
                    "targets": {
                        "stress_score": stress_score,
                        "stress_label": stress_label,
                        "fatigue": fatigue_score,
                        "attention": attention_score,
                        "primary_emotion": self.EMOTIONS[np.argmax(vision_seq[:, 11:].mean(axis=0))],
                    },
                }
            )

        return {
            "subject_id": subject_id,
            "baseline_stress": baseline_stress,
            "num_sequences": num_sequences,
            "sequences": sequences,
        }

    def generate_dataset(
        self,
        num_subjects: int = 20,
        output_dir: str = "data/synthetic",
        num_sequences_per_subject: int = 10,
        window_size: int = 30,
    ) -> list:
        """Generates dataset files partitioned by subject into target directory."""
        os.makedirs(output_dir, exist_ok=True)
        subject_filepaths = []

        print(f"Generating synthetic multimodal dataset with {num_subjects} subjects...")

        metadata = {"total_subjects": num_subjects, "subjects": []}

        for s_idx in range(1, num_subjects + 1):
            subject_id = f"subject_{s_idx:03d}"

            # Create varying stress distribution to support non-IID client experiments later
            if s_idx <= num_subjects * 0.3:
                baseline_stress = float(np.random.uniform(10.0, 33.0)) # Mostly low stress
            elif s_idx <= num_subjects * 0.7:
                baseline_stress = float(np.random.uniform(34.0, 66.0)) # Mostly medium stress
            else:
                baseline_stress = float(np.random.uniform(67.0, 95.0)) # Mostly high stress

            subj_data = self.generate_subject_data(
                subject_id=subject_id,
                num_sequences=num_sequences_per_subject,
                window_size=window_size,
                baseline_stress=baseline_stress,
            )

            file_path = os.path.join(output_dir, f"{subject_id}.json")
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(subj_data, f, indent=2)

            subject_filepaths.append(file_path)
            metadata["subjects"].append(
                {
                    "subject_id": subject_id,
                    "baseline_stress": baseline_stress,
                    "file_path": file_path,
                }
            )

        metadata_path = os.path.join(output_dir, "dataset_metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        print(f"Successfully generated synthetic dataset in '{output_dir}'.")
        return subject_filepaths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthetic Multimodal Dataset Generator")
    parser.add_argument("--num-subjects", type=int, default=20, help="Total subjects to generate")
    parser.add_argument("--output-dir", type=str, default="data/synthetic", help="Output directory")
    parser.add_argument("--seq-per-subj", type=int, default=10, help="Sequences per subject")
    parser.add_argument("--window-size", type=int, default=30, help="Frames per sequence")

    args = parser.parse_args()

    generator = SyntheticMultimodalDatasetGenerator(seed=42)
    generator.generate_dataset(
        num_subjects=args.num_subjects,
        output_dir=args.output_dir,
        num_sequences_per_subject=args.seq_per-subj if hasattr(args, "seq_per_subj") else 10,
        window_size=args.window_size,
    )
