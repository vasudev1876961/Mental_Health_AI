"""
Quick Demo Script: Tests sample predictions using the Real-Time Multimodal Inference Engine.

Usage:
    python demo_predict_sample.py
"""

import os
import json
import numpy as np
from src.inference.realtime import RealtimeInferenceEngine
from src.data.dataset_generator import SyntheticMultimodalDatasetGenerator


def run_sample_demo():
    print("==================================================================")
    print("  PRIVACY-PRESERVING MULTIMODAL MENTAL HEALTH RISK ASSESSMENT")
    print("==================================================================\n")

    # 1. Ensure sample data exists
    data_dir = "data/synthetic"
    sample_file = os.path.join(data_dir, "subject_001.json")

    if not os.path.exists(sample_file):
        print("Generating sample synthetic dataset...")
        gen = SyntheticMultimodalDatasetGenerator(seed=42)
        gen.generate_dataset(num_subjects=5, output_dir=data_dir)

    # 2. Load a sample sequence
    with open(sample_file, "r", encoding="utf-8") as f:
        subj_data = json.load(f)

    sample_seq = subj_data["sequences"][0]
    transcript = sample_seq["transcript"]
    ground_truth = sample_seq["targets"]

    print(f"LOADED SAMPLE SUBJECT: {subj_data['subject_id']}")
    print(f"Sample Transcript     : \"{transcript}\"")
    print(f"Ground-Truth Stress   : {ground_truth['stress_score']}% ({ground_truth['stress_label']} Risk)\n")

    # 3. Initialize Inference Engine
    print("Initializing Real-Time Multimodal Inference Engine...")
    engine = RealtimeInferenceEngine(window_size=30)

    # Create dummy face frame & sample audio waveform
    dummy_frame = np.full((480, 640, 3), 160, dtype=np.uint8)
    sample_audio = np.sin(2 * np.pi * 350 * np.linspace(0, 0.5, 8000)).astype(np.float32)

    # 4. Run Inference
    result = engine.process_frame(
        image=dummy_frame,
        audio_signal=sample_audio,
        transcript_text=transcript,
    )

    # 5. Display Formatted Output
    emotions_list = ["Neutral", "Happy", "Sad", "Angry", "Surprised", "Fearful", "Disgusted"]
    emo_str = emotions_list[0] if isinstance(result['primary_emotion'], float) else str(result['primary_emotion'])

    print("\n------------------------------------------------------------------")
    print("  AI-GENERATED SCREENING & RISK ASSESSMENT REPORT")
    print("------------------------------------------------------------------")
    print(f"Estimated Stress Score : {result['stress_score']}%")
    print(f"Risk Category          : {result['stress_level']} Risk")
    print(f"Fatigue Level          : {result['fatigue_score']}")
    print(f"Attention Level        : {result['attention_score']}")
    print(f"Primary Emotion        : {emo_str}")
    print(f"Confidence Rating      : {result['quality']['confidence_score']} ({'Reliable' if result['quality']['is_reliable'] else 'Low Quality'})")

    print("\n[MODALITY ATTRIBUTION BREAKDOWN]")
    print(f"  • Vision Stream (Face/Landmarks/Pose) : {result['modality_pcts']['vision_pct']}%")
    print(f"  • Audio Stream (Prosody/MFCCs)        : {result['modality_pcts']['audio_pct']}%")
    print(f"  • Text Stream (NLP Transcript)       : {result['modality_pcts']['text_pct']}%")

    print("\n[TOP SHAP BEHAVIORAL FEATURE INFLUENCES]")
    for item in result["shap_ranks"]:
        sign = "+" if item["attribution"] >= 0 else ""
        print(f"  • {item['feature']:<26} : {sign}{item['attribution']:.2f} (Value: {item['value']})")

    print("------------------------------------------------------------------\n")


if __name__ == "__main__":
    run_sample_demo()
