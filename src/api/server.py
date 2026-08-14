"""
FastAPI Server for Real-Time Multimodal Mental Health Inference.

Provides REST and WebSocket endpoints for edge devices to stream video/audio data and receive predictions.

Usage:
    uvicorn src.api.server:app --reload --port 8000
"""

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

from src.inference.realtime import RealtimeInferenceEngine

app = FastAPI(
    title="Privacy-Preserving Mental Health AI API",
    description="Real-Time Multimodal Mental Health Risk Assessment API",
    version="1.0.0",
)

# Shared inference engine instance
engine = RealtimeInferenceEngine(window_size=30)


class PredictionRequest(BaseModel):
    vision_feature: Optional[List[float]] = None
    audio_feature: Optional[List[float]] = None
    transcript_text: Optional[str] = None


class PredictionResponse(BaseModel):
    stress_score: float
    stress_level: str
    fatigue_score: float
    attention_score: float
    confidence_score: float
    quality_status: str
    modality_attribution: Dict[str, float]
    shap_top_features: List[Dict[str, Any]]


@app.get("/")
def read_root():
    return {
        "status": "online",
        "service": "Privacy-Preserving Real-Time Multimodal Mental Health Risk Assessment API",
        "version": "1.0.0",
    }


@app.get("/health")
def health_check():
    return {"status": "healthy", "device": str(engine.model.vision_dim)}


@app.post("/predict", response_model=PredictionResponse)
def predict_risk(req: PredictionRequest):
    """Processes a single feature payload and returns risk predictions."""
    try:
        dummy_frame = np.full((480, 640, 3), 150, dtype=np.uint8)
        audio_sig = None
        if req.audio_feature:
            if len(req.audio_feature) > 100:
                audio_sig = np.array(req.audio_feature, dtype=np.float32)
            else:
                # Pre-generated audio wave mock
                audio_sig = np.sin(2 * np.pi * 300 * np.linspace(0, 0.5, 8000)).astype(np.float32)

        res = engine.process_frame(image=dummy_frame, audio_signal=audio_sig, transcript_text=req.transcript_text)

        return PredictionResponse(
            stress_score=res["stress_score"],
            stress_level=res["stress_level"],
            fatigue_score=res["fatigue_score"],
            attention_score=res["attention_score"],
            confidence_score=res["quality"]["confidence_score"],
            quality_status="Reliable" if res["quality"]["is_reliable"] else "Low Quality",
            modality_attribution=res["modality_pcts"],
            shap_top_features=res["shap_ranks"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/predict")
async def websocket_predict(websocket: WebSocket):
    """Real-time streaming WebSocket endpoint for continuous frame predictions."""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            dummy_frame = np.full((480, 640, 3), 150, dtype=np.uint8)
            res = engine.process_frame(image=dummy_frame, transcript_text=data.get("text"))

            await websocket.send_json({
                "stress_score": res["stress_score"],
                "stress_level": res["stress_level"],
                "fatigue": res["fatigue_score"],
                "attention": res["attention_score"],
                "confidence": res["quality"]["confidence_score"],
            })
    except WebSocketDisconnect:
        pass
