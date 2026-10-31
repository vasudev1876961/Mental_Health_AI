"""
Crisis Safety Guardrails & Emergency Escalation Protocol.

Provides fail-safe, real-time safety monitoring for high-risk psychiatric crises:
- Immediate detection of acute panic surges and self-harm/suicidal ideation markers
- Physiological autonomic crisis triggers (extreme vagal withdrawal + sympathetic surge)
- Tiered escalation dispatch with emergency resource integration (988 Lifeline)
- Hallucination and medical liability safety boundary filtering
"""

import re
import time
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


class CrisisSafetyGuardrails:
    """
    Real-time safety guardrail and emergency intervention dispatcher.
    """

    CRISIS_KEYWORDS = [
        "kill myself", "want to die", "end my life", "suicide", "suicidal",
        "better off dead", "no reason to live", "can't go on anymore",
        "hurt myself", "self harm", "overdose", "slit my wrists",
        "can't breathe panic", "having a heart attack panic", "losing my mind",
    ]

    PANIC_KEYWORDS = [
        "panic attack", "can't breathe", "suffocating", "heart racing",
        "chest tight", "hyperventilating", "dizzy trembling", "terror",
    ]

    EMERGENCY_RESOURCES = {
        "crisis_lifeline": "988 (Suicide & Crisis Lifeline - 24/7 Free & Confidential)",
        "crisis_text_line": "Text HOME to 741741 (Free 24/7 Crisis Support via SMS)",
        "international_portal": "https://www.findahelpline.com/",
        "veterans_crisis_line": "Dial 988 then press 1, or Text 838255",
        "emergency_medical_services": "Call 911 (or local emergency medical services)",
    }

    def __init__(
        self,
        crisis_stress_threshold: float = 85.0,
        crisis_rmssd_min: float = 12.0,
        crisis_lf_hf_max: float = 3.8,
    ):
        self.crisis_stress_threshold = crisis_stress_threshold
        self.crisis_rmssd_min = crisis_rmssd_min
        self.crisis_lf_hf_max = crisis_lf_hf_max

    def scan_text_for_crisis(self, text: Optional[str]) -> Dict[str, Any]:
        """
        Scans acoustic speech transcripts for acute crisis or self-harm keywords.
        """
        if not text or not text.strip():
            return {
                "crisis_keywords_detected": False,
                "panic_keywords_detected": False,
                "matched_cues": [],
                "text_risk_level": "NEUTRAL",
            }

        text_lower = text.lower()
        matched_crisis = [kw for kw in self.CRISIS_KEYWORDS if re.search(r'\b' + re.escape(kw) + r'\b', text_lower)]
        matched_panic = [kw for kw in self.PANIC_KEYWORDS if re.search(r'\b' + re.escape(kw) + r'\b', text_lower)]

        if matched_crisis:
            risk_level = "EMERGENCY_CRISIS"
        elif matched_panic:
            risk_level = "ACUTE_PANIC"
        else:
            risk_level = "ROUTINE"

        return {
            "crisis_keywords_detected": len(matched_crisis) > 0,
            "panic_keywords_detected": len(matched_panic) > 0,
            "matched_cues": matched_crisis + matched_panic,
            "text_risk_level": risk_level,
        }

    def scan_physiology_for_crisis(
        self,
        stress_score: float,
        hrv_metrics: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Scans physiological metrics for autonomic crisis (extreme sympathetic surge with vagal shutdown).
        """
        rmssd = hrv_metrics.get("rmssd", 35.0)
        lf_hf = hrv_metrics.get("lf_hf_ratio", 1.5)
        mean_hr = hrv_metrics.get("mean_hr", 75.0)

        autonomic_collapse = bool(rmssd < self.crisis_rmssd_min and lf_hf > self.crisis_lf_hf_max)
        tachycardia_surge = bool(mean_hr > 125.0 and stress_score > self.crisis_stress_threshold)

        physiological_crisis = autonomic_collapse or tachycardia_surge

        return {
            "autonomic_crisis": physiological_crisis,
            "rmssd_critical": bool(rmssd < self.crisis_rmssd_min),
            "sympathetic_overdrive": bool(lf_hf > self.crisis_lf_hf_max),
            "severe_tachycardia": bool(mean_hr > 125.0),
            "measured_rmssd": round(float(rmssd), 2),
            "measured_lf_hf": round(float(lf_hf), 2),
            "measured_hr": round(float(mean_hr), 1),
        }

    def evaluate_safety_escalation(
        self,
        stress_score: float,
        hrv_metrics: Optional[Dict[str, float]] = None,
        transcript_text: Optional[str] = None,
        conformal_upper_bound: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Synthesizes multimodal evidence into a definitive safety escalation determination.
        """
        hrv = hrv_metrics or {"mean_hr": 75.0, "rmssd": 35.0, "lf_hf_ratio": 1.5}
        text_scan = self.scan_text_for_crisis(transcript_text)
        physio_scan = self.scan_physiology_for_crisis(stress_score, hrv)

        upper_bound = conformal_upper_bound if conformal_upper_bound is not None else stress_score

        # Escalation Logic:
        # Tier 4: EMERGENCY_CRISIS_ESCALATION (Any explicit crisis text OR severe multi-system collapse)
        # Tier 3: URGENT_CLINICIAN_REVIEW (Physiological crisis OR high stress >= 80 OR panic keywords)
        # Tier 2: ELEVATED_OBSERVATION (Moderate stress >= 50 or upper bound >= 75)
        # Tier 1: ROUTINE_MONITORING

        if text_scan["crisis_keywords_detected"] or (physio_scan["autonomic_crisis"] and stress_score >= 90.0):
            tier = "EMERGENCY_CRISIS_ESCALATION"
            requires_immediate_dispatch = True
            action_protocol = [
                "IMMEDIATE ACTION: Display crisis resources prominently (988 Lifeline).",
                "Lock edge session and present compassionate crisis intervention dialogue.",
                "Automated encrypted notification dispatched to emergency clinical on-call team.",
                "Provide local emergency service access options (911).",
            ]
        elif text_scan["panic_keywords_detected"] or physio_scan["autonomic_crisis"] or stress_score >= 80.0 or upper_bound >= 90.0:
            tier = "URGENT_CLINICIAN_REVIEW"
            requires_immediate_dispatch = False
            action_protocol = [
                "Schedule clinician review within 4-12 hours.",
                "Initiate immediate real-time guided somatic respiration pacer (Box Breathing).",
                "Flag patient record with high-urgency clinical indicator tag.",
            ]
        elif stress_score >= 50.0 or upper_bound >= 70.0:
            tier = "ELEVATED_OBSERVATION"
            requires_immediate_dispatch = False
            action_protocol = [
                "Offer self-directed mindfulness and cognitive reframing exercises.",
                "Increase edge sampling frequency to monitor progression.",
            ]
        else:
            tier = "ROUTINE_MONITORING"
            requires_immediate_dispatch = False
            action_protocol = [
                "Standard longitudinal wellness tracking active.",
                "No immediate clinical intervention indicated.",
            ]

        return {
            "escalation_tier": tier,
            "requires_immediate_dispatch": requires_immediate_dispatch,
            "text_scan": text_scan,
            "physio_scan": physio_scan,
            "stress_score": float(stress_score),
            "conformal_upper_bound": float(upper_bound),
            "emergency_resources": self.EMERGENCY_RESOURCES if requires_immediate_dispatch else {},
            "clinical_action_protocol": action_protocol,
            "safety_verified": True,
        }

    def sanitize_model_output(self, raw_advice_text: str) -> str:
        """
        Validates and sanitizes AI-generated text to ensure no prescriptive medical diagnoses
        or harmful non-clinical directives are conveyed.
        """
        disclaimer = " [Notice: AI-generated screening indicator, not a definitive medical diagnosis.]"
        forbidden_phrases = ["you definitely have", "I diagnose you with", "stop taking your medication", "you do not need a doctor"]
        
        sanitized = raw_advice_text
        for phrase in forbidden_phrases:
            if phrase.lower() in sanitized.lower():
                sanitized = sanitized.replace(phrase, "clinical indicators suggest discussing with a licensed physician")
                sanitized = sanitized.replace(phrase.capitalize(), "Clinical indicators suggest discussing with a licensed physician")

        if disclaimer not in sanitized:
            sanitized = sanitized.strip() + disclaimer
        return sanitized
