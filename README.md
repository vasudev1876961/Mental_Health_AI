# Privacy-Preserving Real-Time Multimodal Mental Health Risk Assessment Using Federated Learning

[![CI - Multimodal Mental Health FL Pipeline](https://github.com/vasudev1876961/Mental_Health_AI/actions/workflows/ci.yml/badge.svg)](https://github.com/vasudev1876961/Mental_Health_AI/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A modular, privacy-preserving deep learning platform that analyzes real-time video, audio, and textual behavioral indicators to estimate stress risk, fatigue, attention, and emotional states without transmitting raw private user data off edge devices.

---

## 📖 Complete Documentation & Guides

- **[Detailed User & Developer Guide](docs/USER_GUIDE.md)**: Full guide covering installation, API endpoints, Docker, dataset adapters, and FL setups.
- **[Academic Paper Draft](outputs/paper_draft.md)**: Manuscript draft complete with math equations, abstract, and experimental benchmarks.
- **[System Walkthrough](walkthrough.md)**: Stage-by-stage implementation and verification summary.

---

## Architecture Overview

```
                         REAL-TIME CLIENT
                                │
             ┌──────────────────┼──────────────────┐
             │                  │                  │
             ▼                  ▼                  ▼
           VIDEO              AUDIO              TEXT
             │                  │                  │
      Face Detection        Prosody/MFCC       Speech-to-Text
             │                  │                  │
      Face Landmarks        Audio Features       RoBERTa
             │                  │                  │
     Behavioral Features        │                  │
             │                  │                  │
             └──────────────────┼──────────────────┘
                                ▼
                       Multimodal Fusion
                                │
                                ▼
                      Temporal Transformer
                                │
                                ▼
                      Stress Risk Estimator
                   (Continuous 0-100 & L/M/H)
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
                   XAI                 Local Training
             (SHAP/Grad-CAM)                │
                                            ▼
                                   Differential Privacy
                                            │
                                            ▼
                                    Federated Learning
                                   (FedAvg & FedProx)
```

---

## Key Capabilities

- **Real-Time Vision Pipeline**: Facial detection, 3D face mesh (468 landmarks), Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR), 3D head pose (pitch/yaw/roll), gaze tracking, and affect estimation.
- **Multimodal Audio & NLP**: Pitch/F0, energy, speech rate, MFCCs, local Speech-to-Text (STT), and RoBERTa semantic text embeddings.
- **Multimodal Contrastive Alignment (InfoNCE)**: Self-supervised representation alignment across Vision, Audio, and Text for robustness against missing sensor channels.
- **Temporal Modeling**: Sliding-window BiLSTM / GRU / Temporal Transformer modeling micro-expressions and stress progression.
- **Dynamic Cross-Modal Imputation**: Generative synthesis reconstructing missing sensor modalities (e.g. occluded camera or muted microphone) via cross-modal attention instead of naive zero-masking.
- **Personalized Federated Learning (FedPer)**: Edge client adaptation preserving subject-specific baseline heads while synchronizing global representation backbones.
- **Cryptographic Secure Aggregation (SecAgg)**: Zero-sum pairwise secret sharing preventing server-side gradient inspection.
- **Byzantine-Robust Defense (Multi-Krum, Trimmed Mean, Median)**: Distance-based outlier filtering and coordinate-wise trimming defending against adversarial edge poisoning attacks (sign-flip, gaussian noise).
- **Distribution-Free Conformal Prediction**: Finite-sample statistically guaranteed confidence intervals ($P(Y \in [L, U]) \ge 1 - \alpha$) and discrete classification prediction sets without parametric assumptions.
- **Continual Learning (EWC)**: Elastic Weight Consolidation penalizing catastrophic forgetting of personal baseline affective states during ongoing edge monitoring.
- **Differential Privacy & Security**: PyTorch/Opacus clipping and noise injection with formal $(\epsilon, \delta)$ accounting.
- **Multi-Level XAI**: SHAP behavioral feature importance, Grad-CAM facial heatmaps, and Modality Attribution breakdown (Vision vs Audio vs Text %).
- **Interactive Web Dashboard**: 5-tab Streamlit UI with live webcam feed, real-time gauges, stress timeline, XAI panels, FedPer simulator, SecAgg verification, Contrastive visualizer, and Byzantine Defense & Conformal Uncertainty testbed.
- **FastAPI Edge Server**: High-throughput REST API & WebSocket endpoints for streaming predictions, Byzantine-robust aggregation, conformal uncertainty bounds, and dynamic sensor imputation.

---

## Quick Start Commands

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run interactive Streamlit Dashboard (5 Tabs)
streamlit run dashboard/app.py

# 3. Run terminal sample predictor demo
python demo_predict_sample.py

# 4. Run automated E1-E20 research experiments
python experiments/run_experiments.py --mode fast

# 5. Generate 300 DPI research figures (10 publication plots) & paper drafts (Markdown + LaTeX)
python experiments/generate_paper_plots.py
python experiments/export_paper_report.py
python experiments/build_latex_paper.py

# 6. Run full unit test suite (54 tests across 18 modules)
python -m unittest discover -s tests
```

---

## Ethical & Scientific Disclaimer

Outputs generated by this system are **AI-generated screening/risk indicators** intended for research and early warning support, **NOT clinical diagnostic assessments**.
