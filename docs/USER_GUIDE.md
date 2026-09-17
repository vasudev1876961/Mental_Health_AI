# Developer & User Guide: Privacy-Preserving Multimodal Mental Health Risk Assessment Using Federated Learning

Welcome to the official developer and user guide for the **Privacy-Preserving Real-Time Multimodal Mental Health Risk Assessment System**.

---

## 1. System Overview & Architecture

This system combines computer vision, audio processing, text NLP, temporal sequence modeling, multimodal fusion, federated learning, differential privacy, and explainable AI into a unified platform.

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

## 2. Installation & Quick Setup

### Prerequisites
- Python 3.10, 3.11, or 3.12
- Git

### Installation Steps
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/AI-MENTAL.git
cd AI-MENTAL

# Install dependencies
pip install -r requirements.txt
```

---

## 3. How to Launch System Features

### A. Run Interactive Streamlit Web Dashboard (5 Tabs)
```bash
streamlit run dashboard/app.py
```
Open browser at: **`http://localhost:8501`**
- **Tab 1**: Real-Time Multimodal Assessment (Vision 468 mesh, Audio MFCC, NLP RoBERTa, SHAP XAI, Grad-CAM, and 90% Conformal Interval bounds)
- **Tab 2**: Personalized Federated Learning (FedPer client profiles & head adaptation)
- **Tab 3**: Cryptographic Secure Aggregation (SecAgg zero-sum noise cancellation)
- **Tab 4**: Self-Supervised Contrastive Alignment (InfoNCE cosine hyper-sphere)
- **Tab 5**: Byzantine Poisoning Defense, Conformal Bounds, Dynamic Sensor Imputation & Continual Learning (EWC)

### B. Run FastAPI REST & WebSocket Edge API Server
```bash
uvicorn src.api.server:app --reload --port 8000
```
- Interactive Swagger API Docs: **`http://127.0.0.1:8000/docs`**
- WebSocket Streaming Endpoint: `ws://127.0.0.1:8000/ws/predict`
- Byzantine Defense Endpoint: `POST /defense/byzantine/aggregate`
- Conformal Prediction Endpoint: `POST /uncertainty/conformal/predict`
- Sensor Imputation Endpoint: `POST /fusion/impute`

### C. Run Terminal Sample Predictor Demo
```bash
python demo_predict_sample.py
```

### D. Run Automated Research Experiments (E1 to E20)
```bash
python experiments/run_experiments.py --mode fast
```
Outputs exported to `outputs/metrics/experiment_summary.csv` and `outputs/metrics/experiment_summary.json`.

### E. Generate 300 DPI Publication Paper Figures (Fig 1 to 10)
```bash
python experiments/generate_paper_plots.py
```
Figures saved to `outputs/plots/`:
- `fig1_modality_ablation.png`
- `fig2_federated_heterogeneity.png`
- `fig3_privacy_utility_tradeoff.png`
- `fig4_edge_latency_optimization.png`
- `fig5_personalized_fedper.png`
- `fig6_contrastive_alignment_matrix.png`
- `fig7_secagg_noise_cancellation.png`
- `fig8_byzantine_defense_robustness.png`
- `fig9_conformal_coverage_calibration.png`
- `fig10_continual_ewc_forgetting.png`

### F. Export Academic Research Paper Draft & LaTeX Manuscript
```bash
python experiments/export_paper_report.py
python experiments/build_latex_paper.py
```
Manuscript drafts generated at `outputs/paper_draft.md` and `outputs/paper_ieee.tex`.

---

## 4. Running Docker Stack

Launch the containerized Streamlit Dashboard and FastAPI server simultaneously:
```bash
docker-compose up --build
```
- Dashboard: `http://localhost:8501`
- API Server: `http://localhost:8000`

---

## 5. Running the Unit Test Suite (55 Tests)

Run the full automated test suite across all 18 modules:
```bash
python -m unittest discover -s tests -v
```

---

## 6. Dataset Integration (DAIC-WOZ / RECOLA / AVEC)

Convert raw public datasets into subject-partitioned format:
```bash
python -m src.data.ingest_daic_woz --data-dir path/to/daic_woz --output-dir data/processed
```
