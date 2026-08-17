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

### A. Run Interactive Streamlit Web Dashboard
```bash
streamlit run dashboard/app.py
```
Open browser at: **`http://localhost:8501`**

### B. Run FastAPI REST & WebSocket Edge API Server
```bash
uvicorn src.api.server:app --reload --port 8000
```
- Interactive Swagger API Docs: **`http://127.0.0.1:8000/docs`**
- WebSocket Endpoint: `ws://127.0.0.1:8000/ws/predict`

### C. Run Terminal Sample Predictor Demo
```bash
python demo_predict_sample.py
```

### D. Run Automated Research Experiments (E1 to E12)
```bash
python experiments/run_experiments.py --mode fast
```
Outputs exported to `outputs/metrics/experiment_summary.csv`.

### E. Generate 300 DPI Publication Paper Figures
```bash
python experiments/generate_paper_plots.py
```
Figures saved to `outputs/plots/` (`fig1_modality_ablation.png`, `fig2_federated_heterogeneity.png`, `fig3_privacy_utility_tradeoff.png`, `fig4_edge_latency_optimization.png`).

### F. Export Academic Research Paper Draft
```bash
python experiments/export_paper_report.py
```
Manuscript draft generated at `outputs/paper_draft.md`.

---

## 4. Running Docker Stack

Launch the containerized Streamlit Dashboard and FastAPI server simultaneously:
```bash
docker-compose up --build
```
- Dashboard: `http://localhost:8501`
- API Server: `http://localhost:8000`

---

## 5. Running the Unit Test Suite (39 Tests)

Run the full automated test suite across all 13 modules:
```bash
python -m unittest discover -s tests -v
```

---

## 6. Dataset Integration (DAIC-WOZ / RECOLA / AVEC)

Convert raw public datasets into subject-partitioned format:
```bash
python -m src.data.ingest_daic_woz --data-dir path/to/daic_woz --output-dir data/processed
```
