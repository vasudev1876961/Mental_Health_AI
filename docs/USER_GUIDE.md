# Developer & User Guide: Privacy-Preserving Multimodal Mental Health Risk Assessment Using Federated Learning

Welcome to the comprehensive developer and user guide for the **Privacy-Preserving Real-Time Multimodal Mental Health Risk Assessment Platform (Zenith Phase 10)**.

---

## 1. System Overview & Architecture

This platform combines computer vision, physiological contact-free pulse extraction (rPPG HRV), vocal acoustics, RoBERTa speech NLP, bi-directional cross-modal co-attention, temporal sequence modeling, federated learning (FedAvg, FedProx, FedPer, SecAgg, FedAsync, Clustered FL, FedDG), differential privacy, multi-level explainable AI, clinician-AI decision support, crisis safety guardrails, and dynamic battery-aware runtime scheduling into a unified edge system.

```
                         REAL-TIME EDGE CLIENT
                                   │
         ┌─────────────────────────┼─────────────────────────┐
         │                         │                         │
         ▼                         ▼                         ▼
       VIDEO                     AUDIO                     TEXT
         │                         │                         │
  Face Mesh (468)             Vocal Prosody             Speech-to-Text
  EAR, MAR, Head Pose         Pitch/F0 & MFCCs             RoBERTa
  rPPG CHROM / HRV                 │                         │
         │                         │                         │
         └─────────────────────────┼─────────────────────────┘
                                   ▼
              Bi-Directional Cross-Modal Co-Attention & DGF
                                   │
                                   ▼
                       Temporal Transformer (BiLSTM)
                                   │
                                   ▼
                 Multimodal Mental Health Risk Estimator
                  (Continuous 0-100 & Low/Med/High Risk)
                                   │
     ┌─────────────────────────────┼─────────────────────────────┐
     ▼                             ▼                             ▼
Explainable AI           Clinician-AI Copilot             Edge Safety & FL
- SHAP Attributions      - DSM-5 MDD/GAD Criteria         - Real-Time Crisis Guardrails
- Grad-CAM Heatmaps      - PHQ-9 & GAD-7 Estimates        - Differential Privacy (Opacus)
- Counterfactual Recourse- Structured SOAP Notes          - Federated Learning (FedPer/SecAgg)
                         - HL7 FHIR DiagnosticReport      - Byzantine Defense (Multi-Krum)
                         - SHA-256 Sign-off Audit Hash    - Clustered FL & FedDG (MMD)
                                                          - Dynamic INT8 & Battery Scheduler
```

---

## 2. Installation & Quick Setup

### Prerequisites
- Python 3.10, 3.11, or 3.12
- Git

### Installation Steps
```bash
# Clone the repository
git clone https://github.com/vasudev1876961/Mental_Health_AI.git
cd AI-MENTAL

# Install dependencies
pip install -r requirements.txt
```

---

## 3. How to Launch System Features

### A. Master Interactive Terminal CLI
```bash
python launch_all.py
```
Provides an interactive numerical menu to launch the web dashboard, API server, terminal predictor, test suite, experiments, and paper figure generators in one click.

### B. Run Interactive Streamlit Web Dashboard (8 Distinct Tabs)
```bash
streamlit run dashboard/app.py
```
Open your browser at: **`http://localhost:8501`**
- **Tab 1: 🎥 Real-Time Multimodal Assessment**: Live webcam video feed, 468-point 3D face mesh, rPPG heart rate, EAR/MAR gauges, stress timeline, SHAP attributions, Grad-CAM heatmaps, and distribution-free conformal prediction intervals.
- **Tab 2: 🤝 Personalized Federated Learning (FedPer)**: Edge client adaptation simulator preserving personal heads while synchronizing global shared backbones.
- **Tab 3: 🔒 Cryptographic Secure Aggregation (SecAgg)**: Zero-sum pairwise secret sharing verifying exact global summation without raw gradient leakage.
- **Tab 4: 🌌 Multimodal Contrastive Alignment (InfoNCE)**: Hyper-spherical representation space aligning video, audio, and text embeddings against missing sensor modalities.
- **Tab 5: 🛡️ Byzantine Defense & Conformal Uncertainty**: Multi-Krum / Trimmed Mean defense against adversarial edge poisoning attacks and conformal prediction set validation.
- **Tab 6: 🚀 Phase 8: Physiological rPPG & Edge Optimization**: Contact-free CHROM pulse wave, HRV frequency spectrum, algorithmic counterfactual recourse interventions, FedAsync staleness simulator, and ONNX Runtime speedup.
- **Tab 7: 🧬 Phase 9: Clustered FL, Bi-CoAttention & Clinical Active Learning**: Cosine parameter clustering across psychiatric phenotypes, cross-modal co-saliency heatmaps, FedActive clinician review budget curves, and Pareto ROC operating frontiers.
- **Tab 8: 🩺 Phase 10: Clinician Copilot, FedDG & Crisis Safeguards**: Automated psychiatric SOAP note generator, estimated PHQ-9 & GAD-7 severity scoring, DSM-5 symptom radar, HL7 FHIR DiagnosticReport JSON export, tamper-proof SHA-256 audit hashing, 988 Crisis Lifeline escalation, FedDG cross-domain MMD simulator, and edge battery scheduler.

### C. Run FastAPI REST & WebSocket Edge API Server (v2.4.0)
```bash
uvicorn src.api.server:app --reload --port 8000
```
- Interactive OpenAPI / Swagger Docs: **`http://127.0.0.1:8000/docs`**
- WebSocket Real-Time Stream: `ws://127.0.0.1:8000/ws/predict`
- Key REST Endpoints:
  - `POST /copilot/generate_soap_report`
  - `POST /safety/crisis_triage`
  - `POST /federated/domain_generalization`
  - `POST /edge/adaptive_profile`
  - `POST /calibration/pareto/threshold`
  - `POST /active/query/sample`
  - `POST /fusion/coattention/saliency`
  - `POST /physiological/rppg/extract`
  - `POST /explainability/counterfactual/recourse`

### D. Run Quick Sample Terminal Predictor Demo
```bash
python demo_predict_sample.py
```

### E. Run Automated Research Experiments (E1 to E35)
```bash
python experiments/run_experiments.py --mode fast
```
Outputs saved to `outputs/metrics/experiment_summary.csv` and `outputs/metrics/experiment_summary.json`.

### F. Generate 300 DPI Publication Paper Figures (Figures 1 to 22)
```bash
python experiments/generate_paper_plots.py
```
Figures exported to `outputs/plots/`:
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
- `fig11_rppg_hrv_pulse_wave.png`
- `fig12_counterfactual_recourse_interventions.png`
- `fig13_fedasync_staleness_convergence.png`
- `fig14_onnx_runtime_speedup.png`
- `fig15_clustered_fl_similarity.png`
- `fig16_coattention_saliency_matrix.png`
- `fig17_fedactive_budget_curve.png`
- `fig18_clinical_pareto_frontier.png`
- `fig19_feddg_domain_invariance.png`
- `fig20_copilot_clinical_concordance.png`
- `fig21_edge_battery_tradeoff.png`
- `fig22_crisis_safety_roc.png`

### G. Export Academic Research Paper Draft & LaTeX Manuscript
```bash
python experiments/export_paper_report.py
python experiments/build_latex_paper.py
```
Generates `outputs/paper_draft.md` and publication-ready IEEE conference LaTeX file `outputs/paper_ieee.tex`.

---

## 4. Running Docker Stack

Launch containerized Streamlit Dashboard and FastAPI server:
```bash
docker-compose up --build
```
- Dashboard: `http://localhost:8501`
- API Server: `http://localhost:8000`

---

## 5. Running the Complete Automated Unit Test Suite

Run the full automated test suite across all 21 test modules:
```bash
pytest tests/ -v
```

---

## 6. Dataset Integration (DAIC-WOZ / RECOLA / AVEC)

Convert raw public clinical datasets into subject-partitioned format:
```bash
python -m src.data.ingest_daic_woz --data-dir path/to/daic_woz --output-dir data/processed
```
