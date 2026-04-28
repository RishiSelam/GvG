---
title: GvG Defense API
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: docker
app_file: app.py
pinned: false
---

# GvG: GAN-vs-GAN Adversarial Defense for Intrusion Detection Systems

A full-stack adversarial training platform that pits a **conditional GAN attacker** against a **Transformer-LSTM defender** to build a robust Intrusion Detection System (IDS). The project includes a Python ML backend (FastAPI), a React/TypeScript SOC-style dashboard, and cloud deployment support for Hugging Face Spaces + Vercel.

---

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Architecture](#architecture)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Running the Pipeline](#running-the-pipeline)
- [FastAPI Backend](#fastapi-backend)
- [React Frontend Dashboard](#react-frontend-dashboard)
- [Deployment](#deployment)
- [API Reference](#api-reference)
- [Artifacts & Outputs](#artifacts--outputs)
- [How the Models Work](#how-the-models-work)
- [Custom Input Scoring](#custom-input-scoring)
- [Utility Scripts](#utility-scripts)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [Troubleshooting](#troubleshooting)

---

GvG is an end-to-end intrusion-detection project built around adversarial training. The repository now contains three working layers:

<<<<<<< HEAD
This project implements an end-to-end adversarial intrusion detection workflow:

1. **Preprocess** raw CICIDS2017 network flow CSVs into clean, balanced, scaled, and sequenced datasets.
2. **Train a baseline Transformer-LSTM IDS** (defender) on clean traffic.
3. **Train a conditional GAN** (attacker) to generate evasive attack traffic.
4. **Adversarially fine-tune** the generator against the IDS to produce highly stealthy samples.
5. **Retrain the IDS** on clean + GAN-generated attacks to produce a robust model.
6. **Evaluate** baseline vs robust performance on clean and adversarial test sets.
7. **Serve predictions** via a FastAPI backend and visualize everything through a React dashboard.

### Single-command execution
=======
- a CICIDS2017 preprocessing pipeline
- a completed training pipeline that produces baseline and robust IDS artifacts
- a FastAPI backend plus React dashboard for metrics, artifact exploration, and live scoring

## Current Project Status
>>>>>>> dfc8aa63c8970aa6b161418c11243fba5f167077

The repository is beyond the planning stage and already includes generated artifacts from a completed run.

<<<<<<< HEAD
This runs preprocessing → training → adversarial evaluation → custom-input scoring in one shot.

---
=======
- Training manifest available in `artifacts/training/training_manifest.json`
- Baseline and robust IDS checkpoints available in `artifacts/models/`
- cGAN attacker checkpoint available in `artifacts/models/attacker_cgan.pt`
- Generated adversarial samples available in `artifacts/generated/`
- EDA summaries and plots available in `artifacts/eda/`
- Custom input scoring sample outputs available in `artifacts/custom_input/`

Current recorded artifact summary:

- `66` engineered features
- `99 / 33 / 33` tabular train, validation, and test rows
- `90 / 24 / 24` sequence train, validation, and test windows
- `15` label classes including `BENIGN`
- `18` IDS epochs and `40` GAN epochs in the tracked run

Current recorded test metrics from `artifacts/training/metrics_summary.csv`:

- Baseline tabular accuracy: `96.97%`
- Robust tabular accuracy: `96.97%`
- Baseline sequence accuracy: `91.67%`
- Robust sequence accuracy: `91.67%`
- Baseline adversarial sequence accuracy: `95.45%`
- Robust adversarial sequence accuracy: `100%`

## What The Project Implements

The codebase follows the GAN-vs-GAN workflow conceptually, but the implementation is a practical approximation rather than the exact deep architecture from the original presentation.

Current implementation:
>>>>>>> dfc8aa63c8970aa6b161418c11243fba5f167077

- Defender IDS: `HybridIDSModel`
  - tabular branch
  - sequence branch
  - saved as `baseline_ids.pt` and `robust_ids.pt`
- Attacker model: `AdversarialTrafficGenerator`
  - conditional GAN style generator for malicious traffic mutation
  - adversarial fine-tuning rounds using IDS feedback
  - saved as `attacker_cgan.pt`
- Orchestration: `AdversarialTrainingPipeline`
  - baseline fit
  - generator fit
  - adversarial fine-tuning
  - augmented retraining
  - evaluation and artifact export

<<<<<<< HEAD
Traditional IDS pipelines perform well on known traffic distributions but degrade when attack behavior changes. Adversarial generation methods can craft malicious flows that are nearly indistinguishable from benign traffic.

This project addresses that gap by:

- Training a deep-learning defender (Tabular MLP + Transformer-LSTM sequence head)
- Training a conditional GAN attacker that learns to evade the defender
- Using adversarial retraining to shift the decision boundary and improve robustness
- Providing a live simulation endpoint (`/simulate_evasion`) that shows how the GAN morphs attacks and suggests countermeasures

---

## Architecture

The system is split into two adversarial sides:

### Attacker Side (Conditional GAN)

- **ConditionalGenerator**: takes noise vector `z` + class label embedding → produces synthetic attack feature vectors via `tanh` activation
- **ConditionalDiscriminator**: classifies real vs generated features conditioned on class labels
- **Adversarial Fine-Tuning**: freezes the IDS and optimizes the generator to simultaneously fool the discriminator, reduce IDS detection probability, and stay near class centroids

### Defender Side (Hybrid IDS)

- **TabularIDSHead**: 3-layer MLP with LayerNorm + GELU + Dropout → single logit per flow record
- **TransformerLSTMSequenceHead**: Linear projection → Transformer Encoder (multi-head self-attention) → Bidirectional LSTM → mean pooling → MLP classifier → single logit per window
- **Training**: BCE-with-logits loss with class-imbalance `pos_weight`, AdamW optimizer

### Orchestration

- **AdversarialTrainingPipeline** (`training/pipeline.py`): baseline training → cGAN fitting → adversarial fine-tuning → data augmentation → robust retraining → evaluation
- **FastAPI** (`app.py`): serves prediction, evasion simulation, and artifact endpoints
- **React Dashboard** (`frontend/`): visualizes the full pipeline, metrics, training curves, and live scoring

---

## Technology Stack

### Backend

| Component | Technology |
|---|---|
| Deep Learning | PyTorch (MLP, Transformer, LSTM, cGAN) |
| ML Utilities | scikit-learn (metrics, preprocessing, splitting) |
| Data Processing | pandas, NumPy |
| API Framework | FastAPI + Uvicorn |
| Containerization | Docker (Python 3.11-slim) |
| Deployment | Hugging Face Spaces (port 7860) |

### Frontend

| Component | Technology |
|---|---|
| Framework | React 18 + TypeScript |
| Build Tool | Vite 6 |
| Styling | Tailwind CSS v4 |
| Charts | Recharts |
| Animations | Motion (Framer Motion) |
| Routing | React Router 7 |
| Icons | Lucide React |
| Deployment | Vercel |

---

## Project Structure

```
GvG/
├── main.py                          # Single entrypoint: preprocess → train → evaluate → score
├── app.py                           # FastAPI backend (prediction, evasion, artifact endpoints)
├── __init__.py                      # Root package exports
├── preprocessing_CICIDS2017.py      # 11-stage CICIDS2017 preprocessing pipeline (809 lines)
├── Dockerfile                       # Docker config for HF Spaces (python:3.11-slim, port 7860)
├── requirements.txt                 # Python dependencies (torch, fastapi, scikit-learn, etc.)
├── .gitignore
│
├── training/                        # Core ML training package
│   ├── __init__.py                  # Exports PipelineConfig, AdversarialTrainingPipeline
│   ├── config.py                    # PipelineConfig dataclass (all hyperparameters & paths)
│   ├── data_loader.py               # PreprocessedDataLoader → PreprocessedDataBundle
│   ├── ids_model.py                 # TabularIDSHead, TransformerLSTMSequenceHead, HybridIDSModel
│   ├── attacker_generator.py        # ConditionalGenerator, ConditionalDiscriminator, AdversarialTrafficGenerator
│   ├── evaluation.py                # MetricsRecorder (accuracy, precision, recall, F1, ROC-AUC, FPR, FNR)
│   └── pipeline.py                  # AdversarialTrainingPipeline (5-step orchestration)
│
├── custom_input/                    # Custom CSV scoring module
│   ├── __init__.py
│   ├── runner.py                    # CustomInputRunner (align features, score, save)
│   └── custom_test.py              # Standalone scoring entrypoint
│
├── eda.py                           # Post-training EDA: load metrics, plot loss curves & summary
├── regenerate_eda_plots.py          # Regenerate high-quality EDA plots from saved CSVs
├── pretty_run.py                    # Orchestrates: preprocessing → training → EDA in sequence
├── run_preprocessing.py             # Standalone preprocessing with skip-if-exists logic
├── run_training.py                  # Standalone training with skip-if-exists logic
├── test_pipeline.py                 # Validation test: builds synthetic input, scores, verifies output
├── upload_to_hf.py                  # Upload backend to Hugging Face Spaces via huggingface_hub
│
├── GvG_Preprocessed.ipynb           # Jupyter notebook for interactive exploration
├── KNOWLEDGEBASE.md                 # Deep technical reference (math, internals, speaking points)
├── README-API-integration.md        # Frontend ↔ backend integration guide
│
├── frontend/                        # React/TypeScript SOC dashboard
│   ├── package.json                 # Dependencies (React 18, Recharts, Motion, Tailwind v4)
│   ├── vite.config.ts               # Vite config with /api proxy to localhost:8000
│   ├── vercel.json                  # Vercel SPA rewrite rules
│   ├── index.html
│   ├── src/
│   │   ├── main.tsx                 # React entry point
│   │   ├── styles/                  # Global CSS
│   │   └── app/
│   │       ├── App.tsx              # Root component
│   │       ├── routes.tsx           # React Router config (5 routes)
│   │       ├── layouts/
│   │       │   └── RootLayout.tsx   # Sidebar navigation layout
│   │       ├── components/          # Reusable UI (MetricCard, GlassCard, ThreatIntelPanel, etc.)
│   │       ├── lib/
│   │       │   ├── api.ts           # API service layer (typed fetch wrappers for all endpoints)
│   │       │   ├── mockData.ts      # Fallback data when backend is offline
│   │       │   ├── utils.ts         # Utility functions
│   │       │   └── toast.ts         # Notification helpers
│   │       └── pages/
│   │           ├── Dashboard.tsx    # System overview, key metrics, performance charts
│   │           ├── Architecture.tsx # GAN-vs-GAN framework explanation
│   │           ├── TrainingLab.tsx  # Training monitor, loss curves, generator feedback
│   │           ├── Analytics.tsx    # Metrics comparison, confusion matrices, reports
│   │           └── LiveDemo.tsx     # Real-time CSV scoring + evasion simulation
│   └── dist/                        # Production build output
│
├── datasets/                        # Raw CICIDS2017 CSVs (gitignored)
│   └── MachineLearningCVE/
├── preprocessed/                    # Processed splits & sequences (gitignored)
│   ├── training_dataset.csv
│   ├── splits/                      # X_train.csv, y_train.csv, etc.
│   └── sequences/                   # sequences_train.npy, etc.
└── artifacts/                       # All generated artifacts
    ├── feature_names.txt            # Canonical feature order
    ├── label_encoder.pkl            # Label ↔ integer mapping
    ├── scaler_standard.pkl          # Fitted StandardScaler
    ├── eda/                         # EDA reports and plots
    │   ├── eda_report.txt
    │   ├── label_distribution.csv
    │   ├── null_counts.csv
    │   ├── numeric_summary.csv
    │   └── plots/                   # PNG visualizations
    ├── models/                      # PyTorch checkpoints (gitignored *.pt)
    │   ├── baseline_ids.pt
    │   ├── robust_ids.pt
    │   └── attacker_cgan.pt
    ├── generated/                   # GAN-generated adversarial CSVs
    │   ├── synthetic_training_attacks.csv
    │   ├── test_adversarial_samples.csv
    │   └── validation_adversarial_round_*.csv
    ├── training/                    # Metrics, reports, manifests
    │   ├── training_manifest.json
    │   ├── metrics_summary.csv
    │   ├── final_report.csv
    │   ├── generator_state.json
    │   ├── generator_feedback.csv
    │   ├── baseline_training_history.csv
    │   ├── robust_training_history.csv
    │   ├── *_metrics.json
    │   ├── *_classification_report.json
    │   └── *_confusion_matrix.csv
    └── custom_input/                # Scored custom CSV outputs
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+ and npm (for the frontend)
- CICIDS2017 dataset in CSV format

### 1. Install Python dependencies
=======
## Repository Layout

```text
GvG/
├── app.py
├── main.py
├── run_preprocessing.py
├── run_training.py
├── preprocessing_CICIDS2017.py
├── training/
│   ├── config.py
│   ├── data_loader.py
│   ├── ids_model.py
│   ├── attacker_generator.py
│   ├── evaluation.py
│   └── pipeline.py
├── custom_input/
│   ├── runner.py
│   └── sample_custom_input.csv
├── artifacts/
│   ├── eda/
│   ├── generated/
│   ├── models/
│   ├── training/
│   └── custom_input/
├── frontend/
│   ├── src/app/pages/
│   └── README.md
└── README-API-integration.md
```

## Backend Workflow

### 1. Preprocessing

`Preprocess` in `preprocessing_CICIDS2017.py`:

- locates CICIDS2017 CSV files or extracts them from `datasets/MachineLearningCSV.zip`
- cleans numeric columns and labels
- balances the dataset
- creates train, validation, and test splits
- builds fixed-length sequence windows
- writes EDA reports, plots, feature names, and encoded labels

### 2. Training

`AdversarialTrainingPipeline` in `training/pipeline.py`:

1. loads preprocessed tabular and sequence data
2. trains the baseline IDS
3. trains the attacker generator
4. runs adversarial fine-tuning rounds
5. augments the clean training set with generated attacks
6. trains the robust IDS
7. exports metrics, confusion matrices, histories, and manifests

### 3. Custom Input Scoring

`CustomInputRunner`:

- aligns any CSV input to the trained feature schema
- fills missing features with `0.0`
- scores row-level tabular predictions
- scores sequence windows when enough rows are present
- writes scored CSVs into `artifacts/custom_input/`

## FastAPI Backend

Run the API with:

```bash
python app.py
```

Available endpoints:

- `GET /` health check
- `POST /predict` score one row or a list of rows
- `POST /simulate_evasion` generate a single evasive mutation summary
- `GET /artifacts/manifest`
- `GET /artifacts/metrics`
- `GET /artifacts/training-history`
- `GET /artifacts/generator`
- `GET /artifacts/eda`
- `GET /artifacts/confusion-matrices`
- `GET /artifacts/eda/plots/{filename}`

The backend requires trained artifacts. If `robust_ids.pt` or `feature_names.txt` are missing, prediction endpoints return `503`.

## Frontend Dashboard

The frontend lives in `frontend/` and is already wired to the current API service layer in `frontend/src/app/lib/api.ts`.

Current shipped routes:

- `/` Dashboard
- `/architecture`
- `/training-lab`
- `/analytics`
- `/live-demo`

The dashboard reads real backend artifacts when the API is online and falls back gracefully when endpoints are unavailable.

## Running The Project

### Python backend

Install dependencies:
>>>>>>> dfc8aa63c8970aa6b161418c11243fba5f167077

```bash
pip install -r requirements.txt
```

<<<<<<< HEAD
Required packages: `numpy`, `pandas`, `scikit-learn`, `torch`, `fastapi`, `uvicorn`

Optional: `matplotlib` (EDA plots), `imbalanced-learn` (SMOTE balancing)

### 2. Place the dataset

Put CICIDS2017 CSVs into `datasets/MachineLearningCVE/`, or place `datasets/MachineLearningCSV.zip` and the preprocessor will extract it automatically.

### 3. Install frontend dependencies

```bash
cd frontend
npm install
```

---

## Running the Pipeline

### Full pipeline (single command)
=======
Run the full local pipeline:
>>>>>>> dfc8aa63c8970aa6b161418c11243fba5f167077

```bash
python main.py
```

<<<<<<< HEAD
Runs: Preprocessing → Baseline IDS training → cGAN training → Adversarial fine-tuning → Robust IDS retraining → Evaluation → Custom input scoring.

### Step-by-step (with skip-if-exists)

```bash
python run_preprocessing.py     # Only runs if preprocessed/ doesn't exist
python run_training.py          # Only runs if trained models don't exist
python eda.py                   # Generate EDA plots and metric summaries
```

### Pretty run (all three with section headers)

```bash
python pretty_run.py
```

### Regenerate EDA plots only

```bash
python regenerate_eda_plots.py
```

### Run validation test

```bash
python test_pipeline.py
```

---

## FastAPI Backend

### Start the server

```bash
# Local development (port 8000)
python app.py

# Or with uvicorn directly
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

The `PORT` environment variable is respected (defaults to `8000` locally, `7860` in Docker/HF Spaces).

### Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `POST` | `/predict` | Score traffic rows (tabular + sequence) |
| `POST` | `/simulate_evasion` | GAN-based evasion simulation with countermeasures |
| `GET` | `/artifacts/manifest` | Training manifest (feature count, row counts, hyperparams) |
| `GET` | `/artifacts/metrics` | All evaluation metrics (individual JSON + summary CSV) |
| `GET` | `/artifacts/training-history` | Baseline & robust training loss curves |
| `GET` | `/artifacts/generator` | Generator state and feedback history |
| `GET` | `/artifacts/eda` | EDA report, label distribution, plot filenames |
| `GET` | `/artifacts/confusion-matrices` | All confusion matrices |
| `GET` | `/artifacts/eda/plots/{filename}` | Serve EDA plot images |

---

## React Frontend Dashboard

The frontend is a cybersecurity SOC-style dashboard built with React 18, TypeScript, Tailwind CSS v4, and Recharts.

### Start development server

```bash
cd frontend
npm run dev
```

The Vite dev server proxies `/api` requests to `http://localhost:8000` automatically.

### Pages

| Route | Page | Description |
|---|---|---|
| `/` | Dashboard | System overview, key metrics, performance comparison |
| `/architecture` | Architecture | GAN-vs-GAN framework explanation and component breakdown |
| `/training-lab` | Training Lab | Training loss curves, generator feedback, model checkpoints |
| `/analytics` | Analytics | Metrics comparison, confusion matrices, detailed reports |
| `/live-demo` | Live Demo | CSV upload scoring + real-time evasion simulation |

### Key Components

- **AnimatedMetricCard**: Animated metric displays with sparklines
- **GlassCard**: Glassmorphism card containers
- **ThreatIntelPanel**: Real-time threat intelligence visualization
- **ParticleBackground**: Animated particle canvas background
- **LiveFeed**: Simulated live traffic feed

### Design System

Dark cybersecurity theme with curated colors:
- **Dark Graphite** (`#1a1d24`, `#0f1117`) — backgrounds
- **Signal Blue** (`#3b82f6`, `#0ea5e9`) — primary actions
- **Cyber Green** (`#10b981`) — success/detection
- **Ember** (`#ef4444`, `#f97316`) — attack indicators

### Build for production

```bash
cd frontend
npm run build    # Output in frontend/dist/
```

---

## Deployment

### Backend → Hugging Face Spaces (Docker)

The `Dockerfile` builds a Python 3.11-slim container exposing port 7860:

```dockerfile
FROM python:3.11-slim
ENV PORT=7860
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY . /app
EXPOSE 7860
CMD ["python", "app.py"]
```

Upload to Hugging Face:

```bash
python upload_to_hf.py
```

This script uses `huggingface_hub` to upload the backend (excluding `frontend/`, `node_modules/`, `.git/`, `venv/`).

### Frontend → Vercel

1. Set the **Root Directory** to `frontend/` in Vercel project settings
2. Set environment variable: `VITE_API_URL=https://your-space.hf.space`
3. The `vercel.json` handles SPA routing:

```json
{
  "version": 2,
  "rewrites": [{ "source": "/(.*)", "destination": "/index.html" }]
}
```

---

## Artifacts & Outputs

### Models (`artifacts/models/`)

| File | Description |
|---|---|
| `baseline_ids.pt` | PyTorch checkpoint: baseline Transformer-LSTM IDS |
| `robust_ids.pt` | PyTorch checkpoint: adversarially retrained IDS |
| `attacker_cgan.pt` | PyTorch checkpoint: conditional GAN (generator + discriminator) |

### Metrics (`artifacts/training/`)

| File | Description |
|---|---|
| `metrics_summary.csv` | All metrics across splits and stages |
| `final_report.csv` | Complete evaluation report |
| `training_manifest.json` | Run metadata (feature count, row counts, hyperparams) |
| `generator_state.json` | Generator hyperparameters and feedback history |
| `generator_feedback.csv` | Per-round detection rate, stealth weight, loss |
| `baseline_training_history.csv` | Baseline IDS loss curves per epoch |
| `robust_training_history.csv` | Robust IDS loss curves per epoch |
| `*_metrics.json` | Per-split metric details |
| `*_confusion_matrix.csv` | Confusion matrices |
| `*_classification_report.json` | sklearn classification reports |

### Evaluation Metrics

Computed for each split (validation/test × tabular/sequence × baseline/robust × clean/adversarial):

- Accuracy, Precision, Recall, F1, ROC-AUC, False Positive Rate, False Negative Rate

---

## How the Models Work

### Defender: HybridIDSModel

Two parallel prediction paths trained with independent AdamW optimizers:

1. **TabularIDSHead** — per-flow detection:
   - `Linear(F → 2F)` → `LayerNorm` → `GELU` → `Dropout(0.2)` → `Linear(2F → F)` → `LayerNorm` → `GELU` → `Dropout(0.2)` → `Linear(F → 1)`

2. **TransformerLSTMSequenceHead** — session-window detection:
   - `Linear(F → model_dim=64)` → `TransformerEncoder(2 layers, 4 heads, GELU)` → `BiLSTM(hidden=64)` → `MeanPooling` → `LayerNorm` → `Linear(128 → 64)` → `GELU` → `Dropout` → `Linear(64 → 1)`

- **Loss**: `BCEWithLogitsLoss` with `pos_weight` derived from class imbalance
- **Optimizer**: AdamW (lr=1e-3, weight_decay=1e-4)
- **Default epochs**: 18

### Attacker: AdversarialTrafficGenerator

- **ConditionalGenerator**: `[z ∥ Embedding(label)]` → `Linear(48 → 256)` → `LayerNorm` → `GELU` → `Linear(256 → 256)` → `LayerNorm` → `GELU` → `Linear(256 → F)` → `Tanh`
- **ConditionalDiscriminator**: `[features ∥ Embedding(label)]` → `Linear(F+16 → 256)` → `LeakyReLU(0.2)` → `Dropout(0.2)` → `Linear(256 → 128)` → `LeakyReLU(0.2)` → `Dropout(0.2)` → `Linear(128 → 1)`
- **Optimizer**: Adam (lr=2e-4, betas=(0.5, 0.999))
- **Generator loss**: `L_adv + 0.15 × L_recon` where `L_recon = MSE(generated, class_centroid)`
- **Fine-tuning loss**: `L_adv + 0.15 × L_recon + 0.05 × L_benign_pull + stealth_weight × IDS_detection_prob`
- **Default epochs**: 40, **Latent dim**: 32

---

## Custom Input Scoring

Place any CSV in `custom_input/` and run:

```bash
python main.py
# or, if models already exist:
python -m custom_input.custom_test
```

The runner:
1. Aligns columns to the training feature schema
2. Fills missing features with `0.0`
3. Coerces all values to numeric
4. Scores with the robust IDS (tabular + optional sequence predictions)
5. Saves results to `artifacts/custom_input/`

---

## Utility Scripts

| Script | Purpose |
|---|---|
| `main.py` | Full pipeline: preprocess → train → evaluate → score |
| `pretty_run.py` | Preprocessing → training → EDA with formatted section headers |
| `run_preprocessing.py` | Standalone preprocessing (skips if outputs exist) |
| `run_training.py` | Standalone training (skips if outputs exist) |
| `eda.py` | Load metrics and generate EDA plots from training artifacts |
| `regenerate_eda_plots.py` | Re-render high-quality EDA plots from saved CSV summaries |
| `test_pipeline.py` | Validation test using synthetic data through CustomInputRunner |
| `upload_to_hf.py` | Upload backend to Hugging Face Spaces |

---

## Limitations

- Class balancing via undersampling can reduce training set size significantly when rare classes exist, making metrics less stable
- Sequence generation reshapes independent tabular samples into windows (no true temporal coherence in generated sequences)
- Custom input assumes feature-space compatibility with CICIDS2017 preprocessing output
- The frontend uses mock/fallback data when the backend is offline

---

## Future Improvements

- Replace cGAN with Wasserstein GAN-GP for more stable training
- Add TimeGAN for temporally coherent sequence generation
- Implement SMOTE or hybrid oversampling by default for better class balance
- Add experiment tracking (MLflow, W&B)
- Add WebSocket support for real-time training progress in the dashboard
- Add multiclass attack-type prediction
- Add CLI arguments or YAML configuration

---

## Troubleshooting

| Issue | Solution |
|---|---|
| Dataset not found | Ensure `datasets/MachineLearningCVE/` contains CSVs or provide `datasets/MachineLearningCSV.zip` |
| Missing `Label` column | Preprocessing expects exactly `Label` (case-sensitive). Rename if your CSV uses `label` or similar |
| Custom input not scoring | Verify `artifacts/models/robust_ids.pt` exists and CSV is in `custom_input/` |
| Frontend can't reach API | Check `VITE_API_URL` env var or ensure Vite proxy is configured (dev: `/api` → `localhost:8000`) |
| Metrics look too perfect | Expected with small balanced datasets; the pipeline is demonstration-oriented |
| GPU/CUDA issues | Set device in `training/config.py` or ensure `torch.cuda.is_available()` returns `True` |
| matplotlib cache warning | Non-fatal in restricted environments; doesn't affect the ML pipeline |

---

## License

This project is designed for research and educational purposes.
=======
Useful single-purpose entrypoints:

```bash
python run_preprocessing.py
python run_training.py
python app.py
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

By default the frontend uses `VITE_API_URL` if provided, otherwise it targets `/api`.

## Dataset Expectations

Expected raw dataset locations:

- `datasets/MachineLearningCVE/`
- or `datasets/MachineLearningCSV.zip`

The code expects CICIDS2017 CSV exports such as:

- `Monday-WorkingHours.pcap_ISCX.csv`
- `Tuesday-WorkingHours.pcap_ISCX.csv`
- `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`

## Notes And Limitations

- The checked-in artifacts reflect a compact processed dataset, not the full raw CICIDS2017 corpus.
- The docs and frontend now match the current route and endpoint structure, not the older multi-page mock-only layout.
- The project is research-oriented and optimized for demonstration, experimentation, and extension rather than production deployment.
>>>>>>> dfc8aa63c8970aa6b161418c11243fba5f167077
