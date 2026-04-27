---
title: GvG Defense API
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: docker
app_file: app.py
pinned: false
---

# GvG: GAN-vs-GAN Defense for Intrusion Detection Systems

GvG is an end-to-end intrusion-detection project built around adversarial training. The repository now contains three working layers:

- a CICIDS2017 preprocessing pipeline
- a completed training pipeline that produces baseline and robust IDS artifacts
- a FastAPI backend plus React dashboard for metrics, artifact exploration, and live scoring

## Current Project Status

The repository is beyond the planning stage and already includes generated artifacts from a completed run.

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

```bash
pip install -r requirements.txt
```

Run the full local pipeline:

```bash
python main.py
```

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
