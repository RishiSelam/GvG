# GvG Knowledgebase

This knowledgebase is a comprehensive technical reference for the GvG project — covering the mathematics, model internals, preprocessing details, API specifications, frontend architecture, deployment topology, hyperparameter guide, and presentation-ready talking points.

---

## 1) Project Summary

**Goal**: Measure and improve IDS robustness against evasive, GAN-generated attack traffic through an adversarial training loop between a deep-learning defender and a conditional GAN attacker.

**Stack**:
- **ML / Training**: Python 3.11, PyTorch (Transformer, LSTM, cGAN), scikit-learn (metrics, preprocessing), pandas, NumPy
- **API**: FastAPI + Uvicorn
- **Frontend**: React 18, TypeScript, Vite 6, Tailwind CSS v4, Recharts, Motion (Framer Motion), React Router 7
- **Deployment**: Docker → Hugging Face Spaces (backend), Vercel (frontend)

**Entrypoints**:
| Script | Purpose |
|---|---|
| `main.py` | Full end-to-end: preprocess → train → generate → evaluate → score custom input |
| `app.py` | FastAPI server for prediction, evasion simulation, and artifact endpoints |
| `pretty_run.py` | Preprocessing → training → EDA with formatted section headers |
| `run_preprocessing.py` | Standalone preprocessing (skip-if-exists) |
| `run_training.py` | Standalone training (skip-if-exists) |
| `eda.py` | Post-training EDA: load metrics JSON/CSV, plot loss curves and summaries |
| `regenerate_eda_plots.py` | Re-render high-quality EDA plots from saved CSV artifacts |
| `test_pipeline.py` | Validation: build synthetic input → score → verify output |
| `upload_to_hf.py` | Push backend to Hugging Face Spaces via `huggingface_hub` |

---

## 2) Key Concepts and Mathematics

### 2.1 Conditional GAN Objectives

The generator G and discriminator D play a minimax game conditioned on class label y:

$$
\mathcal{L}_D = -\mathbb{E}_{x\sim p_{data}(x|y)}[\log D(x|y)] - \mathbb{E}_{z\sim p_z(z)}[\log(1 - D(G(z|y)|y))]
$$

$$
\mathcal{L}_G = -\mathbb{E}_{z\sim p_z(z)}[\log D(G(z|y)|y)]
$$

A reconstruction regularizer keeps generated features near the attack class centroid c_y:

$$
\mathcal{L}_{recon} = \|G(z|y) - c_y\|_2^2
$$

**Generator total loss (initial training)**:

$$
\mathcal{L}_G^{total} = \mathcal{L}_G + 0.15 \cdot \mathcal{L}_{recon}
$$

**Generator total loss (adversarial fine-tuning against IDS)**:

$$
\mathcal{L}_G^{finetune} = \mathcal{L}_G + 0.15 \cdot \mathcal{L}_{recon} + 0.05 \cdot \mathcal{L}_{benign\_pull} + w_{stealth} \cdot P_{IDS}(attack|G(z|y))
$$

Where `L_benign_pull = MSE(mean(generated), benign_centroid)` and `w_stealth` adapts based on detection rate (increases if IDS detects >80%, decreases otherwise).

### 2.2 IDS Training Objective

Binary cross-entropy with logits, weighted by class imbalance:

$$
\mathrm{BCE}(s, t) = -t \cdot \log\sigma(s) \cdot w_{pos} - (1-t) \cdot \log(1-\sigma(s))
$$

Where `w_pos = count_negatives / count_positives` to avoid majority-class bias.

### 2.3 Why Adversarial Retraining Helps

- The classifier learns a decision boundary separating benign from attack samples
- The GAN generator pushes attack samples toward the benign centroid, crossing the boundary
- By augmenting training data with these near-boundary samples, the decision boundary expands to cover more attack modes
- Result: lower false negatives on evasive traffic after retraining

### 2.4 Evaluation Metrics

| Metric | Formula |
|---|---|
| Accuracy | (TP+TN) / (TP+TN+FP+FN) |
| Precision | TP / (TP+FP) |
| Recall | TP / (TP+FN) |
| F1 | 2 × (Precision × Recall) / (Precision + Recall) |
| FPR | FP / (FP+TN) |
| FNR | FN / (FN+TP) |
| ROC-AUC | Area under True Positive Rate vs False Positive Rate curve |

---

## 3) Preprocessing Pipeline (`preprocessing_CICIDS2017.py`)

The `Preprocess` class runs 11 sequential stages:

| Stage | Method | Description |
|---|---|---|
| 1. Load & merge | `stage_load_and_merge_data()` | Find CSVs (or extract from zip), read with pandas, concatenate |
| 2. Fix columns | `stage_fix_columns()` | Strip whitespace from column names, validate `Label` column exists |
| 3. Clean data | `stage_clean_data()` | Coerce to numeric, replace ±inf with NaN, drop NaN/duplicates, cast to float32 |
| 4. Labels | `stage_encode_labels()` | `LabelEncoder` → multiclass ints + binary (0=BENIGN, 1=attack), save `label_encoder.pkl` |
| 5. Balance | `stage_balance_dataset()` | Undersample (default) or SMOTE; reduces each class to min-class count |
| 6. Features | `stage_feature_engineering()` | `VarianceThreshold` filter, save `feature_names.txt` |
| 7. Scale | `stage_scale_features()` | `StandardScaler` (default) or `MinMaxScaler`, save scaler pickle |
| 8. Split | `stage_split_dataset()` | `train_test_split` with stratification (fallback: non-stratified), 60% train / 20% val / 20% test |
| 9. Sequences | `stage_create_sequences()` | Sliding windows of shape `(N, seq_len, F)` with configurable stride |
| 10. Save | `stage_save_outputs()` | Write CSVs, NPY files, and combined `training_dataset.csv` |
| 11. EDA | `stage_eda_report()` | Generate `eda_report.txt`, `label_distribution.csv`, `null_counts.csv`, `numeric_summary.csv`, PNG plots |

**Constructor parameters**:

| Parameter | Default | Description |
|---|---|---|
| `dataset_dir` | `datasets/MachineLearningCVE` | Path to raw CSV folder |
| `zip_path` | `datasets/MachineLearningCSV.zip` | Zip fallback |
| `balance_strategy` | `undersample` | `undersample`, `smote`, or `none` |
| `variance_threshold` | `0.0` | Drop features with variance below this |
| `scaler_name` | `standard` | `standard` or `minmax` |
| `sequence_length` | `10` | Sliding window length |
| `sequence_stride` | `1` | Sliding window step |
| `test_size` | `0.20` | Fraction for test split |
| `validation_size` | `0.25` | Fraction of remaining for validation |

**Edge cases**:
- If no CSV files found: raises `FileNotFoundError`
- If `Label` column missing: raises `ValueError`
- If sequences can't be created (too few rows): returns empty arrays; downstream handles gracefully
- If SMOTE requested but `imblearn` not installed: falls back to undersampling with a warning
- Stratified splitting falls back to non-stratified if any class has too few samples

---

## 4) Model & Generator Internals

### 4.1 Defender: `training/ids_model.py`

**TabularIDSHead** (per-flow MLP):
```
Input(F) → Linear(F, max(64,2F)) → LayerNorm → GELU → Dropout(0.2)
         → Linear(max(64,2F), max(32,F)) → LayerNorm → GELU → Dropout(0.2)
         → Linear(max(32,F), 1)
Output: scalar logit
```

**TransformerLSTMSequenceHead** (session-window model):
```
Input(batch, seq_len, F)
→ Linear(F, 64)                          # input projection
→ TransformerEncoder(2 layers, 4 heads, dim_ff=256, GELU, dropout=0.2, batch_first=True)
→ BiLSTM(input=64, hidden=64, 1 layer)   # temporal transitions
→ MeanPooling(dim=1)                      # (batch, 128)
→ LayerNorm(128) → Linear(128, 64) → GELU → Dropout(0.2) → Linear(64, 1)
Output: scalar logit per window
```

**HybridIDSModel** (wrapper):
- Trains both heads in a single `fit()` loop with independent AdamW optimizers
- `positive_class_weight` computed as `negatives / positives` from training labels
- Loss: `BCEWithLogitsLoss(pos_weight=...)`
- Saves/loads as PyTorch checkpoint (`.pt`) containing both state dicts + hyperparameters + training history
- `predict_tabular()` / `predict_sequences()` return `PredictionBundle(labels, probabilities)`
- `score_custom_rows()` runs both tabular and (if enough rows) sequence predictions

### 4.2 Attacker: `training/attacker_generator.py`

**ConditionalGenerator**:
```
[z(32) ∥ Embedding(label, 16)] → Linear(48, 256) → LayerNorm → GELU
                                → Linear(256, 256) → LayerNorm → GELU
                                → Linear(256, F) → Tanh
```

**ConditionalDiscriminator**:
```
[features(F) ∥ Embedding(label, 16)] → Linear(F+16, 256) → LeakyReLU(0.2) → Dropout(0.2)
                                      → Linear(256, 128) → LeakyReLU(0.2) → Dropout(0.2)
                                      → Linear(128, 1)
Output: scalar logit (higher = more real)
```

**Training loop** (`fit()`):
1. Filter to attack-only rows (`y_binary == 1`)
2. Compute per-class centroids and benign centroid
3. For each batch:
   - Discriminator: `D_loss = 0.5 * (BCE(D(real), 1) + BCE(D(G(z), 0)))`
   - Generator: `G_loss = BCE(D(G(z)), 1) + 0.15 * MSE(G(z), centroid)`

**Adversarial fine-tuning** (`adversarial_fine_tune()`):
1. Freeze IDS tabular model parameters
2. For each round:
   - Sample noise + attack labels → generate features
   - Compute: `total_loss = L_adv + 0.15*L_recon + 0.05*L_benign_pull + stealth_weight * sigmoid(IDS_logits).mean()`
   - Update generator only
   - Measure detection rate of generated samples
   - Adapt `stealth_weight`: increase (+0.08) if detection >80%, decrease (-0.03) otherwise
   - Record `GeneratorFeedback(round_id, detection_rate, stealth_weight, loss)`

**Generation** (`generate_tabular()`, `generate_sequences()`):
- Sample labels from training attack distribution
- Generate features with frozen generator (no grad)
- Return `(features, binary_labels=1, multiclass_labels)`
- Sequences: generate `N × seq_len` rows, reshape to `(N, seq_len, F)`

**Persistence**:
- `save_checkpoint(path)`: saves generator + discriminator state dicts, latent_dim, num_classes, feature_dim, stealth_weight
- `load(path, feature_names, device)`: reconstructs from checkpoint

### 4.3 Pipeline: `training/pipeline.py`

**`AdversarialTrainingPipeline.run()` sequence**:

| Step | Action | Output |
|---|---|---|
| 1 | Load preprocessed data bundle | `PreprocessedDataBundle` with all splits and sequences |
| 2 | Build & train baseline IDS | `baseline_ids.pt`, `baseline_training_history.csv` |
| 3 | Build & train cGAN on attack rows | `attacker_cgan.pt` |
| 4 | Adversarial fine-tune generator | `generator_state.json`, `validation_adversarial_round_*.csv` |
| 5 | Augment training data + retrain IDS | `robust_ids.pt`, `robust_training_history.csv` |
| 6 | Evaluate (baseline + robust × clean + adversarial) | `metrics_summary.csv`, `final_report.csv`, per-split JSONs/CSVs |

---

## 5) Data Shapes and API Usage

### Typical data shapes

| Array | Shape | Notes |
|---|---|---|
| `X_train` | `(N_train, F)` | F = number of features (e.g., 66) |
| `y_train.csv` | `(N_train, 2)` | Columns: `label_multiclass`, `label_binary` |
| `sequences_train.npy` | `(M_windows, seq_len, F)` | `seq_len` default = 10 |

### POST /predict — Example

Request:
```json
[{"Duration": 0.023, "SrcPort": 51718, "DstPort": 22, "Protocol": 6, "FlowBytes": 512}]
```

Response:
```json
{
  "input_rows": 1,
  "tabular_predictions": [
    {"row": 0, "predicted_binary_label": 1, "attack_probability": 0.943}
  ],
  "sequence_predictions": null
}
```

### POST /simulate_evasion — Example

Request: same as `/predict` (first row is used as the source).

Response:
```json
{
  "original_features": [0.023, 51718, ...],
  "morphed_features": [0.015, 51718, ...],
  "top_shifts": [
    {"feature": "FlowBytes", "original": 512.0, "morphed": 1024.0, "shift": "increased"}
  ],
  "counter_measures": [
    "Deploy Deep Packet Inspection (DPI) to block payloads with unexpected FlowBytes."
  ]
}
```

**Countermeasure logic**: auto-generated based on feature name keywords (`port` → rate-limit, `length/size` → DPI, `flag` → firewall rules, `duration/time` → connection timeouts, default → IDS signature monitoring).

---

## 6) Frontend Architecture

### Technology stack

React 18 + TypeScript, Vite 6 (build tool), Tailwind CSS v4 (styling), Recharts (charts), Motion/Framer Motion (animations), React Router 7 (routing), Lucide React (icons), Radix UI (primitives), MUI (supplementary components).

### Route map

| Path | Component | Data Source |
|---|---|---|
| `/` | `Dashboard` | `/artifacts/manifest`, `/artifacts/metrics` |
| `/architecture` | `Architecture` | Static content |
| `/training-lab` | `TrainingLab` | `/artifacts/training-history`, `/artifacts/generator` |
| `/analytics` | `Analytics` | `/artifacts/metrics`, `/artifacts/confusion-matrices` |
| `/live-demo` | `LiveDemo` | `/predict`, `/simulate_evasion` |

### API integration layer (`frontend/src/app/lib/api.ts`)

- `API_BASE` = `import.meta.env.VITE_API_URL || "/api"`
- Typed fetch wrappers for every backend endpoint
- Client-side CSV parser (`parseCSV()`) with quoted-field support
- All response types are fully typed (TypeScript interfaces)
- 10-second timeout on artifact fetches, 5-second on health check

### Fallback data (`frontend/src/app/lib/mockData.ts`)

When the backend is offline, the frontend initializes with zero-value defaults so the UI renders correctly (empty charts, zero metrics). This allows:
- Demo presentations without a live backend
- Frontend development in isolation
- UI/UX testing

### Vite dev proxy (`frontend/vite.config.ts`)

```typescript
server: {
  proxy: {
    '/api': {
      target: 'http://localhost:8000',
      changeOrigin: true,
      rewrite: (path) => path.replace(/^\/api/, ''),
    },
  },
}
```

In development, frontend calls to `/api/predict` are proxied to `http://localhost:8000/predict`.

In production, `VITE_API_URL` is set to the full HF Spaces URL (e.g., `https://username-gvg-backend.hf.space`).

### Key custom components

| Component | File | Purpose |
|---|---|---|
| `AnimatedMetricCard` | `components/AnimatedMetricCard.tsx` | Animated metric display with trend sparklines |
| `GlassCard` | `components/GlassCard.tsx` | Glassmorphism container card |
| `ThreatIntelPanel` | `components/ThreatIntelPanel.tsx` | Real-time GAN threat intelligence visualization |
| `ParticleBackground` | `components/ParticleBackground.tsx` | Animated canvas particle background |
| `LiveFeed` | `components/LiveFeed.tsx` | Simulated live traffic event feed |
| `ThreatLevelIndicator` | `components/ThreatLevelIndicator.tsx` | Animated threat level gauge |
| `GlowingBadge` | `components/GlowingBadge.tsx` | Pulsing status badges |

### Additional legacy pages (not in active routes)

These page components exist in the `pages/` directory but are not connected to active routes. They were part of earlier frontend iterations:

`Pipeline.tsx`, `Dataset.tsx`, `Training.tsx`, `Adversarial.tsx`, `Metrics.tsx`, `CustomInput.tsx`, `Artifacts.tsx`

---

## 7) Deployment Architecture

```
┌────────────────────────┐         ┌─────────────────────────┐
│  Frontend (Vercel)     │  HTTPS  │  Backend (HF Spaces)    │
│  React + Vite          │ ──────→ │  FastAPI + PyTorch       │
│  Tailwind CSS v4       │         │  Docker (port 7860)      │
│  VITE_API_URL env var  │         │  Models loaded from      │
│                        │         │  artifacts/models/*.pt   │
└────────────────────────┘         └─────────────────────────┘
```

### Backend Docker (`Dockerfile`)

- Base: `python:3.11-slim`
- Env: `PYTHONDONTWRITEBYTECODE=1`, `PYTHONUNBUFFERED=1`, `PORT=7860`
- `app.py` reads `PORT` env var: `int(os.environ.get("PORT", 8000))`
- CORS enabled for all origins (`allow_origins=["*"]`)

### Frontend Vercel (`vercel.json`)

- SPA catch-all rewrite: `{ "source": "/(.*)", "destination": "/index.html" }`
- Environment variable: `VITE_API_URL` must point to the deployed backend

### Upload script (`upload_to_hf.py`)

Uses `HfApi.upload_folder()` with ignore patterns:
- `frontend/**`, `node_modules/**`, `.git/**`, `venv/**`, `.env`, `__pycache__/**`, `*.pyc`, `.ipynb_checkpoints/**`

---

## 8) Hyperparameter Reference

### Pipeline config (`training/config.py` — `PipelineConfig`)

| Parameter | Default | Description |
|---|---|---|
| `sequence_length` | 10 | Sliding window size for sequence models |
| `random_state` | 42 | Global random seed |
| `adversarial_rounds` | 2 | Number of adversarial fine-tuning rounds |
| `synthetic_multiplier` | 0.75 | Fraction of attack rows to generate as synthetic |
| `ids_epochs` | 18 | Training epochs for the IDS |
| `gan_epochs` | 40 | Training epochs for the cGAN |
| `ids_learning_rate` | 1e-3 | AdamW learning rate for IDS |
| `gan_learning_rate` | 2e-4 | Adam learning rate for generator & discriminator |
| `ids_batch_size` | 64 | IDS training batch size |
| `gan_batch_size` | 64 | GAN training batch size |
| `latent_dim` | 32 | Noise vector dimension for generator |
| `device` | auto (`cuda` if available) | PyTorch device |

### Preprocessing config (constructor args)

| Parameter | Default | Description |
|---|---|---|
| `variance_threshold` | 0.0 | Drop features with variance below this |
| `balance_strategy` | `undersample` | `undersample`, `smote`, or `none` |
| `scaler_name` | `standard` | `standard` or `minmax` |
| `sequence_stride` | 1 | Sliding window step size |
| `test_size` | 0.20 | Fraction for test split |
| `validation_size` | 0.25 | Fraction of remaining for validation |

### Generator internals

| Parameter | Value | Location |
|---|---|---|
| `class_embedding_dim` | 16 | `ConditionalGenerator.__init__` |
| `reconstruction_weight` | 0.15 | `fit()` and `adversarial_fine_tune()` |
| `benign_pull_weight` | 0.05 | `adversarial_fine_tune()` |
| `initial_stealth_weight` | 0.35 | `AdversarialTrafficGenerator.__init__` |
| `stealth_weight_increase` | +0.08 | if detection_rate > 0.80 |
| `stealth_weight_decrease` | -0.03 | if detection_rate ≤ 0.80 |
| `stealth_weight_bounds` | [0.10, 1.20] | clamped per round |

---

## 9) Complete Artifact Reference

### Root-level artifacts (`artifacts/`)

| File | Format | Created By |
|---|---|---|
| `feature_names.txt` | Text (one feature per line) | Preprocessing stage 6 |
| `label_encoder.pkl` | Pickle (sklearn LabelEncoder) | Preprocessing stage 4 |
| `scaler_standard.pkl` | Pickle (sklearn StandardScaler) | Preprocessing stage 7 |

### EDA artifacts (`artifacts/eda/`)

| File | Description |
|---|---|
| `eda_report.txt` | Summary: shape, duplicates, nulls, label distribution, top variance features |
| `label_distribution.csv` | Columns: `label`, `count` |
| `null_counts.csv` | Columns: `column`, `null_count` |
| `numeric_summary.csv` | `DataFrame.describe().T` (mean, std, min, 25%, 50%, 75%, max) |
| `plots/label_distribution.png` | Bar chart of class counts |
| `plots/missing_values.png` | Horizontal bar chart of top null columns |
| `plots/top_feature_variance.png` | Bar chart of top-15 features by std (log scale) |
| `plots/correlation_heatmap.png` | Annotated heatmap of top-15 features (from training_dataset.csv) |

### Model checkpoints (`artifacts/models/`)

| File | Contents |
|---|---|
| `baseline_ids.pt` | TabularIDSHead + TransformerLSTMSequenceHead state dicts, hyperparams, history |
| `robust_ids.pt` | Same structure, adversarially retrained |
| `attacker_cgan.pt` | ConditionalGenerator + ConditionalDiscriminator state dicts, latent_dim, num_classes, feature_dim |

### Training artifacts (`artifacts/training/`)

| File | Description |
|---|---|
| `training_manifest.json` | Feature count, row counts per split, label names, device, epochs, latent_dim |
| `metrics_summary.csv` | All metrics: split × stage × metric values |
| `final_report.csv` | Same data as summary, saved separately |
| `baseline_training_history.csv` | Per-epoch: tabular/sequence train/validation loss |
| `robust_training_history.csv` | Same for robust model |
| `generator_state.json` | Latent dim, epochs, LR, stealth_weight, feedback_history array |
| `generator_feedback.csv` | Per-round: round_id, detection_rate, stealth_weight, generator_loss |
| `{stage}_{split}_metrics.json` | Per-split metric dict |
| `{stage}_{split}_classification_report.json` | sklearn classification_report as dict |
| `{stage}_{split}_confusion_matrix.csv` | 2×2 matrix (actual_0/1 × pred_0/1) |

### Generated samples (`artifacts/generated/`)

| File | Description |
|---|---|
| `synthetic_training_attacks.csv` | GAN-generated attacks used to augment training data |
| `test_adversarial_samples.csv` | GAN-generated attacks used for adversarial evaluation |
| `validation_adversarial_round_{N}.csv` | Per-round generated samples during fine-tuning |

---

## 10) Slide-Ready Speaking Points

**Slide 1 — Title & Motivation**
- Title: "GvG — Adversarially Robust Intrusion Detection via Conditional GAN"
- Why IDS fail on distribution shifts; crafted payloads evade signature-based detection

**Slide 2 — Dataset & Preprocessing**
- CICIDS2017: 2.8M+ flows, 8 CSV files, 15 attack categories
- 11-stage pipeline: clean → encode → balance → feature select → scale → split → sequence
- Show `label_distribution.csv` and EDA plots

**Slide 3 — Defender Architecture**
- Two-path model: Tabular MLP + Transformer+BiLSTM
- Why combining per-row and window views captures both immediate anomalies and temporal context

**Slide 4 — Attacker (cGAN)**
- Show math: G+D losses, reconstruction term, benign pull, IDS stealth objective
- Explain `adversarial_fine_tune` loop with adaptive stealth weight

**Slide 5 — Adversarial Loop**
- Pipeline diagram from `main.py` steps
- Evaluation on clean vs adversarial splits

**Slide 6 — Results**
- Comparison table from `metrics_summary.csv` (Baseline vs Robust)
- Highlight recall improvement (lower FN on adversarial data)

**Slide 7 — Live Demo**
- Start API: `uvicorn app:app --reload`
- POST sample to `/predict`, then same sample to `/simulate_evasion`
- Show morphed features and auto-generated countermeasures
- Show React dashboard: training curves, confusion matrices, real-time scoring

**Slide 8 — Limitations & Future Work**
- Current generator uses feature-space MLP, not Wasserstein GAN or TimeGAN
- Undersampling can reduce training set significantly
- Future: WGAN-GP, TimeGAN for sequences, experiment tracking, multiclass prediction

---

## 11) Troubleshooting Reference

| Symptom | Root Cause | Fix |
|---|---|---|
| `FileNotFoundError: Dataset not found` | CSVs not in expected location | Place CSVs in `datasets/MachineLearningCVE/` or zip at `datasets/MachineLearningCSV.zip` |
| `ValueError: Label column not found` | Column name mismatch | Preprocessing expects `Label` (exact case); rename if CSV uses `label` |
| `RuntimeError: No attack rows available` | All rows labeled BENIGN after balancing | Check label encoding; ensure dataset has attack samples |
| Custom input returns all zeros | Model not trained or wrong model path | Verify `artifacts/models/robust_ids.pt` exists |
| Frontend shows empty charts | Backend offline or CORS issue | Check API server is running; verify `VITE_API_URL` env var |
| `torch.cuda.is_available()` returns False | CUDA not installed or wrong torch version | Install CUDA-enabled PyTorch: `pip install torch --index-url https://download.pytorch.org/whl/cu121` |
| matplotlib cache warning | Restricted environment (e.g., Docker) | Non-fatal; does not affect ML pipeline |
| Metrics look perfect (1.0 accuracy) | Small balanced dataset after undersampling | Expected in demo mode; use SMOTE or `balance_strategy="none"` for larger training sets |
| HF Spaces build fails | Missing dependency or wrong port | Check `requirements.txt` includes all deps; ensure `PORT=7860` in Dockerfile |
| Vercel 404 on page refresh | SPA routing not configured | Verify `vercel.json` has `{"source":"/(.*)", "destination":"/index.html"}` |
