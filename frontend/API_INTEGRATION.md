# Frontend API Integration Guide

This frontend is already integrated with the current FastAPI backend. This file documents the contract that exists in the code today.

## Base URL

`src/app/lib/api.ts` uses:

```ts
const API_BASE = import.meta.env.VITE_API_URL || "/api";
```

Recommended local setup:

```bash
VITE_API_URL=http://localhost:8000 npm run dev
```

## Current Requests Used By The UI

### Health

Used by `RootLayout` and `Dashboard`.

```http
GET /
```

Purpose:

- backend online/offline badge
- lightweight connectivity check

### Prediction

Used by `LiveDemo`.

```http
POST /predict
Content-Type: application/json
```

Accepted body:

- a single row object
- or an array of row objects

Frontend expectation:

- `tabular_predictions` is always present
- `sequence_predictions` may be `null`

### Evasion Simulation

Used by `LiveDemo` for the threat-intel style panel.

```http
POST /simulate_evasion
Content-Type: application/json
```

Frontend expectation:

- `original_features`
- `morphed_features`
- `top_shifts`
- `counter_measures`

### Training Manifest

Used by `Dashboard` and `TrainingLab`.

```http
GET /artifacts/manifest
```

Expected fields:

- `feature_count`
- `train_rows`
- `validation_rows`
- `test_rows`
- `sequence_train_rows`
- `sequence_validation_rows`
- `sequence_test_rows`
- `label_names`
- `device`
- `ids_epochs`
- `gan_epochs`
- `latent_dim`

### Metrics

Used by `Dashboard` and `Analytics`.

```http
GET /artifacts/metrics
```

Frontend uses the `summary` rows for:

- baseline vs robust comparisons
- clean test metrics
- adversarial evaluation metrics

### Training History

Used by `Dashboard` and `TrainingLab`.

```http
GET /artifacts/training-history
```

Expected response:

```ts
{
  baseline?: TrainingEpoch[];
  robust?: TrainingEpoch[];
}
```

Each epoch includes:

- `epoch`
- `tabular_train_loss`
- `sequence_train_loss`
- `tabular_validation_loss`
- `sequence_validation_loss`

### Generator Data

Used by `TrainingLab`.

```http
GET /artifacts/generator
```

Used fields:

- `state`
- `feedback`
- `state.feedback_history`

### EDA

Used by `Dashboard`.

```http
GET /artifacts/eda
```

Expected fields:

- `report`
- `label_distribution`
- `plots`

### Confusion Matrices

Used by `Analytics`.

```http
GET /artifacts/confusion-matrices
```

Expected shape:

```ts
type ConfusionMatricesResponse = Record<
  string,
  {
    rows: string[];
    cols: string[];
    values: number[][];
  }
>;
```

## Common Local Flow

1. Start the Python backend with `python app.py`
2. Start the frontend with `VITE_API_URL=http://localhost:8000 npm run dev`
3. Open the app and verify the header shows `Backend Online`

## Important Notes

- Older documentation that referenced `/api/status`, `/api/metrics/quick`, or many separate dataset pages is outdated.
- The current implementation centers on artifact-backed analytics plus live CSV scoring.
- If model artifacts are missing, `LiveDemo` requests will fail with backend `503` responses until training artifacts are available.
