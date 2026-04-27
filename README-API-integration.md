# GvG Backend and API Integration

This document describes the current backend contract exposed by `app.py` and how the frontend consumes it today.

## Current Backend Shape

The active backend is a FastAPI app with artifact-backed endpoints. It is not using the older `/api/status` or `/api/metrics/quick` contract from the initial frontend planning notes.

Run it with:

```bash
python app.py
```

Default local base URL:

```text
http://localhost:8000
```

## Live Endpoints

### Health

`GET /`

Response:

```json
{
  "status": "ok",
  "service": "gvg-ids-api"
}
```

### Predict

`POST /predict`

Accepts either:

- one JSON object representing a single row
- a non-empty JSON array of row objects

Example request:

```json
[
  {
    "Flow Duration": 1123.0,
    "Total Fwd Packets": 5,
    "Total Backward Packets": 2
  }
]
```

Response shape:

```json
{
  "input_rows": 1,
  "tabular_predictions": [
    {
      "row": 0,
      "predicted_binary_label": 1,
      "attack_probability": 0.9821
    }
  ],
  "sequence_predictions": null
}
```

If enough rows are supplied to form windows, `sequence_predictions` becomes a list with `window_end_row`, `predicted_binary_label`, and `attack_probability`.

### Evasion Simulation

`POST /simulate_evasion`

Accepts the same payload format as `/predict`, but currently analyzes the first aligned row and returns a generator-produced mutation summary.

Response shape:

```json
{
  "original_features": [0.1, 0.2],
  "morphed_features": [0.12, 0.18],
  "top_shifts": [
    {
      "feature": "Destination Port",
      "original": 80.0,
      "morphed": 443.0,
      "shift": "increased"
    }
  ],
  "counter_measures": [
    "Block or rate-limit specific destination port anomalies relating to Destination Port."
  ]
}
```

## Artifact Endpoints

These are the endpoints the current dashboard uses for analytics and visualization.

### `GET /artifacts/manifest`

Training metadata:

- feature count
- train, validation, test row counts
- sequence row counts
- label names
- device
- epoch counts
- latent dimension

### `GET /artifacts/metrics`

Returns:

- `individual`: raw metric JSONs discovered in `artifacts/training/*_metrics.json`
- `summary`: flat rows from `artifacts/training/metrics_summary.csv`

### `GET /artifacts/training-history`

Returns baseline and robust loss histories from:

- `baseline_training_history.csv`
- `robust_training_history.csv`

### `GET /artifacts/generator`

Returns generator state and feedback from:

- `generator_state.json`
- `generator_feedback.csv`

### `GET /artifacts/eda`

Returns:

- EDA report text
- label distribution rows
- available plot filenames

### `GET /artifacts/confusion-matrices`

Returns every `*_confusion_matrix.csv` file under `artifacts/training/` in a structured JSON form.

### `GET /artifacts/eda/plots/{filename}`

Serves stored EDA plot images.

## Frontend Integration Notes

The frontend API service is in `frontend/src/app/lib/api.ts`.

Current default base resolution:

```ts
const API_BASE = import.meta.env.VITE_API_URL || "/api";
```

This means:

- for local direct backend usage, set `VITE_API_URL=http://localhost:8000`
- for reverse-proxy deployments, keep the default `/api` and map requests to the FastAPI app

## Required Trained Artifacts

Prediction endpoints require:

- `artifacts/feature_names.txt`
- `artifacts/models/robust_ids.pt`

Evasion simulation also requires:

- `artifacts/models/attacker_cgan.pt`

If these are missing, the backend returns `503`.

## Current Frontend Consumers

- `Dashboard` reads health, manifest, metrics, EDA, and training history
- `TrainingLab` reads training history, generator state, and manifest
- `Analytics` reads summary metrics and confusion matrices
- `LiveDemo` uses `/predict` and `/simulate_evasion`

## Suggested Local Setup

Terminal 1:

```bash
pip install -r requirements.txt
python app.py
```

Terminal 2:

```bash
cd frontend
npm install
VITE_API_URL=http://localhost:8000 npm run dev
```
