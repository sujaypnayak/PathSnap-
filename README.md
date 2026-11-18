# PathSnap

GPS map matching and route reconstruction using adaptive road-segment candidates, a RandomForest classifier, and Hidden Markov Model (HMM)/Viterbi sequence decoding.

## Overview

PathSnap processes noisy GPS observations against a directed road network. It (1) generates candidate road arcs and geometric/temporal features, (2) scores candidates with a trained classifier, and (3) decodes a road-arc sequence using network-aware HMM/Viterbi transitions. The decoder applies a reversal penalty of 5.0 to discourage unrealistic back-and-forth transitions.

The application has a Python/FastAPI backend and React/Vite/Leaflet frontend. Both run on the same laptop by default. The official hackathon guide describes a two-laptop architecture; that deployment has not been verified.

## Repository layout

- `src/features/` — raw trip reading, candidate generation, feature engineering.
- `src/hmm/` — road graph and Viterbi decoder.
- `src/api/` — model loading, prediction pipeline, FastAPI endpoints.
- `src/models/` — deployed model (`best_model.joblib`, tracked with Git LFS) and feature metadata.
- `scripts/` — preprocessing, model training, Bayesian tuning, evaluation, and route visualization.
- `experiments/` — model experiments, out-of-fold predictions, comparisons, and result tables.
- `reports/` — exploratory analysis, preprocessing summaries, Folium maps, and plots.
- `tests/` — Python automated tests.
- `ui/` — React evaluator interface.
- `configs/` — configuration and split definitions.

## Requirements and installation

Install Python, Node.js/npm, and Git LFS. From the repository root in PowerShell:

```powershell
git lfs install
git lfs pull
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
cd ui
npm install
cd ..
```

The deployed model is stored via Git LFS. Verify that `src/models/best_model.joblib` contains the actual model, not just a text pointer.

## Dataset layout

The API reads data from the project-relative `data/raw/Student folder/` directory, configured in `src/api/config.py`:

```text
data/raw/Student folder/
  train/<trip_id>/<trip_id>.track, .nodes, .arcs, .route
  student_test/<trip_id>/<trip_id>.track, .nodes, .arcs, .route
  evaluator/<trip_id>/<trip_id>.track, .nodes, .arcs, .route
```

The `.track` file stores longitude, latitude and timestamp; `.nodes` stores road-node coordinates; `.arcs` stores directed node connections; `.route` contains ordered reference arc IDs. Reference routes are not verified per-GPS-point labels. Raw evaluator data and credentials must not be committed.

## Run locally

Open two terminals on the **same laptop**.

Backend, from the project root:

```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

Frontend, from the project root:

```powershell
cd ui
npm run dev
```

Open http://localhost:5173. The Vite proxy sends `/api` requests to `http://127.0.0.1:8000` by default. FastAPI documentation is at http://127.0.0.1:8000/docs.

## API

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API information |
| GET | `/health` | Health status |
| GET | `/dev/trips` | List student-test trips |
| GET | `/trips` | List evaluator trips (requires ten locally available evaluator trips) |
| POST | `/predict` | Predict route for a student-test trip |
| POST | `/evaluate` | Predict and evaluate an evaluator trip |

POST payload example:

```json
{"trip_id": "00000030"}
```

Responses include predicted arc IDs, GPS points, and route geometry. `/evaluate` additionally returns route precision/recall/F1, ordered route precision/recall/F1, and connectivity rate. The UI displays the GPS trace and predicted road route. Per-GPS-point accuracy is returned as `null` because verified per-observation reference labels are unavailable. Evaluation metrics are computed server-side; the reference arc sequence is not sent to the UI.

## Model experimentation and results

Five candidate-level classifiers were trained and compared using a trip-separated 64-trip training / 16-trip validation split:

| Model | Validation Top-1 | Validation ROC-AUC |
|---|---:|---:|
| Logistic Regression | 68.64% | 0.9071 |
| Gaussian Naive Bayes | 70.98% | 0.9113 |
| HistGradientBoosting | 72.92% | 0.9569 |
| RandomForest | **74.28%** | 0.9555 |
| XGBoost | 72.32% | 0.9548 |

The deployed RandomForest was subsequently refitted using sampled candidates from all 80 training trips. The validation scores above belong to the earlier trip-separated experiment, **not** to an untouched test of the refitted artifact.

Bayesian optimization used Optuna TPE. On its comparison protocol, baseline RandomForest Top-1 was 73.36% and tuned Top-1 was 73.44%; the tuned configuration had slightly worse macro F1 and log loss, so the baseline configuration was retained.

The HMM/Viterbi reversal penalty was assessed in a separate 16-trip validation ablation. A penalty of 5.0 improved ordered-route F1 and reduced reversals relative to a zero-penalty decoder.

On ten evaluator trips, the recorded post-development diagnostics were:

| Metric | Mean |
|---|---:|
| Route F1 (unique directed arc sets) | 95.24% |
| Ordered route F1 (sequence-sensitive) | 93.95% |
| Connectivity rate (macro) | 99.72% |

These evaluator results are **post-development diagnostics, not an independent untouched test**, because evaluator observations informed the reversal-penalty improvement. Route F1 is not GPS-point accuracy.

Reproducible experiment summaries are under `experiments/results/`, including `adaptive_model_comparison.csv`, `bayesian_comparison.csv`, `reversal_validation_summary.csv`, and `evaluator_summary.csv`.

## Reproduction and verification

Key scripts:

```powershell
python scripts/run_preprocessing.py
python scripts/train_models.py
python scripts/train_final_model.py
python scripts/tune_random_forest.py
python scripts/validate_reversal_penalty.py
python scripts/evaluate_evaluator.py
```

Check each script's input requirements and CLI options before rerunning expensive experiments. Training/evaluation datasets are required locally.

Tests and frontend checks:

```powershell
python -m pytest -q
cd ui
npm run lint
npm run build
```

The latest reported checks passed: 20 Python tests and frontend lint/build.

## Visualizations and submission

`reports/preprocessing/` contains EDA summaries and Folium map HTML outputs; `reports/figures/` contains existing model comparison plots. `scripts/visualize_prediction.py` supports prediction visualization. The final technical report should distinguish training, validation, student-test and evaluator partitions, include missing-value analysis, model comparison, Bayesian tuning, route decoding, geographical figures, and limitations. The official submission calls for a public GitHub repository and one consolidated PDF report.

## Limitations and data handling

- A reference `.route` is an ordered sequence of road arcs, not verified per-observation labels.
- Candidate-level Top-1, route-level F1, and sequence-level F1 measure different outcomes and must not be conflated.
- Evaluator data stays local and must not be committed.
- Single-device integration is confirmed; two-laptop deployment is not verified.
- Do not commit secrets or private credentials.

See `CONTRACTS.md`, `PROJECT_PLAN.md`, and `reports/preprocessing/README.md` for additional project details.
