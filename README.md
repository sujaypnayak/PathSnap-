# PathSnap

A distributed machine learning system for GPS map matching and route reconstruction using candidate classification, Hidden Markov Models (HMM), and Viterbi decoding.

## 1. Project Overview

PathSnap reconstructs vehicle routes from noisy GPS observations by identifying the most likely road segments travelled.

The system follows a two-stage pipeline:

1. **Candidate Classification:** Generates nearby road-segment candidates for each GPS observation and assigns probabilities using a trained machine learning classifier.
2. **HMM and Viterbi Decoding:** Uses road-network connectivity and candidate probabilities to reconstruct a coherent sequence of road segments.

## 2. Technology Stack

- **Backend:** Python, FastAPI
- **Frontend:** React, Vite
- **Machine Learning:** scikit-learn, NumPy, Pandas
- **Route Decoding:** NetworkX, HMM, Viterbi
- **Geospatial Processing:** Shapely, PyProj
- **Model Storage:** Joblib
- **Map Visualization:** Leaflet, React Leaflet
- **Data Storage:** Parquet

## 3. Project Structure

```text
PathSnap/
├── configs/             # Shared configuration and data splits
├── data/                # Processed datasets
├── experiments/         # Experiment and connectivity results
├── reports/             # EDA and preprocessing reports
├── scripts/             # Preprocessing scripts
├── src/
│   ├── api/             # FastAPI backend and prediction pipeline
│   ├── features/        # Candidate generation and features
│   ├── hmm/             # Road graph and Viterbi decoder
│   └── models/          # Trained model artifacts
├── tests/               # Unit tests
├── ui/                  # React frontend
├── CONTRACTS.md
├── PROJECT_PLAN.md
├── README.md
└── requirements.txt
```

## 4. Setup and Installation

### Backend

Python 3.12 or a compatible Python version is recommended.

From the project root:

```cmd
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Frontend

Open a separate terminal:

```cmd
cd ui
npm install
```

## 5. Dataset Configuration

The backend dataset location is configured in:

`src/api/config.py`

Update `DATASET_ROOT` to the local directory containing the dataset.

Expected structure:

```text
Dataset/
├── train/
│   └── <trip_id>/
│       ├── <trip_id>.track
│       ├── <trip_id>.nodes
│       ├── <trip_id>.arcs
│       └── <trip_id>.route
├── student_test/
│   └── <trip_id>/
│       ├── <trip_id>.track
│       ├── <trip_id>.nodes
│       ├── <trip_id>.arcs
│       └── <trip_id>.route
└── evaluator/
```

Raw datasets and evaluator data must remain local and must not be committed to GitHub.

The evaluator dataset is maintained exclusively by Person 4 and must only be used for final deployment evaluation.

## 6. Running the Application

### Start the Backend

From the project root, with the virtual environment activated:

```cmd
uvicorn src.api.main:app --reload
```

Backend address:

`http://127.0.0.1:8000`

API documentation:

`http://127.0.0.1:8000/docs`

### Start the Frontend

Open another terminal:

```cmd
cd ui
npm run dev
```

Frontend address:

`http://localhost:5173`

The frontend communicates with the backend through the configured Vite proxy.

## 7. API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | API information |
| GET | `/health` | Health check |
| GET | `/trips` | Lists evaluator trips when the evaluator dataset is configured |
| GET | `/dev/trips` | Lists student-testing trips |
| POST | `/predict` | Requests route prediction |

Example prediction request:

```json
{
  "trip_id": "00000000"
}
```

The prediction response includes trip information, GPS observations, prediction status and, when available, predicted arc IDs.

Ground-truth information must remain server-side and must never be exposed to the frontend.

## 8. Testing

Run backend tests from the project root:

```cmd
pytest -q
```

Build the frontend:

```cmd
cd ui
npm run build
```

## 9. Development Rules

- Each team member works on an assigned Git branch.
- Changes to `main` are submitted through pull requests.
- The shared random seed is 42 wherever applicable.
- Model interfaces must follow the scikit-learn `predict_proba(X)` convention.
- Candidate generation must use the shared `build_candidates()` function.
- The HMM decoder must follow `decode(candidates_df, probs, road_graph)`.
- Evaluator data, credentials and secrets must never be committed.
- Results must be reproducible.

Refer to [CONTRACTS.md](CONTRACTS.md) for the complete shared interfaces and development rules.

## 10. Team

PathSnap is developed collaboratively by a four-member team as a machine learning project.
