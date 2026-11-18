# PathSnap – Shared Contracts

These contracts define the interfaces, data formats and rules that all four team members must follow during development.

All members must agree on these contracts before parallel implementation begins.

---

## Contract 1: Repository Layout

The shared repository structure is:

```text
PathSnap/
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── __init__.py
│   ├── features/
│   │   └── __init__.py
│   ├── models/
│   │   └── __init__.py
│   ├── hmm/
│   │   └── __init__.py
│   └── api/
│       └── __init__.py
├── ui/
├── experiments/
├── reports/
├── configs/
├── tests/
├── CONTRACTS.md
├── README.md
└── requirements.txt
```

All members must follow this structure. Any structural changes must be discussed and agreed upon by the team.

## Contract 2: Candidate Table Schema

The candidate dataset will be stored in Parquet format.

Each row represents one candidate road arc for one GPS observation.

Required columns:

- `trip_id`
- `obs_idx`
- `arc_id`
- Feature columns defined in the shared configuration
- `label`

The label indicates whether the candidate arc matches the reference route.

The shared feature column list will be maintained in:

`configs/config.yaml`

Person 1 is responsible for defining and freezing the final feature columns.

## Contract 3: Model Interface

Every trained classifier must follow the scikit-learn interface.

Required method:

`predict_proba(X)`

Models must:

- Accept the same feature columns in the same order.
- Return class probabilities.
- Be saved using Joblib.
- Use the shared random seed of 42 wherever applicable.

This ensures that models can be exchanged without changing the downstream pipeline.

## Contract 4: Decoder Interface

The HMM and Viterbi decoder must expose the following interface:

`decode(candidates_df, probs, road_graph)`

Inputs:

- `candidates_df`: Candidate road arcs for each GPS observation.
- `probs`: Emission probabilities from the trained classifier.
- `road_graph`: Road network connectivity graph.

Output:

- One predicted `arc_id` per GPS observation, in sequence order.

The decoder must use road connectivity when determining the most likely route.

## Contract 5: API Schema

The FastAPI backend must expose the following endpoints.

### GET /trips

Returns the ten evaluator trip IDs.

### POST /predict

Request:

```json
{
  "trip_id": "trip_id_here"
}
```

Returns:

- Predicted arc IDs.
- GPS points.
- Predicted route geometry.
- Accuracy against the reference route.

Ground truth must remain server-side and must never be exposed to the UI or prediction pipeline.

## Contract 6: Candidate Generation

A single shared function must be used for candidate generation during both model training and API inference.

Required interface:

`build_candidates(trip_path)`

The function must produce candidates using the same preprocessing and feature-generation logic in both situations.

This prevents differences between training-time and inference-time data processing.

Person 1 owns the implementation of this function.

## Contract 7: Data Splits and Evaluator Protection

The development dataset consists of:

- 80 training trips.
- 10 student-testing trips.

The file:

`configs/splits.json`

will contain only:

- The 80 training trip IDs.
- The 10 student-testing trip IDs.

The ten evaluator-testing trip IDs and their data must NOT be included in the repository.

Evaluator data:

- Is held exclusively by Person 4.
- Remains local on Laptop 1.
- Must never be committed or pushed to GitHub.
- Must only be used for the final deployment evaluation.

Evaluator data must not influence feature selection, model training, model selection, hyperparameter tuning or threshold selection.

## Contract 8: Coordinate Reference System

All geographic distance calculations must use a projected metric coordinate reference system.

Raw longitude and latitude values must not be used directly for distance calculations.

The initial shared configuration is:

`local_utm_by_trip_median_location` (WGS84 to the appropriate UTM zone inferred from each trip median GPS location)

Each trip must use its own appropriate UTM zone, inferred from the median GPS location during preprocessing.

The selected CRS must be consistent across feature generation, model training, HMM processing and API inference.

---

## Shared Rules

1. All members must follow these contracts.
2. Every member works on their own branch.
3. Changes to `main` must happen through pull requests.
4. Random seed 42 must be used wherever applicable.
5. No evaluator data, secrets or credentials may be committed.
6. All results must be reproducible from the repository code.
7. Any changes to shared interfaces must be discussed and agreed upon by all four members.
