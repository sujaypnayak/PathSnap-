# PathSnap preprocessing

## Run

The preprocessing script reads the 80 training trips and 10 student-test trips from:

- `data/raw/Student folder/train/`
- `data/raw/Student folder/student_test/`

It validates both partition counts and does not access evaluator data.

Run using Python 3.13:

```cmd
py -3.13 -m pip install -r requirements.txt
py -3.13 scripts/run_preprocessing.py --pilot 5
py -3.13 scripts/run_preprocessing.py
```

## Outputs

Candidate datasets are written to `data/processed/`:

- `train_candidates.parquet`
- `student_test_candidates.parquet`

EDA reports, candidate miss-rate reports, feature configuration, maps and run summary are written to `reports/preprocessing/`.

The completed run in the Python 3.13 environment successfully used PyArrow and generated Parquet files. The script also supports clearly named CSV fallbacks when Parquet is unavailable; CSV data is never written with a `.parquet` extension.

## Reusable API

```python
from src.features.candidates import build_candidates

# Inference: does not read reference-route labels.
inference_candidates = build_candidates(trip_path)

# Training: explicitly enables reference-route labels.
training_candidates = build_candidates(trip_path, include_labels=True)
```

Both modes use the same metric projection, spatial index, candidate search and feature logic. The inference default does not read reference labels. Remove the `label` column before passing features to a model.

## Assumptions and edge cases

- Input files are headerless, whitespace-separated: `.track` = lon/lat/time seconds, `.nodes` = lon/lat, `.arcs` = from/to node row IDs, `.route` = ordered arc row IDs.
- GPS points are transformed from WGS84 to the trip's local UTM zone inferred from median longitude and latitude.
- The STRtree indexes directed arc geometries, preserving reverse arcs as distinct candidates where supplied.
- The candidate search radius is 50 m. If no arc falls within it, the nearest arc is retained with `candidate_within_radius=0`.
- Reference labels use the closest supplied route arc subject to nondecreasing route order across GPS observations. Labels are enabled only for training data; student-test labels remain null.
- Invalid GPS rows are skipped while original source row numbers remain in `obs_idx`.
- Nonpositive time differences are retained in EDA and produce zero-filled speed/time feature values. Unavailable endpoint bearings and speeds are zero-filled.
- `stopped_flag` uses a 1 m/s GPS displacement-speed threshold; it is not a learned vehicle-state estimate.
- Training candidate miss rates are evaluated against assigned reference arcs at 25, 50, 75, 100 and 150 m.

The completed run generated 1,885,382 training candidate rows and 209,121 student-test candidate rows, using a 50 m radius. The run summary records `evaluator_data_accessed: false`.
