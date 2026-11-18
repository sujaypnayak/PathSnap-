# PathSnap preprocessing

## Run

The script reads only `student folder/Student folder/Student folder/train/` (80 trips)
and `student_test/` (10 trips). It validates those partition counts before processing
and does not search for or open any evaluator partition.

Install dependencies in a regular Python environment, then run the pilot and full pass:

```powershell
pip install -r requirements.txt
python scripts/run_preprocessing.py --pilot 5
python scripts/run_preprocessing.py
```

The full run writes combined Parquet files when PyArrow's Parquet engine is available.
If the host blocks PyArrow's native `_parquet` extension, the script writes clearly
named CSV fallbacks instead; it never gives CSV files a `.parquet` extension. On the
Codex Windows runtime used for this run, Windows Application Control blocked that
extension, so the executed outputs are CSV. Use a Python environment allowed to load
PyArrow to produce the requested Parquet files.

## Reusable API

```python
from src.features.candidates import build_candidates

# Inference: safe default; does not read a .route file and leaves label null.
inference_candidates = build_candidates(trip_path)

# Training only: explicitly opt into labels from the supplied reference route.
training_candidates = build_candidates(trip_path, include_labels=True)
```

Both modes use the same metric projection, spatial index, candidate search and feature
logic. The inference default never reads reference labels. Remove the `label` column
before passing features to a model.

## Assumptions and edge cases

- Input files are headerless, whitespace-separated: `.track` = lon/lat/time seconds,
  `.nodes` = lon/lat, `.arcs` = from/to node row IDs, `.route` = ordered arc row IDs.
- GPS points are transformed from WGS84 to the trip's local UTM zone inferred from
  median longitude and latitude. Distances and bearings are computed in projected meters.
- The STRtree indexes directed arc geometries, preserving reverse arcs as distinct
  candidates where the supplied arc table contains them.
- The candidate radius is 50 m. If no arc falls within it, the nearest arc is retained
  with `candidate_within_radius=0`, so every valid GPS observation has a candidate.
- Reference labels use the closest supplied route arc subject to nondecreasing route
  order across GPS observations. A label is positive only when that arc is in the
  observation's candidate set. No labels are inferred from predictions.
- Invalid GPS rows are skipped while original source row numbers remain in `obs_idx`.
  Nonpositive time differences are retained in EDA and produce zero-filled speed/time
  feature values; bearing/speed unavailable at endpoints or during stops are zero-filled.
- `stopped_flag` uses a 1 m/s GPS displacement-speed threshold. This is a simple
  threshold, not a learned vehicle-state estimate.
- Miss rates are computed against the assigned reference arc at 25, 50, 75, 100 and
  150 m, per trip and over all 90 development trips.

Outputs are in this directory: `train_candidates.*`, `student_test_candidates.*`,
`eda_summary.csv`, `candidate_miss_rates.csv`, `feature_config.json`, map HTML files,
and `run_summary.json`. `*.parquet` outputs are only present when Parquet writing works.
