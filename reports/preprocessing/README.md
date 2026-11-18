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

## Feature definitions (15 features)

- distance_m: GPS-to-road distance.
- gps_bearing_deg: GPS movement direction.
- road_bearing_deg: Candidate road direction.
- bearing_diff_deg: Difference between GPS and road direction.
- vehicle_speed_mps: Estimated movement speed in m/s.
- time_diff_s: Time since previous observation.
- stopped_flag: 1 when speed is below 1 m/s.
- prev_obs_distance_m: Candidate road distance to previous GPS observation.
- next_obs_distance_m: Candidate road distance to next GPS observation.
- prev_time_diff_s: Time since previous observation; currently same as time_diff_s.
- next_time_diff_s: Time to next observation.
- prev_bearing_deg: Previous GPS movement direction.
- next_bearing_deg: Next GPS movement direction.
- segment_length_m: Candidate road segment length.
- candidate_within_radius: 1 if candidate is within 50 m.

Unavailable feature values are zero-filled.

## Candidate miss-rate investigation

Miss rate here means the assigned reference-route distance exceeds the search radius. It does not mean that no candidate was generated, because the nearest-road fallback still supplies one.

| Radius | Reference-route distance beyond radius |
|---|---:|
| 25 m | 13.22% |
| 50 m | 11.17% |
| 75 m | 10.26% |
| 100 m | 9.85% |
| 150 m | 4.46% |

Trips 00000033 and 00000054 have high miss rates at 50 m. Trip 00000054 has aligned GPS and route bounds and connected route arcs, but large distance errors occur along parts of the trip. The exact cause is not confirmed. The 50 m default is retained based on the completed one-trip candidate-count and processing-time comparison

## Radius benchmark

### Completed one-trip comparison

A completed benchmark on one training trip compared both search radii:

| Search radius | Candidate rows | Processing time |
|---|---:|---:|
| 50 m | 15,964 | 5.01 seconds |
| 150 m | 86,999 | 21.94 seconds |

The 150 m radius generated approximately 5.45 times as many candidates and took approximately 4.4 times longer than the 50 m radius. The 50 m radius remains the default.

### Interrupted five-trip benchmark

A separate benchmark on the first five training trips generated 132,095 candidate rows at 50 m in 36.20 seconds. The 150 m run was interrupted before completion. Therefore, no completed five-trip comparison is available, and no 150 m timing or row count is reported for that experiment.
