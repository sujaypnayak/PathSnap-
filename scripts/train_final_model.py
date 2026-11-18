
"""Fit the deployable PathSnap RandomForest on all training trips."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/train_candidates.parquet"
CONFIG = ROOT / "reports/preprocessing/feature_config.json"
MODEL = ROOT / "src/models/best_model.joblib"
METADATA = ROOT / "src/models/best_model_metadata.json"

SEED = 42


def sample_observations(df, max_rows):
    if max_rows == 0 or len(df) <= max_rows:
        return df

    keys = pd.MultiIndex.from_frame(
        df[["trip_id", "obs_idx"]]
    )
    groups, _ = pd.factorize(keys, sort=False)
    counts = np.bincount(groups)

    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(counts))

    selected = []
    used = 0

    for group_id in order:
        count = int(counts[group_id])

        if used + count > max_rows:
            continue

        selected.append(group_id)
        used += count

        if used >= 0.98 * max_rows:
            break

    return df.loc[np.isin(groups, selected)].copy()


def main(max_rows):
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    features = config["feature_columns"]

    print("Loading training data...", flush=True)
    df = pd.read_parquet(
        DATA,
        columns=["trip_id", "obs_idx", "label", *features],
    )

    print(f"Available candidate rows: {len(df):,}")
    print(f"Available trips: {df.trip_id.nunique()}")

    df = sample_observations(df, max_rows)

    print(f"Selected candidate rows: {len(df):,}")
    print(f"Selected trips: {df.trip_id.nunique()}")

    if df.trip_id.nunique() != 80:
        raise RuntimeError(
            "Final training sample does not cover all 80 trips."
        )

    x = df[features].replace(
        [np.inf, -np.inf], np.nan
    )
    y = df["label"].astype(np.int8)

    model = make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(
            n_estimators=100,
            max_depth=18,
            min_samples_leaf=3,
            n_jobs=-1,
            random_state=SEED,
        ),
    )

    print("Fitting final RandomForest...", flush=True)
    start = time.perf_counter()
    model.fit(x, y)
    elapsed = time.perf_counter() - start

    MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL)

    metadata = {
        "model": "RandomForest",
        "model_path": str(MODEL.relative_to(ROOT)),
        "feature_columns": features,
        "training_rows": int(len(df)),
        "training_trips": int(df.trip_id.nunique()),
        "training_seconds": round(elapsed, 2),
        "search_radius_m": config["search_radius_m"],
        "min_candidates": config["min_candidates"],
        "max_radius_m": config["max_radius_m"],
        "random_seed": SEED,
        "held_out_cv_top1": 0.7095,
        "cv_note": (
            "Cross-validation used smaller per-fold training "
            "samples; this refitted model has no independent "
            "held-out evaluation."
        ),
    }

    METADATA.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\nFINAL MODEL SAVED")
    print(f"Model: {MODEL}")
    print(f"Training trips: {df.trip_id.nunique()}")
    print(f"Training rows: {len(df):,}")
    print(f"Training time: {elapsed:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-rows",
        type=int,
        default=1000000,
        help="Candidate-row cap; 0 means use all rows",
    )
    args = parser.parse_args()
    main(args.max_rows)
