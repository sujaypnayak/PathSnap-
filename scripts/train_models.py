
"""Train and compare PathSnap road-arc ranking models.

Run:
    python -m scripts.train_models
    python -m scripts.train_models --max-train-rows 500000
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data/processed/train_candidates.parquet"
CONFIG_PATH = ROOT / "reports/preprocessing/feature_config.json"
OUTPUT_DIR = ROOT / "models"
REPORT_DIR = ROOT / "experiments/results"

SEED = 42


def load_data():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    features = config["feature_columns"]

    columns = ["trip_id", "obs_idx", "label", *features]
    df = pd.read_parquet(DATA_PATH, columns=columns)

    if df["label"].isna().any():
        raise ValueError("Training labels contain missing values.")

    df["label"] = df["label"].astype(np.int8)

    if not df["label"].isin([0, 1]).all():
        raise ValueError("Labels must be binary.")

    return df, features


def top1_accuracy(frame, probabilities):
    """Fraction of GPS observations whose top-ranked arc is correct."""
    keys = pd.MultiIndex.from_frame(frame[["trip_id", "obs_idx"]])

    scores = pd.DataFrame({
        "group": pd.factorize(keys, sort=False)[0],
        "probability": probabilities,
        "label": frame["label"].to_numpy(),
    })

    winners = scores.groupby("group", sort=False)["probability"].idxmax()
    return float(scores.loc[winners, "label"].mean())


def sample_training_rows(frame, maximum):
    """Sample entire observations to preserve their candidate sets."""
    if maximum <= 0 or len(frame) <= maximum:
        return frame

    observation_keys = pd.MultiIndex.from_frame(
        frame[["trip_id", "obs_idx"]]
    )
    groups, _ = pd.factorize(observation_keys, sort=False)

    counts = np.bincount(groups)
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(counts))

    selected = []
    used = 0

    for group_id in order:
        count = int(counts[group_id])
        if used + count > maximum:
            continue

        selected.append(group_id)
        used += count

        if used >= maximum * 0.98:
            break

    mask = np.isin(groups, selected)
    return frame.loc[mask].copy()


def create_models():
    models = {
        "LogisticRegression": make_pipeline(
            SimpleImputer(strategy="median"),
            LogisticRegression(
                max_iter=250,
                class_weight="balanced",
                solver="lbfgs",
                random_state=SEED,
            ),
        ),
        "GaussianNB": make_pipeline(
            SimpleImputer(strategy="median"),
            GaussianNB(),
        ),
        "HistGradientBoosting": make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingClassifier(
                max_iter=120,
                max_leaf_nodes=31,
                learning_rate=0.08,
                random_state=SEED,
            ),
        ),
        "RandomForest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=100,
                max_depth=18,
                min_samples_leaf=3,
                n_jobs=-1,
                random_state=SEED,
            ),
        ),
    }

    try:
        from xgboost import XGBClassifier

        models["XGBoost"] = make_pipeline(
            SimpleImputer(strategy="median"),
            XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.07,
                subsample=0.85,
                colsample_bytree=0.85,
                tree_method="hist",
                n_jobs=4,
                random_state=SEED,
                eval_metric="logloss",
            ),
        )
    except ImportError:
        print("XGBoost not installed; skipping it.")

    return models


def main(max_train_rows):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading adaptive candidate dataset...", flush=True)
    df, features = load_data()

    unique_trips = df["trip_id"].nunique()
    print(
        f"Loaded {len(df):,} candidate rows "
        f"from {unique_trips} trips.",
        flush=True,
    )

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=SEED,
    )

    train_idx, val_idx = next(
        splitter.split(df, groups=df["trip_id"])
    )

    train_df = df.iloc[train_idx].copy()
    val_df = df.iloc[val_idx].copy()

    train_df = sample_training_rows(
        train_df,
        max_train_rows,
    )

    print(
        f"Training rows: {len(train_df):,}; "
        f"validation rows: {len(val_df):,}",
        flush=True,
    )

    print(
        f"Training trips: {train_df.trip_id.nunique()}; "
        f"validation trips: {val_df.trip_id.nunique()}",
        flush=True,
    )

    x_train = train_df[features].replace(
        [np.inf, -np.inf], np.nan
    )
    y_train = train_df["label"].to_numpy()

    x_val = val_df[features].replace(
        [np.inf, -np.inf], np.nan
    )
    y_val = val_df["label"].to_numpy()

    candidate_ceiling = float(
        val_df.groupby(["trip_id", "obs_idx"])["label"]
        .max()
        .mean()
    )

    print(
        f"Validation candidate-recall ceiling: "
        f"{candidate_ceiling:.4%}",
        flush=True,
    )

    results = []
    best_name = None
    best_score = -1.0
    best_model = None

    for name, model in create_models().items():
        print(f"\nTraining {name}...", flush=True)
        start = time.perf_counter()

        try:
            model.fit(x_train, y_train)

            probabilities = model.predict_proba(x_val)[:, 1]
            top1 = top1_accuracy(val_df, probabilities)

            try:
                auc = roc_auc_score(y_val, probabilities)
            except ValueError:
                auc = float("nan")

            elapsed = time.perf_counter() - start

            result = {
                "model": name,
                "validation_top1_accuracy": top1,
                "validation_roc_auc": auc,
                "training_seconds": round(elapsed, 2),
                "training_rows": len(train_df),
                "validation_rows": len(val_df),
                "validation_candidate_recall": candidate_ceiling,
            }

            results.append(result)

            print(
                f"{name}: Top-1={top1:.2%}, "
                f"AUC={auc:.4f}, "
                f"time={elapsed:.1f}s",
                flush=True,
            )

            if top1 > best_score:
                best_score = top1
                best_name = name
                best_model = model

        except Exception as exc:
            print(
                f"{name} FAILED: {type(exc).__name__}: {exc}",
                flush=True,
            )
            results.append({
                "model": name,
                "error": str(exc),
            })

    if best_model is None:
        raise RuntimeError("No model trained successfully.")

    results_df = pd.DataFrame(results)
    results_df.to_csv(
        REPORT_DIR / "adaptive_model_comparison.csv",
        index=False,
    )

    # This artifact is fitted on the training split only.
    # It is appropriate for validation and integration checks,
    # not yet a final model refitted on all 80 trips.
    model_path = OUTPUT_DIR / "best_adaptive_validation.joblib"
    joblib.dump(best_model, model_path)

    metadata = {
        "best_model": best_name,
        "validation_top1_accuracy": best_score,
        "validation_candidate_recall": candidate_ceiling,
        "feature_columns": features,
        "model_path": str(model_path.relative_to(ROOT)),
        "training_trip_ids": sorted(
            train_df.trip_id.astype(str).unique().tolist()
        ),
        "validation_trip_ids": sorted(
            val_df.trip_id.astype(str).unique().tolist()
        ),
        "max_train_rows": max_train_rows,
        "random_seed": SEED,
        "evaluation_method": "GroupShuffleSplit by trip",
        "student_test_labels_used": False,
    }

    (REPORT_DIR / "adaptive_training_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\nMODEL COMPARISON")
    print(
        results_df.to_string(index=False),
        flush=True,
    )

    print(f"\nBest model: {best_name}")
    print(f"Validation Top-1: {best_score:.2%}")
    print(f"Saved: {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=500000,
        help=(
            "Maximum training candidate rows. "
            "Use 0 for all training rows."
        ),
    )
    args = parser.parse_args()
    main(args.max_train_rows)
