
"""PathSnap trip-grouped cross-validation.

Evaluates candidate ranking accuracy without trip leakage.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data/processed/train_candidates.parquet"
CONFIG_PATH = ROOT / "reports/preprocessing/feature_config.json"
OUTPUT_PATH = ROOT / "experiments/results/adaptive_cross_validation.csv"

SEED = 42


def top1_accuracy(frame, probabilities):
    keys = pd.MultiIndex.from_frame(
        frame[["trip_id", "obs_idx"]]
    )

    group_ids, _ = pd.factorize(keys, sort=False)

    scored = pd.DataFrame({
        "group": group_ids,
        "probability": probabilities,
        "label": frame["label"].to_numpy(),
    })

    winners = scored.groupby("group")["probability"].idxmax()
    return float(scored.loc[winners, "label"].mean())


def sample_complete_observations(frame, max_rows, seed):
    """Sample complete candidate groups, never individual rows."""
    if len(frame) <= max_rows:
        return frame

    keys = pd.MultiIndex.from_frame(
        frame[["trip_id", "obs_idx"]]
    )
    groups, _ = pd.factorize(keys, sort=False)

    counts = np.bincount(groups)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(counts))

    selected = []
    total = 0

    for group_id in order:
        count = int(counts[group_id])

        if total + count > max_rows:
            continue

        selected.append(group_id)
        total += count

        if total >= max_rows * 0.98:
            break

    return frame.loc[np.isin(groups, selected)].copy()


def make_models():
    return {
        "RandomForest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=80,
                max_depth=18,
                min_samples_leaf=3,
                n_jobs=-1,
                random_state=SEED,
            ),
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
    }


def main():
    config = json.loads(
        CONFIG_PATH.read_text(encoding="utf-8")
    )
    features = config["feature_columns"]

    df = pd.read_parquet(
        DATA_PATH,
        columns=["trip_id", "obs_idx", "label", *features],
    )

    df[features] = df[features].replace(
        [np.inf, -np.inf], np.nan
    )

    splitter = GroupKFold(n_splits=5)
    results = []

    for fold, (train_idx, val_idx) in enumerate(
        splitter.split(df, groups=df["trip_id"]),
        start=1,
    ):
        train = df.iloc[train_idx]
        val = df.iloc[val_idx]

        train = sample_complete_observations(
            train,
            max_rows=400000,
            seed=SEED + fold,
        )

        ceiling = (
            val.groupby(["trip_id", "obs_idx"])["label"]
            .max()
            .mean()
        )

        print(
            f"\nFOLD {fold}/5 | "
            f"train rows={len(train):,} | "
            f"validation rows={len(val):,} | "
            f"candidate recall={ceiling:.2%}",
            flush=True,
        )

        for name, model in make_models().items():
            start = time.perf_counter()

            model.fit(
                train[features],
                train["label"],
            )

            probabilities = model.predict_proba(
                val[features]
            )[:, 1]

            top1 = top1_accuracy(val, probabilities)

            elapsed = time.perf_counter() - start

            result = {
                "fold": fold,
                "model": name,
                "top1_accuracy": top1,
                "candidate_recall": ceiling,
                "validation_observations": int(
                    val.groupby(["trip_id", "obs_idx"]).ngroups
                ),
                "validation_trips": int(val.trip_id.nunique()),
                "training_rows": len(train),
                "seconds": round(elapsed, 2),
            }

            results.append(result)

            print(
                f"{name}: Top-1={top1:.2%} "
                f"({elapsed:.1f}s)",
                flush=True,
            )

        OUTPUT_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        pd.DataFrame(results).to_csv(
            OUTPUT_PATH,
            index=False,
        )

    results_df = pd.DataFrame(results)

    print("\nCROSS-VALIDATION SUMMARY")

    for name, group in results_df.groupby("model"):
        weighted_top1 = np.average(
            group["top1_accuracy"],
            weights=group["validation_observations"],
        )

        weighted_ceiling = np.average(
            group["candidate_recall"],
            weights=group["validation_observations"],
        )

        print(
            f"{name}: "
            f"weighted Top-1={weighted_top1:.2%}, "
            f"candidate ceiling={weighted_ceiling:.2%}, "
            f"mean fold Top-1={group.top1_accuracy.mean():.2%}"
        )


if __name__ == "__main__":
    main()
