
"""Bayesian optimization of PathSnap RandomForest.

Uses trip-grouped cross-validation, complete observation candidate
groups, and Optuna's TPE sampler. Never touches evaluator data or
overwrites the deployed model.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/train_candidates.parquet"
CONFIG = ROOT / "reports/preprocessing/feature_config.json"
OUTPUT = ROOT / "experiments/results"
SEED = 42


def sample_complete_groups(df, limit, seed):
    """Sample complete GPS-observation groups, never individual rows."""
    if limit <= 0 or len(df) <= limit:
        return df.copy()

    keys = pd.MultiIndex.from_frame(
        df[["trip_id", "obs_idx"]]
    )
    group_ids, _ = pd.factorize(keys, sort=False)
    counts = np.bincount(group_ids)

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(counts))

    selected = []
    used = 0

    for group_id in order:
        count = int(counts[group_id])
        if used + count <= limit:
            selected.append(group_id)
            used += count
        if used >= limit * 0.98:
            break

    return df.loc[np.isin(group_ids, selected)].copy()


def top1_accuracy(df, probabilities):
    """Fraction of GPS observations whose top-scored arc is labeled 1."""
    scored = df[["trip_id", "obs_idx", "label"]].copy()
    scored["probability"] = probabilities

    winners = scored.loc[
        scored.groupby(
            ["trip_id", "obs_idx"], sort=False
        )["probability"].idxmax()
    ]

    return float(winners["label"].mean())


def metrics(df, probabilities):
    y = df["label"].astype(int).to_numpy()
    predictions = (probabilities >= 0.5).astype(int)

    return {
        "top1": top1_accuracy(df, probabilities),
        "macro_f1": float(
            f1_score(
                y, predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "log_loss": float(
            log_loss(
                y,
                np.clip(probabilities, 1e-9, 1 - 1e-9),
                labels=[0, 1],
            )
        ),
    }


def create_model(params):
    return make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(
            **params,
            random_state=SEED,
            n_jobs=-1,
        ),
    )


def main(args):
    OUTPUT.mkdir(parents=True, exist_ok=True)

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    features = config["feature_columns"]

    print("Loading training candidates...", flush=True)
    df = pd.read_parquet(
        DATA,
        columns=["trip_id", "obs_idx", "label", *features],
    )

    df["trip_id"] = df["trip_id"].astype(str).str.zfill(8)
    df["label"] = df["label"].astype(int)

    print(
        f"Loaded {len(df):,} rows from "
        f"{df.trip_id.nunique()} training trips",
        flush=True,
    )

    unique_trips = np.array(sorted(df["trip_id"].unique()))

    splitter = GroupKFold(n_splits=args.folds)
    folds = []

    for fold_id, (train_idx, valid_idx) in enumerate(
        splitter.split(
            unique_trips,
            groups=unique_trips,
        ),
        start=1,
    ):
        train_trips = set(unique_trips[train_idx])
        valid_trips = set(unique_trips[valid_idx])

        train_df = sample_complete_groups(
            df[df.trip_id.isin(train_trips)],
            args.train_rows,
            SEED + fold_id,
        )
        valid_df = sample_complete_groups(
            df[df.trip_id.isin(valid_trips)],
            args.valid_rows,
            SEED + 100 + fold_id,
        )

        if set(train_df.trip_id) & set(valid_df.trip_id):
            raise RuntimeError("Trip leakage detected.")

        if train_df["label"].nunique() < 2:
            raise RuntimeError(
                f"Fold {fold_id}: training sample has one class."
            )

        folds.append((train_df, valid_df))

        print(
            f"Fold {fold_id}: "
            f"train={len(train_df):,} rows, "
            f"validation={len(valid_df):,} rows, "
            f"validation trips={valid_df.trip_id.nunique()}",
            flush=True,
        )

    def evaluate(params, trial=None):
        fold_results = []

        for fold_id, (train_df, valid_df) in enumerate(
            folds, start=1
        ):
            model = create_model(params)

            x_train = train_df[features].replace(
                [np.inf, -np.inf], np.nan
            )
            x_valid = valid_df[features].replace(
                [np.inf, -np.inf], np.nan
            )

            model.fit(
                x_train,
                train_df["label"],
            )

            classes = list(model.classes_)
            if 1 not in classes:
                raise RuntimeError("Positive class missing.")

            probabilities = model.predict_proba(
                x_valid
            )[:, classes.index(1)]

            result = metrics(valid_df, probabilities)
            result["fold"] = fold_id
            result["observations"] = int(
                valid_df.groupby(
                    ["trip_id", "obs_idx"]
                ).ngroups
            )
            fold_results.append(result)

            if trial is not None:
                trial.report(
                    np.average(
                        [r["top1"] for r in fold_results],
                        weights=[
                            r["observations"]
                            for r in fold_results
                        ],
                    ),
                    step=fold_id,
                )

        weights = [
            result["observations"]
            for result in fold_results
        ]

        summary = {
            metric: float(
                np.average(
                    [result[metric] for result in fold_results],
                    weights=weights,
                )
            )
            for metric in ("top1", "macro_f1", "log_loss")
        }

        return summary, fold_results

    baseline_params = {
        "n_estimators": 100,
        "max_depth": 18,
        "min_samples_leaf": 3,
        "max_features": "sqrt",
        "min_samples_split": 2,
    }

    print("\nEvaluating baseline...", flush=True)
    baseline_start = time.perf_counter()
    baseline, baseline_folds = evaluate(baseline_params)

    print(
        f"Baseline Top-1: {baseline['top1']:.4f} | "
        f"Macro F1: {baseline['macro_f1']:.4f} | "
        f"Log Loss: {baseline['log_loss']:.4f} | "
        f"Time: {time.perf_counter()-baseline_start:.1f}s",
        flush=True,
    )

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int(
                "n_estimators", 60, 160, step=20
            ),
            "max_depth": trial.suggest_int(
                "max_depth", 10, 26, step=4
            ),
            "min_samples_leaf": trial.suggest_int(
                "min_samples_leaf", 1, 7
            ),
            "min_samples_split": trial.suggest_int(
                "min_samples_split", 2, 10, step=2
            ),
            "max_features": trial.suggest_categorical(
                "max_features", ["sqrt", 0.7, 1.0]
            ),
        }

        summary, _ = evaluate(params, trial=trial)

        for name, value in summary.items():
            trial.set_user_attr(name, value)

        print(
            f"Trial {trial.number + 1}/{args.trials}: "
            f"Top-1={summary['top1']:.4f}, "
            f"F1={summary['macro_f1']:.4f}, "
            f"LogLoss={summary['log_loss']:.4f}",
            flush=True,
        )

        return summary["top1"]

    sampler = optuna.samplers.TPESampler(seed=SEED)

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name="pathsnap_random_forest",
    )

    print(
        f"\nStarting {args.trials} Bayesian trials...",
        flush=True,
    )

    study.optimize(
        objective,
        n_trials=args.trials,
        gc_after_trial=True,
    )

    tuned_params = study.best_params
    tuned = {
        key: study.best_trial.user_attrs[key]
        for key in ("top1", "macro_f1", "log_loss")
    }

    comparison = pd.DataFrame([
        {"model": "Baseline RandomForest", **baseline},
        {"model": "Bayesian-tuned RandomForest", **tuned},
    ])
    comparison.to_csv(
        OUTPUT / "bayesian_comparison.csv",
        index=False,
    )

    study.trials_dataframe().to_csv(
        OUTPUT / "bayesian_trials.csv",
        index=False,
    )

    summary = {
        "method": "Optuna TPE Bayesian optimization",
        "objective": "grouped validation Top-1 accuracy",
        "trials": args.trials,
        "folds": args.folds,
        "max_train_rows_per_fold": args.train_rows,
        "max_validation_rows_per_fold": args.valid_rows,
        "baseline_params": baseline_params,
        "baseline_metrics": baseline,
        "baseline_fold_metrics": baseline_folds,
        "best_params": tuned_params,
        "best_trial": study.best_trial.number,
        "best_metrics": tuned,
        "note": (
            "Metrics are from the sampled tuning folds. "
            "The best-trial score is used for model selection "
            "and is not an independent test estimate. "
            "The deployed model is not overwritten."
        ),
    }

    (OUTPUT / "bayesian_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\nBAYESIAN OPTIMIZATION COMPLETE")
    print("Baseline Top-1:", round(baseline["top1"], 4))
    print("Tuned Top-1:", round(tuned["top1"], 4))
    print("Best parameters:", tuned_params)
    print("Results:", OUTPUT.resolve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--train-rows", type=int, default=100000)
    parser.add_argument("--valid-rows", type=int, default=100000)

    args = parser.parse_args()

    if args.trials < 1 or args.folds < 2:
        parser.error("Use at least 1 trial and 2 folds.")

    main(args)
