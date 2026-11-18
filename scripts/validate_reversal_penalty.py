
"""Compare HMM reversal penalties on trip-separated validation data."""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.features.candidates import build_candidates, read_trip
from src.hmm.road_graph import build_road_graph
from src.hmm.viterbi import decode


ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = ROOT / "data" / "raw" / "Student folder" / "train"
RESULTS_DIR = ROOT / "experiments" / "results"

METADATA_PATH = RESULTS_DIR / "adaptive_training_metadata.json"
MODEL_PATH = ROOT / "models" / "best_adaptive_validation.joblib"

PENALTIES = [0.0, 1.0, 2.0, 5.0]


def collapse(sequence):
    result = []
    for value in sequence:
        value = int(value)
        if not result or result[-1] != value:
            result.append(value)
    return result


def lcs_length(a, b):
    if len(b) > len(a):
        a, b = b, a

    previous = [0] * (len(b) + 1)

    for value in a:
        current = [0] * (len(b) + 1)
        for j, other in enumerate(b, start=1):
            if value == other:
                current[j] = previous[j - 1] + 1
            else:
                current[j] = max(
                    previous[j],
                    current[j - 1],
                )
        previous = current

    return previous[-1]


def f1(precision, recall):
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def route_metrics(predicted, reference):
    predicted = collapse(predicted)
    reference = collapse(reference)

    pset = set(predicted)
    rset = set(reference)

    overlap = len(pset & rset)

    precision = overlap / len(pset) if pset else 0.0
    recall = overlap / len(rset) if rset else 0.0

    ordered_matches = lcs_length(predicted, reference)

    ordered_precision = (
        ordered_matches / len(predicted)
        if predicted else 0.0
    )
    ordered_recall = (
        ordered_matches / len(reference)
        if reference else 0.0
    )

    return {
        "route_f1": f1(precision, recall),
        "ordered_f1": f1(
            ordered_precision,
            ordered_recall,
        ),
        "predicted_route_entries": len(predicted),
        "reference_route_entries": len(reference),
    }


def count_reversals(predicted, arcs):
    starts = arcs["from_node"].to_numpy()
    ends = arcs["to_node"].to_numpy()

    reversals = 0

    for a, b in zip(predicted, predicted[1:]):
        a = int(a)
        b = int(b)

        if (
            a != b
            and starts[a] == ends[b]
            and ends[a] == starts[b]
        ):
            reversals += 1

    return reversals


def main():
    metadata = json.loads(
        METADATA_PATH.read_text(encoding="utf-8")
    )

    train_ids = set(metadata["training_trip_ids"])
    validation_ids = metadata["validation_trip_ids"]

    if train_ids.intersection(validation_ids):
        raise RuntimeError(
            "Training and validation trip IDs overlap."
        )

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(MODEL_PATH)

    model = joblib.load(MODEL_PATH)
    feature_columns = metadata["feature_columns"]

    results = []
    failures = []

    for index, trip_id in enumerate(validation_ids, start=1):
        trip_path = TRAIN_DIR / trip_id

        print(
            f"[{index}/{len(validation_ids)}] "
            f"Validation trip {trip_id}",
            flush=True,
        )

        try:
            trip = read_trip(
                trip_path,
                include_reference_route=True,
            )

            reference = trip["route"]

            if reference is None or len(reference) == 0:
                raise ValueError("Empty reference route")

            if np.any(reference < 0) or np.any(
                reference >= len(trip["arcs"])
            ):
                raise ValueError("Invalid reference arc ID")

            # Generate candidates without reference labels.
            candidates = build_candidates(
                trip_path,
                search_radius_m=50.0,
                min_candidates=5,
                max_radius_m=150.0,
                fallback_nearest=True,
                include_labels=False,
            )

            if candidates.empty:
                raise ValueError("No candidates generated")

            X = candidates[feature_columns].replace(
                [np.inf, -np.inf],
                np.nan,
            )

            probabilities = model.predict_proba(X)

            classes = list(model.classes_)
            if 1 not in classes:
                raise ValueError("Positive class missing")

            probs = np.clip(
                probabilities[:, classes.index(1)],
                1e-9,
                1.0,
            )

            road_graph = build_road_graph(
                trip_path / f"{trip_id}.arcs"
            )

            for penalty in PENALTIES:
                start = time.perf_counter()

                predicted = decode(
                    candidates,
                    probs,
                    road_graph,
                    reversal_penalty=penalty,
                )

                elapsed = time.perf_counter() - start

                metrics = route_metrics(
                    predicted,
                    reference,
                )

                reversals = count_reversals(
                    predicted,
                    trip["arcs"],
                )

                row = {
                    "trip_id": trip_id,
                    "reversal_penalty": penalty,
                    "gps_observations": len(predicted),
                    "reversal_count": reversals,
                    "reversal_rate": (
                        reversals / (len(predicted) - 1)
                        if len(predicted) > 1 else 0.0
                    ),
                    "decoding_seconds": round(elapsed, 2),
                    **metrics,
                }

                results.append(row)

                print(
                    f"  penalty={penalty:.1f} "
                    f"route_F1={metrics['route_f1']:.2%} "
                    f"ordered_F1={metrics['ordered_f1']:.2%} "
                    f"reversals={reversals}",
                    flush=True,
                )

        except Exception as error:
            failures.append({
                "trip_id": trip_id,
                "error": str(error),
            })
            print(f"  FAILED: {error}", flush=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if failures:
        pd.DataFrame(failures).to_csv(
            RESULTS_DIR / "reversal_validation_failures.csv",
            index=False,
        )

    if not results:
        raise RuntimeError("No validation results generated")

    df = pd.DataFrame(results)

    df.to_csv(
        RESULTS_DIR / "reversal_validation_per_trip.csv",
        index=False,
    )

    summary = (
        df.groupby("reversal_penalty", as_index=False)
        .agg(
            trips=("trip_id", "nunique"),
            mean_route_f1=("route_f1", "mean"),
            mean_ordered_f1=("ordered_f1", "mean"),
            total_reversals=("reversal_count", "sum"),
            mean_reversal_rate=("reversal_rate", "mean"),
            mean_decoding_seconds=("decoding_seconds", "mean"),
        )
    )

    summary.to_csv(
        RESULTS_DIR / "reversal_validation_summary.csv",
        index=False,
    )

    print("\nVALIDATION SUMMARY")
    print(summary.to_string(index=False))

    if failures:
        print(f"\nWARNING: {len(failures)} trips failed.")


if __name__ == "__main__":
    main()
