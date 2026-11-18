
"""Evaluate frozen PathSnap predictions against evaluator reference routes.

Metrics:
- Unique-arc route precision, recall and F1
- Ordered route similarity using longest common subsequence (LCS)
- Prediction coverage and route transition connectivity

Reference routes are used ONLY after inference.
This script never trains or modifies the model.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.api.config import EVALUATOR_DATA_DIR
from src.api.prediction import predict_trip
from src.features.candidates import read_trip


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments" / "results"


def collapse_consecutive(sequence):
    """Remove repeated adjacent arc IDs while preserving route order."""
    result = []

    for value in sequence:
        value = int(value)
        if not result or result[-1] != value:
            result.append(value)

    return result


def lcs_length(first, second):
    """Length of longest common subsequence, using linear memory."""
    if len(second) > len(first):
        first, second = second, first

    previous = [0] * (len(second) + 1)

    for value in first:
        current = [0] * (len(second) + 1)

        for j, other in enumerate(second, start=1):
            if value == other:
                current[j] = previous[j - 1] + 1
            else:
                current[j] = max(
                    previous[j],
                    current[j - 1],
                )

        previous = current

    return previous[-1]


def calculate_metrics(predicted, reference):
    predicted_route = collapse_consecutive(predicted)
    reference_route = collapse_consecutive(reference)

    predicted_set = set(predicted_route)
    reference_set = set(reference_route)

    overlap = len(predicted_set & reference_set)

    precision = (
        overlap / len(predicted_set)
        if predicted_set else 0.0
    )
    recall = (
        overlap / len(reference_set)
        if reference_set else 0.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    ordered_matches = lcs_length(
        predicted_route,
        reference_route,
    )

    ordered_precision = (
        ordered_matches / len(predicted_route)
        if predicted_route else 0.0
    )
    ordered_recall = (
        ordered_matches / len(reference_route)
        if reference_route else 0.0
    )
    ordered_f1 = (
        2 * ordered_precision * ordered_recall
        / (ordered_precision + ordered_recall)
        if ordered_precision + ordered_recall else 0.0
    )

    return {
        "predicted_route_entries": len(predicted_route),
        "reference_route_entries": len(reference_route),
        "predicted_unique_arcs": len(predicted_set),
        "reference_unique_arcs": len(reference_set),
        "matching_unique_arcs": overlap,
        "route_precision": precision,
        "route_recall": recall,
        "route_f1": f1,
        "ordered_lcs_matches": ordered_matches,
        "ordered_precision": ordered_precision,
        "ordered_recall": ordered_recall,
        "ordered_f1": ordered_f1,
    }


def connectivity(predicted, arcs):
    """Check consecutive observation-level arc transitions."""
    from_nodes = arcs["from_node"].to_numpy()
    to_nodes = arcs["to_node"].to_numpy()

    total = max(0, len(predicted) - 1)
    connected = 0

    for current, following in zip(predicted, predicted[1:]):
        current = int(current)
        following = int(following)

        if current == following or (
            to_nodes[current] == from_nodes[following]
        ):
            connected += 1

    return {
        "transitions": total,
        "connected_transitions": connected,
        "connectivity_rate": (
            connected / total if total else 1.0
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=EVALUATOR_DATA_DIR,
    )
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()

    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"Evaluator directory not found: {data_dir}"
        )

    trip_dirs = sorted(
        path for path in data_dir.iterdir()
        if path.is_dir()
    )

    if not trip_dirs:
        raise RuntimeError("No evaluator trip folders found.")

    print(f"Evaluator trips found: {len(trip_dirs)}")
    print("Using frozen trained model; no retraining.\n")

    rows = []
    failures = []

    for index, trip_path in enumerate(trip_dirs, start=1):
        trip_id = trip_path.name

        print(
            f"[{index}/{len(trip_dirs)}] "
            f"Evaluating {trip_id}...",
            flush=True,
        )

        try:
            trip = read_trip(
                trip_path,
                include_reference_route=True,
            )

            reference = trip["route"]
            arcs = trip["arcs"]

            if reference is None or len(reference) == 0:
                raise ValueError("Reference route is empty.")

            if np.any(reference < 0) or np.any(
                reference >= len(arcs)
            ):
                raise ValueError(
                    "Reference contains invalid arc IDs."
                )

            start = time.perf_counter()

            # No reference labels are passed to prediction.
            prediction = predict_trip(trip_path)

            elapsed = time.perf_counter() - start

            predicted = prediction["predicted_arc_ids"]

            if not predicted:
                raise ValueError("Prediction is empty.")

            if np.any(np.asarray(predicted) < 0) or np.any(
                np.asarray(predicted) >= len(arcs)
            ):
                raise ValueError(
                    "Prediction contains invalid arc IDs."
                )

            result = {
                "trip_id": trip_id,
                "gps_observations": len(trip["track"]),
                "predicted_observations": len(predicted),
                "candidate_count": prediction["candidate_count"],
                "runtime_seconds": round(elapsed, 2),
            }

            result.update(
                calculate_metrics(predicted, reference)
            )
            result.update(connectivity(predicted, arcs))

            rows.append(result)

            print(
                f"  Route F1: {result['route_f1']:.2%} | "
                f"Ordered F1: {result['ordered_f1']:.2%} | "
                f"Connectivity: {result['connectivity_rate']:.2%} | "
                f"Time: {elapsed:.1f}s",
                flush=True,
            )

        except Exception as error:
            failures.append({
                "trip_id": trip_id,
                "error": str(error),
            })
            print(
                f"  FAILED: {error}",
                flush=True,
            )

    if not rows:
        raise RuntimeError(
            "All evaluator predictions failed."
        )

    OUTPUT.mkdir(parents=True, exist_ok=True)

    results = pd.DataFrame(rows)

    results.to_csv(
        OUTPUT / "evaluator_route_metrics.csv",
        index=False,
    )

    metric_columns = [
        "route_precision",
        "route_recall",
        "route_f1",
        "ordered_precision",
        "ordered_recall",
        "ordered_f1",
        "connectivity_rate",
    ]

    summary = {
        "trips_total": len(trip_dirs),
        "trips_successful": len(rows),
        "trips_failed": len(failures),
        "total_gps_observations": int(
            results["gps_observations"].sum()
        ),
        "total_runtime_seconds": round(
            float(results["runtime_seconds"].sum()), 2
        ),
    }

    for column in metric_columns:
        summary[f"macro_{column}"] = float(
            results[column].mean()
        )

    summary["micro_connectivity_rate"] = (
        float(results["connected_transitions"].sum())
        / float(results["transitions"].sum())
        if results["transitions"].sum() else 1.0
    )

    pd.DataFrame([summary]).to_csv(
        OUTPUT / "evaluator_summary.csv",
        index=False,
    )

    if failures:
        pd.DataFrame(failures).to_csv(
            OUTPUT / "evaluator_failures.csv",
            index=False,
        )

    print("\n" + "=" * 55)
    print("EVALUATOR TEST COMPLETE")
    print("=" * 55)

    print(
        f"Successful trips: {len(rows)}/{len(trip_dirs)}"
    )
    print(
        f"Mean route F1: "
        f"{summary['macro_route_f1']:.2%}"
    )
    print(
        f"Mean ordered route F1: "
        f"{summary['macro_ordered_f1']:.2%}"
    )
    print(
        f"Overall transition connectivity: "
        f"{summary['micro_connectivity_rate']:.2%}"
    )
    print(f"Results saved to: {OUTPUT}")

    if failures:
        print(
            f"WARNING: {len(failures)} trips failed. "
            "Inspect evaluator_failures.csv."
        )


if __name__ == "__main__":
    main()
