
"""Validate PathSnap inference on all student-test trips."""

from __future__ import annotations

import csv
import time
from pathlib import Path

from src.api.config import STUDENT_TEST_DIR
from src.api.prediction import predict_trip
from src.hmm.road_graph import build_road_graph


def main():
    trip_paths = sorted(
        path
        for path in STUDENT_TEST_DIR.iterdir()
        if path.is_dir()
        and (path / f"{path.name}.track").is_file()
    )

    if not trip_paths:
        raise RuntimeError("No student-test trips found.")

    results = []

    for index, trip_path in enumerate(trip_paths, start=1):
        trip_id = trip_path.name

        print(
            f"\n[{index}/{len(trip_paths)}] {trip_id}",
            flush=True,
        )

        start_time = time.perf_counter()

        try:
            prediction = predict_trip(trip_path)
            elapsed = time.perf_counter() - start_time

            arc_ids = prediction["predicted_arc_ids"]

            graph = build_road_graph(
                trip_path / f"{trip_id}.arcs"
            )

            same = 0
            connected = 0
            nonadjacent = 0

            for previous, current in zip(
                arc_ids, arc_ids[1:]
            ):
                if previous == current:
                    same += 1
                elif graph.has_edge(previous, current):
                    connected += 1
                else:
                    nonadjacent += 1

            observations = prediction["observation_count"]

            if len(arc_ids) != observations:
                raise ValueError(
                    "Predicted arc count differs from "
                    "observation count."
                )

            row = {
                "trip_id": trip_id,
                "status": "success",
                "observations": observations,
                "candidates": prediction["candidate_count"],
                "predicted_arcs": len(arc_ids),
                "same_arc": same,
                "connected": connected,
                "nonadjacent": nonadjacent,
                "runtime_seconds": round(elapsed, 2),
                "error": "",
            }

        except Exception as error:
            row = {
                "trip_id": trip_id,
                "status": "failed",
                "observations": 0,
                "candidates": 0,
                "predicted_arcs": 0,
                "same_arc": 0,
                "connected": 0,
                "nonadjacent": 0,
                "runtime_seconds": round(
                    time.perf_counter() - start_time, 2
                ),
                "error": str(error),
            }

        results.append(row)

        print(
            f"Status: {row['status']} | "
            f"Observations: {row['observations']} | "
            f"Nonadjacent: {row['nonadjacent']} | "
            f"Time: {row['runtime_seconds']}s",
            flush=True,
        )

    output_dir = Path("experiments/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = (
        output_dir / "student_test_inference_validation.csv"
    )

    with output_file.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(results[0].keys()),
        )
        writer.writeheader()
        writer.writerows(results)

    successful = sum(
        row["status"] == "success"
        for row in results
    )

    total_nonadjacent = sum(
        row["nonadjacent"]
        for row in results
    )

    print("\nFINAL STUDENT-TEST VALIDATION")
    print("-----------------------------")
    print("Trips tested:", len(results))
    print("Successful:", successful)
    print("Failed:", len(results) - successful)
    print("Total nonadjacent transitions:", total_nonadjacent)
    print("Report:", output_file.resolve())

    if successful != len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
