
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
import shapely

from src.features.candidates import prepare_trip


def audit_trip(path: Path) -> dict:
    g = prepare_trip(path, include_reference_route=True)

    ref_dist = np.asarray(g.label_distance, dtype=float)

    nearest = g.tree.nearest(g.points)
    network_dist = np.asarray(
        shapely.distance(g.points, g.lines[nearest]),
        dtype=float,
    )

    mismatch = (ref_dist > 50) & (network_dist <= 50)

    return {
        "trip_id": path.name,
        "observations": len(ref_dist),
        "reference_over_50_pct": 100 * np.mean(ref_dist > 50),
        "reference_over_200_pct": 100 * np.mean(ref_dist > 200),
        "network_over_50_pct": 100 * np.mean(network_dist > 50),
        "coverage_mismatch_pct": 100 * np.mean(mismatch),
        "reference_p95_m": np.percentile(ref_dist, 95),
        "network_p95_m": np.percentile(network_dist, 95),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        default="data/raw/Student folder/train",
    )
    parser.add_argument(
        "--splits",
        default="configs/splits.json",
    )
    parser.add_argument(
        "--output",
        default="reports/preprocessing/reference_coverage.csv",
    )
    args = parser.parse_args()

    splits = json.loads(Path(args.splits).read_text())
    rows = []

    for trip_id in splits["train"]:
        trip_id = str(trip_id).zfill(8)
        path = Path(args.data_root) / trip_id

        rows.append(audit_trip(path))
        print(f"Audited {trip_id}", flush=True)

    df = pd.DataFrame(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)

    print("\nTOP 15 COVERAGE MISMATCHES")
    print(
        df.sort_values(
            "coverage_mismatch_pct",
            ascending=False,
        ).head(15).round(2).to_string(index=False)
    )

    print("\nWEIGHTED DATASET SUMMARY")
    for column in [
        "reference_over_50_pct",
        "network_over_50_pct",
        "coverage_mismatch_pct",
    ]:
        value = np.average(
            df[column],
            weights=df["observations"],
        )
        print(f"{column}: {value:.2f}%")

    print(f"\nSaved report: {output}")


if __name__ == "__main__":
    main()
