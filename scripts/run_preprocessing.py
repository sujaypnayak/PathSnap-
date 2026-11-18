
"""PathSnap preprocessing with adaptive road-arc candidates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features.candidates import (
    FEATURE_COLUMNS,
    build_candidates,
    prepare_trip,
)

DATA_ROOT = ROOT / "data" / "raw" / "Student folder"
DATA_OUT = ROOT / "data" / "processed"
REPORT_OUT = ROOT / "reports" / "preprocessing"

SEARCH_RADIUS_M = 50.0
MIN_CANDIDATES = 5
MAX_RADIUS_M = 150.0


def load_partitions() -> tuple[list[Path], list[Path]]:
    """Load only the authorized training and student-test splits."""
    splits_path = ROOT / "configs" / "splits.json"
    splits = json.loads(splits_path.read_text(encoding="utf-8"))

    train_ids = [str(x).zfill(8) for x in splits["train"]]

    student_ids = [
        str(x).zfill(8)
        for x in splits.get(
            "student_test",
            splits.get("student", []),
        )
    ]

    if len(train_ids) != 80 or len(student_ids) != 10:
        raise ValueError(
            "Expected 80 training and 10 student-test trips. "
            f"Found {len(train_ids)} and {len(student_ids)}."
        )

    train_paths = [
        DATA_ROOT / "train" / trip_id
        for trip_id in train_ids
    ]

    student_paths = [
        DATA_ROOT / "student_test" / trip_id
        for trip_id in student_ids
    ]

    for path in train_paths + student_paths:
        if not path.is_dir():
            raise FileNotFoundError(
                f"Missing trip directory: {path}"
            )

    return train_paths, student_paths


def process_partition(
    paths: list[Path],
    partition: str,
    output_path: Path,
) -> dict:
    """Build candidate datasets and collect coverage statistics."""
    include_labels = partition == "train"

    frames = []
    total_observations = 0
    total_hits = 0
    total_candidates = 0
    trip_summaries = []

    for index, path in enumerate(paths, start=1):
        print(
            f"[{partition}] {index}/{len(paths)} "
            f"Processing {path.name}",
            flush=True,
        )

        candidates = build_candidates(
            path,
            search_radius_m=SEARCH_RADIUS_M,
            min_candidates=MIN_CANDIDATES,
            max_radius_m=MAX_RADIUS_M,
            fallback_nearest=True,
            include_labels=include_labels,
        )

        if candidates.empty:
            raise RuntimeError(
                f"No candidates generated for {path.name}"
            )

        observations = candidates["obs_idx"].nunique()
        candidate_rows = len(candidates)

        total_observations += observations
        total_candidates += candidate_rows

        trip_info = {
            "trip_id": path.name,
            "partition": partition,
            "observations": int(observations),
            "candidate_rows": int(candidate_rows),
            "avg_candidates": float(
                candidate_rows / observations
            ),
        }

        if include_labels:
            # Each observation should have at most one
            # matching directed reference arc.
            positive_per_obs = candidates.groupby(
                "obs_idx"
            )["label"].sum()

            if (positive_per_obs > 1).any():
                raise RuntimeError(
                    f"Duplicate positive labels in {path.name}"
                )

            hits = int((positive_per_obs > 0).sum())
            total_hits += hits

            trip_info["candidate_hits"] = hits
            trip_info["candidate_recall"] = (
                hits / observations
            )

        trip_summaries.append(trip_info)
        frames.append(candidates)

    result = pd.concat(frames, ignore_index=True)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_parquet(
        output_path,
        index=False,
    )

    summary = {
        "partition": partition,
        "trips": len(paths),
        "observations": total_observations,
        "candidate_rows": total_candidates,
        "avg_candidates": (
            total_candidates / total_observations
        ),
        "output": str(output_path.relative_to(ROOT)),
    }

    if include_labels:
        summary["candidate_hits"] = total_hits
        summary["candidate_recall"] = (
            total_hits / total_observations
        )

    pd.DataFrame(trip_summaries).to_csv(
        REPORT_OUT / f"{partition}_candidate_summary.csv",
        index=False,
    )

    return summary


def run(pilot: int = 0) -> None:
    DATA_OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_paths, student_paths = load_partitions()

    if pilot:
        if pilot < 1 or pilot > len(train_paths):
            raise ValueError(
                "--pilot must be between 1 and 80"
            )

        train_paths = train_paths[:pilot]

        summary = process_partition(
            train_paths,
            "train",
            DATA_OUT / "pilot_train_candidates.parquet",
        )

        print(
            json.dumps(summary, indent=2),
            flush=True,
        )
        return

    train_summary = process_partition(
        train_paths,
        "train",
        DATA_OUT / "train_candidates.parquet",
    )

    student_summary = process_partition(
        student_paths,
        "student_test",
        DATA_OUT / "student_test_candidates.parquet",
    )

    config = {
        "search_radius_m": SEARCH_RADIUS_M,
        "min_candidates": MIN_CANDIDATES,
        "max_radius_m": MAX_RADIUS_M,
        "fallback_nearest": True,
        "feature_columns": FEATURE_COLUMNS,
        "label_policy": (
            "Monotone reference-route alignment; "
            "candidate generation does not use labels"
        ),
        "evaluator_data_accessed": False,
    }

    (REPORT_OUT / "feature_config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    summary = {
        "configuration": config,
        "train": train_summary,
        "student_test": student_summary,
    }

    (REPORT_OUT / "run_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\nPREPROCESSING COMPLETE")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pilot",
        type=int,
        default=0,
        help="Process the first N training trips only",
    )

    args = parser.parse_args()
    run(pilot=args.pilot)
