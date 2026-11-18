#!/usr/bin/env python
"""
Rebuild candidates_student.parquet WITH labels, using the team's own candidates.py
(build_candidates(..., include_labels=True)), for the 10 student-test trips only.

Run from the repo root (PowerShell, one line):
  py experiments\\build_student_candidates.py --candidates-py src\\candidates.py --raw-dir data\\raw --splits data\\processed\\splits.json --train-parquet data\\processed\\candidates.parquet --out data\\processed\\candidates_student.parquet

--candidates-py : path to Person 1's candidates.py (find it with: dir src -Recurse -Filter candidates.py)
--raw-dir       : folder that contains one sub-folder per trip (e.g. 00000007\\00000007.track ...).
                  The script searches inside it, so any depth works.
--check-trip    : a TRAINING trip used to prove this script reproduces Person 1's training rows
                  (default: first trip in the train list). If it says MISMATCH, the settings differ
                  from what was used for training - do not use the output, ask Person 1.

Only trip IDs listed under "student_test" in the splits file are ever processed.
Evaluator trips are never touched.
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def load_module(path):
    path = Path(path).resolve()
    sys.path.insert(0, str(path.parent))  # lets candidates.py import sibling modules if it needs to
    spec = importlib.util.spec_from_file_location("team_candidates", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["team_candidates"] = mod  # needed for @dataclass inside candidates.py
    spec.loader.exec_module(mod)
    return mod


def norm_id(x):
    return str(x).zfill(8)


def find_trip_dir(raw_dir, trip_id):
    direct = Path(raw_dir) / trip_id
    if (direct / f"{trip_id}.track").exists():
        return direct
    for p in Path(raw_dir).rglob(trip_id):
        if p.is_dir() and (p / f"{trip_id}.track").exists():
            return p
    for t in Path(raw_dir).rglob(f"{trip_id}.track"):
        return t.parent
    raise FileNotFoundError(f"No folder with {trip_id}.track found under {raw_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates-py", required=True)
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--train-parquet", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--radius", type=float, default=50.0, help="keep at the value used for training (default 50)")
    ap.add_argument("--check-trip", default=None)
    args = ap.parse_args()

    mod = load_module(args.candidates_py)
    sp = json.load(open(args.splits))
    train_ids = [norm_id(t) for t in sp["train"]]
    student_ids = [norm_id(t) for t in sp["student_test"]]
    assert not set(train_ids) & set(student_ids), "overlap between train and student_test"

    train = pd.read_parquet(args.train_parquet)
    train["trip_id"] = train["trip_id"].map(norm_id)
    feature_cols = list(mod.FEATURE_COLUMNS)

    # ---- 1. reproducibility check on one TRAINING trip
    chk = norm_id(args.check_trip) if args.check_trip else train_ids[0]
    assert chk in train_ids, f"{chk} is not a training trip"
    print(f"[check] rebuilding training trip {chk} and comparing with {args.train_parquet} ...")
    mine = mod.build_candidates(find_trip_dir(args.raw_dir, chk), search_radius_m=args.radius, include_labels=True)
    mine["trip_id"] = mine["trip_id"].map(norm_id)
    ref = train[train["trip_id"] == chk]
    ok = len(mine) == len(ref)
    if ok:
        a = mine.sort_values(["obs_idx", "arc_id"]).reset_index(drop=True)
        b = ref.sort_values(["obs_idx", "arc_id"]).reset_index(drop=True)
        ok = (a["arc_id"].to_numpy() == b["arc_id"].to_numpy()).all() \
            and (a["label"].astype(int).to_numpy() == b["label"].astype(int).to_numpy()).all() \
            and np.allclose(a[feature_cols].to_numpy(float), b[feature_cols].to_numpy(float), atol=1e-4, equal_nan=True)
    print(f"[check] rows mine={len(mine)} reference={len(ref)} -> {'MATCH' if ok else 'MISMATCH'}")
    if not ok:
        sys.exit("The rebuilt training trip does not match Person 1's training file. Settings differ "
                 "(radius, code version, ...). Do not use this output - ask Person 1 which settings they used.")

    # ---- 2. build the student trips with labels
    parts = []
    for tid in student_ids:
        d = find_trip_dir(args.raw_dir, tid)
        if not (d / f"{tid}.route").exists():
            sys.exit(f"{tid}: no {tid}.route file in {d}. Student trips need their reference route for labels.")
        df = mod.build_candidates(d, search_radius_m=args.radius, include_labels=True)
        df["trip_id"] = df["trip_id"].map(norm_id)
        print(f"  {tid}: {len(df):,} rows, positive rate {df['label'].mean():.3%}")
        parts.append(df)
    out = pd.concat(parts, ignore_index=True)
    out["label"] = out["label"].astype(int)

    # ---- 3. sanity checks, then save
    assert out["label"].isin([0, 1]).all()
    assert list(out.columns) == list(train.columns), "column order/names differ from the training file"
    assert set(out["trip_id"]) == set(student_ids), "unexpected trip IDs in output"
    g = out.groupby(["trip_id", "obs_idx"])["label"].max()
    print(f"\nrows {len(out):,} | trips {out['trip_id'].nunique()} | positives {out['label'].mean():.3%} "
          f"(training: {train['label'].mean():.3%}) | candidate miss rate {1 - g.mean():.3%}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
