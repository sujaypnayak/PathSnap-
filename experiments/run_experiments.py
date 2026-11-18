#!/usr/bin/env python
"""
PathSnap - Person 2 experiment harness (experiments/run_experiments.py)
"""
import argparse
import json
import platform
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, recall_score
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

SEED = 42
THRESHOLD = 0.5
KEYS = ["trip_id", "obs_idx"]

def norm_id(x):
    return str(x).zfill(8)

def model_zoo():
    return {
        "logreg": (
            "Linear", 
            lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)), 
            {"C": 1.0, "max_iter": 1000, "scaler": "StandardScaler"}
        ),
        "gaussian_nb": (
            "Probabilistic", 
            lambda: GaussianNB(), 
            {"var_smoothing": 1e-9}
        ),
        "hist_gb": (
            "Boosting (Sklearn)", 
            lambda: HistGradientBoostingClassifier(learning_rate=0.1, max_iter=200, max_depth=None, random_state=SEED), 
            {"learning_rate": 0.1, "max_iter": 200, "max_depth": None}
        ),
        "random_forest": (
            "Ensemble (Bagging)", 
            lambda: RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_leaf=5, n_jobs=-1, random_state=SEED), 
            {"n_estimators": 100, "max_depth": 20, "min_samples_leaf": 5}
        ),
        "xgboost": (
            "Boosting (XGB)", 
            lambda: XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=SEED, n_jobs=-1, eval_metric='logloss'), 
            {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 6}
        ),
        "mlp": (
            "Neural network", 
            lambda: make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-4, max_iter=200, early_stopping=True, random_state=SEED)), 
            {"hidden_layer_sizes": [64, 32], "alpha": 1e-4, "max_iter": 200, "early_stopping": True}
        ),
        "knn": (
            "Instance-based", 
            lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15, weights="distance", n_jobs=-1)), 
            {"n_neighbors": 15, "weights": "distance", "scaler": "StandardScaler"}
        ),
    }

DEFAULT_MODELS = ["logreg", "gaussian_nb", "hist_gb", "random_forest", "xgboost"]

def evaluate(df, p):
    valid_mask = df["label"].notna()
    df = df[valid_mask]
    p = p[valid_mask]
    if len(df) == 0:
        return {"macro_f1": 0, "top1_acc": 0, "log_loss": 0, "candidate_miss_rate": 1.0,
                "macro_precision": 0, "macro_recall": 0, "accuracy_cand": 0, "top1_acc_given_hit": 0, "n_obs": 0}
    y = df["label"].to_numpy().astype(int)
    pred = (p >= THRESHOLD).astype(int)
    out = {
        "accuracy_cand": accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
        "macro_precision": precision_score(y, pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y, pred, average="macro", zero_division=0),
        "log_loss": log_loss(y, np.clip(p, 1e-15, 1 - 1e-15), labels=[0, 1]),
    }
    tmp = df[KEYS + ["label"]].copy()
    tmp["p"] = p
    n_obs = tmp.groupby(KEYS).ngroups
    has_pos = tmp.groupby(KEYS)["label"].max()
    top = tmp.sort_values("p", ascending=False).drop_duplicates(KEYS)
    correct = int(top["label"].sum())
    n_hit = int(has_pos.sum())
    out["n_obs"] = n_obs
    out["top1_acc"] = correct / n_obs
    out["top1_acc_given_hit"] = correct / n_hit if n_hit else float("nan")
    out["candidate_miss_rate"] = 1.0 - n_hit / n_obs
    return out

def proba_pos(model, X):
    return model.predict_proba(X)[:, 1]

def load_feature_cols(path):
    path = Path(path)
    if path.suffix in (".yaml", ".yml"):
        import yaml
        d = yaml.safe_load(open(path))
    else:
        d = json.load(open(path))
    return list(d["feature_cols"] if isinstance(d, dict) else d)

def load_splits(path):
    d = json.load(open(path))
    train = [norm_id(t) for t in d["train"]]
    student = [norm_id(t) for t in d["student_test"]]
    overlap = set(train) & set(student)
    assert not overlap, f"Trips appear in both partitions: {sorted(overlap)[:5]}"
    return train, student

def read_candidates(path):
    df = pd.read_parquet(path)
    df["trip_id"] = df["trip_id"].map(norm_id)
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates")
    ap.add_argument("--student-candidates")
    ap.add_argument("--splits")
    ap.add_argument("--feature-cols")
    ap.add_argument("--out", default="experiments/results")
    ap.add_argument("--models", default="default")
    ap.add_argument("--cv-folds", type=int, default=5)
    args = ap.parse_args()

    out = Path(args.out)
    (out / "models").mkdir(parents=True, exist_ok=True)
    (out / "student_predictions").mkdir(parents=True, exist_ok=True)
    (out / "oof_predictions").mkdir(parents=True, exist_ok=True)

    feature_cols = load_feature_cols(args.feature_cols)
    train_ids, student_ids = load_splits(args.splits)

    df = read_candidates(args.candidates)
    needed = feature_cols + ["label", "obs_idx", "arc_id"]
    missing = [c for c in needed if c not in df.columns]
    assert not missing, f"Missing columns in candidate table: {missing}"

    tr = df[df["trip_id"].isin(train_ids)].reset_index(drop=True)
    
    if args.student_candidates:
        sdf = read_candidates(args.student_candidates)
        st = sdf[sdf["trip_id"].isin(student_ids)].reset_index(drop=True)
    else:
        sys.exit("ERROR: Student-test frame is empty. Please provide --student-candidates.")

    if tr.empty:
        sys.exit("Train frame is empty.")
    if st.empty:
        sys.exit("Student-test frame is empty.")

    print(f"train trips found {tr['trip_id'].nunique()}/{len(train_ids)} | rows {len(tr):,}")
    print(f"student trips found {st['trip_id'].nunique()}/{len(student_ids)} | rows {len(st):,}")

    X_tr, y_tr, g_tr = tr[feature_cols].to_numpy(), tr["label"].to_numpy().astype(int), tr["trip_id"].to_numpy()
    X_st = st[feature_cols].to_numpy()

    zoo = model_zoo()
    names = DEFAULT_MODELS if args.models == "default" else [m.strip() for m in args.models.split(",")]
    
    n_splits = min(args.cv_folds, tr["trip_id"].nunique())
    gkf = GroupKFold(n_splits=n_splits)

    rows = []
    for name in names:
        family, factory, config = zoo[name]
        print(f"\n=== {name} ({family}) ===", flush=True)

        fold_f1 = []
        oof_probs = np.zeros(len(X_tr))
        t0 = time.perf_counter()
        for k, (a, b) in enumerate(gkf.split(X_tr, y_tr, g_tr)):
            m = factory()
            m.fit(X_tr[a], y_tr[a])
            p_val = proba_pos(m, X_tr[b])
            oof_probs[b] = p_val
            r = evaluate(tr.iloc[b], p_val)
            fold_f1.append(r["macro_f1"])
            print(f"  fold {k}: macro_f1={r['macro_f1']:.4f}", flush=True)
        cv_time = time.perf_counter() - t0

        oof_df = tr[KEYS + ["arc_id"]].copy()
        oof_df["oof_p"] = oof_probs
        oof_df.to_parquet(out / "oof_predictions" / f"oof_{name}.parquet")

        m = factory()
        t0 = time.perf_counter()
        m.fit(X_tr, y_tr)
        train_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        p_st = proba_pos(m, X_st)
        infer_time = time.perf_counter() - t0

        joblib.dump(m, out / "models" / f"{name}.joblib")
        
        print(f"  CV macro F1 {np.mean(fold_f1):.4f} | train {train_time:.1f}s", flush=True)

if __name__ == "__main__":
    main()
