<<<<<<< HEAD
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
    # Safety check: Only evaluate rows where labels exist
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

def make_smoke_data(tmp):
    rng = np.random.default_rng(SEED)
    rows = []
    for t in range(30):
        for o in range(40):
            true_k = rng.integers(0, 6)
            for k in range(6):
                lab = int(k == true_k)
                rows.append((f"{t:08d}", o, k + 1000 * t, rng.normal(10 - 8 * lab, 3), rng.normal(40 - 35 * lab, 20), rng.normal(8, 3), lab))
    df = pd.DataFrame(rows, columns=["trip_id", "obs_idx", "arc_id", "dist_m", "bearing_diff", "speed", "label"])
    cand = Path(tmp) / "cand.parquet"
    df.to_parquet(cand)
    splits = Path(tmp) / "splits.json"
    json.dump({"train": [f"{i:08d}" for i in range(24)], "student_test": [f"{i:08d}" for i in range(24, 30)]}, open(splits, "w"))
    feats = Path(tmp) / "features.json"
    json.dump(["dist_m", "bearing_diff", "speed"], open(feats, "w"))
    return str(cand), str(splits), str(feats)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", help="parquet with the training trips' candidates")
    ap.add_argument("--student-candidates", help="parquet with student-test trips")
    ap.add_argument("--splits")
    ap.add_argument("--feature-cols")
    ap.add_argument("--out", default="experiments/results")
    ap.add_argument("--models", default="default", help="'default', 'all', or comma list")
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--smoke", action="store_true", help="run on synthetic data to test the harness")
    args = ap.parse_args()

    if args.smoke:
        tmp = tempfile.mkdtemp()
        args.candidates, args.splits, args.feature_cols = make_smoke_data(tmp)
        args.student_candidates = None
        args.out = "experiments/smoke_results"
        print("=" * 60 + "\nSMOKE TEST on synthetic data - NOT real results\n" + "=" * 60)
        
    for req in ("candidates", "splits", "feature_cols"):
        if getattr(args, req) is None:
            ap.error(f"--{req.replace('_', '-')} is required (or use --smoke)")

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
    
    # NO FALLBACK LOGIC - EXITS CLEANLY IF MISSING
    if args.student_candidates:
        sdf = read_candidates(args.student_candidates)
        st = sdf[sdf["trip_id"].isin(student_ids)].reset_index(drop=True)
    else:
        st = df[df["trip_id"].isin(student_ids)].reset_index(drop=True)
        
    if tr.empty:
        sys.exit("Train frame is empty - check trip ID format in splits file vs parquet.")
    if st.empty:
        sys.exit("ERROR: Student-test frame is empty. Please provide --student-candidates.")

    print(f"train trips found {tr['trip_id'].nunique()}/{len(train_ids)} | rows {len(tr):,} | "
          f"positives {tr['label'].mean():.3%}")
    print(f"student trips found {st['trip_id'].nunique()}/{len(student_ids)} | rows {len(st):,}")

    X_tr, y_tr, g_tr = tr[feature_cols].to_numpy(), tr["label"].to_numpy().astype(int), tr["trip_id"].to_numpy()
    X_st = st[feature_cols].to_numpy()

    zoo = model_zoo()
    if args.models == "default":
        names = DEFAULT_MODELS
    elif args.models == "all":
        names = list(zoo)
    else:
        names = [m.strip() for m in args.models.split(",")]
    unknown = [n for n in names if n not in zoo]
    assert not unknown, f"Unknown model names: {unknown}"

    n_splits = min(args.cv_folds, tr["trip_id"].nunique())
    gkf = GroupKFold(n_splits=n_splits)

    meta = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "seed": SEED, "threshold": THRESHOLD, "cv_folds": n_splits,
        "feature_cols": feature_cols, "n_train_trips": int(tr["trip_id"].nunique()),
        "n_student_trips": int(st["trip_id"].nunique()),
        "python": platform.python_version(), "sklearn": sklearn.__version__,
        "candidates_file": str(args.candidates), "student_candidates_file": str(args.student_candidates),
        "models_run": names, "smoke": bool(args.smoke),
    }
    json.dump(meta, open(out / "run_meta.json", "w"), indent=2)

    rows = []
    for name in names:
        family, factory, config = zoo[name]
        print(f"\n=== {name} ({family}) ===", flush=True)

        fold_f1, fold_top1, fold_ll = [], [], []
        oof_probs = np.zeros(len(X_tr))  # NEW: OOF Array
        t0 = time.perf_counter()
        for k, (a, b) in enumerate(gkf.split(X_tr, y_tr, g_tr)):
            m = factory()
            m.fit(X_tr[a], y_tr[a])
            p_val = proba_pos(m, X_tr[b])
            oof_probs[b] = p_val  # NEW: Save OOF probs
            
            r = evaluate(tr.iloc[b], p_val)
            fold_f1.append(r["macro_f1"]); fold_top1.append(r["top1_acc"]); fold_ll.append(r["log_loss"])
            print(f"  fold {k}: macro_f1={r['macro_f1']:.4f} top1={r['top1_acc']:.4f} logloss={r['log_loss']:.4f}", flush=True)
        cv_time = time.perf_counter() - t0

        # NEW: Save OOF predictions for Person 3
        oof_df = tr[KEYS + ["arc_id"]].copy()
        oof_df["oof_p"] = oof_probs
        oof_df.to_parquet(out / "oof_predictions" / f"oof_{name}.parquet")

        # Final Fit
        m = factory()
        t0 = time.perf_counter()
        m.fit(X_tr, y_tr)
        train_time = time.perf_counter() - t0
        train_res = evaluate(tr, proba_pos(m, X_tr))

        # Student Test Inference
        t0 = time.perf_counter()
        p_st = proba_pos(m, X_st)
        infer_time = time.perf_counter() - t0
        
        # Save student predictions
        pred = st[KEYS + ["arc_id", "label"]].copy()
        pred["p"] = p_st
        pred.to_parquet(out / "student_predictions" / f"{name}.parquet")

        # ONLY EVALUATE IF LABELS EXIST (Safety Check)
        if not st["label"].isna().all():
            st_res = evaluate(st, p_st)
        else:
            st_res = {"top1_acc": None, "macro_f1": None, "log_loss": None, "candidate_miss_rate": None,
                      "macro_precision": None, "macro_recall": None, "accuracy_cand": None, "top1_acc_given_hit": None}

        row = {
            "model": name, "family": family, "config": json.dumps(config), "seed": SEED,
            "cv_macro_f1_mean": np.mean(fold_f1), "cv_macro_f1_std": np.std(fold_f1),
            "cv_top1_mean": np.mean(fold_top1), "cv_log_loss_mean": np.mean(fold_ll),
            "train_macro_f1": train_res["macro_f1"], "train_top1": train_res["top1_acc"],
            "train_log_loss": train_res["log_loss"],
            "student_top1": st_res["top1_acc"], "student_top1_given_hit": st_res["top1_acc_given_hit"],
            "student_macro_f1": st_res["macro_f1"], "student_macro_precision": st_res["macro_precision"],
            "student_macro_recall": st_res["macro_recall"], "student_accuracy_cand": st_res["accuracy_cand"],
            "student_log_loss": st_res["log_loss"], "student_candidate_miss_rate": st_res["candidate_miss_rate"],
            "cv_time_s": cv_time, "train_time_s": train_time, "infer_time_s": infer_time,
            "infer_ms_per_1k_candidates": infer_time / len(X_st) * 1e6,
            "timestamp": datetime.now().isoformat(timespec="seconds"), "smoke": bool(args.smoke),
        }
        rows.append(row)
        print(f"  CV macro F1 {row['cv_macro_f1_mean']:.4f} +/- {row['cv_macro_f1_std']:.4f}", flush=True)

        # Strict Artifact Path
        joblib.dump(m, out / "models" / f"{name}.joblib")

        log_path = out / "experiment_log.csv"
        pd.DataFrame([row]).to_csv(log_path, mode="a", header=not log_path.exists(), index=False)

    res = pd.DataFrame(rows).sort_values("cv_macro_f1_mean", ascending=False)
    res.to_csv(out / "comparison_table.csv", index=False)
    
    # Handle printing table if student metrics are missing
    cols_to_show = ["model", "family", "cv_macro_f1_mean", "cv_top1_mean", "train_macro_f1", "train_time_s"]
    if not pd.isna(res["student_top1"]).all():
        cols_to_show.extend(["student_top1", "student_macro_f1", "student_log_loss"])
        
    short = res[cols_to_show]
    print("\n" + short.round(4).to_string(index=False))
    try:
        (out / "comparison_table.md").write_text(short.round(4).to_markdown(index=False))
    except ImportError:
        pass
    best = res.iloc[0]
    print(f"\nBest by CV macro F1: {best['model']} ({best['cv_macro_f1_mean']:.4f}).")

if __name__ == "__main__":
    main()
=======
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
        # Slow models - only run if explicitly named
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

# We will use these 5 by default (fast and diverse families)
DEFAULT_MODELS = ["logreg", "gaussian_nb", "hist_gb", "random_forest", "xgboost"]

def evaluate(df, p):
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

def make_smoke_data(tmp):
    rng = np.random.default_rng(SEED)
    rows = []
    for t in range(30):
        for o in range(40):
            true_k = rng.integers(0, 6)
            for k in range(6):
                lab = int(k == true_k)
                rows.append((f"{t:08d}", o, k + 1000 * t, rng.normal(10 - 8 * lab, 3), rng.normal(40 - 35 * lab, 20), rng.normal(8, 3), lab))
    df = pd.DataFrame(rows, columns=["trip_id", "obs_idx", "arc_id", "dist_m", "bearing_diff", "speed", "label"])
    cand = Path(tmp) / "cand.parquet"
    df.to_parquet(cand)
    splits = Path(tmp) / "splits.json"
    json.dump({"train": [f"{i:08d}" for i in range(24)], "student_test": [f"{i:08d}" for i in range(24, 30)]}, open(splits, "w"))
    feats = Path(tmp) / "features.json"
    json.dump(["dist_m", "bearing_diff", "speed"], open(feats, "w"))
    return str(cand), str(splits), str(feats)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", help="parquet with the training trips' candidates")
    ap.add_argument("--student-candidates", help="parquet with student-test trips")
    ap.add_argument("--splits")
    ap.add_argument("--feature-cols")
    ap.add_argument("--out", default="experiments/results")
    ap.add_argument("--models", default="default", help="'default', 'all', or comma list")
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--smoke", action="store_true", help="run on synthetic data to test the harness")
    args = ap.parse_args()

    if args.smoke:
        tmp = tempfile.mkdtemp()
        args.candidates, args.splits, args.feature_cols = make_smoke_data(tmp)
        args.student_candidates = None
        args.out = "experiments/smoke_results"
        print("=" * 60 + "\nSMOKE TEST on synthetic data - NOT real results\n" + "=" * 60)
        
    for req in ("candidates", "splits", "feature_cols"):
        if getattr(args, req) is None:
            ap.error(f"--{req.replace('_', '-')} is required (or use --smoke)")

    out = Path(args.out)
    (out / "models").mkdir(parents=True, exist_ok=True)
    (out / "student_predictions").mkdir(parents=True, exist_ok=True)

    feature_cols = load_feature_cols(args.feature_cols)
    train_ids, student_ids = load_splits(args.splits)

    df = read_candidates(args.candidates)
    needed = feature_cols + ["label", "obs_idx", "arc_id"]
    missing = [c for c in needed if c not in df.columns]
    assert not missing, f"Missing columns in candidate table: {missing}"

    tr = df[df["trip_id"].isin(train_ids)].reset_index(drop=True)
    
    # --- TEMPORARY FALLBACK LOGIC ---
    if args.student_candidates:
        sdf = read_candidates(args.student_candidates)
        st = sdf[sdf["trip_id"].isin(student_ids)].reset_index(drop=True)
    else:
        st = df[df["trip_id"].isin(student_ids)].reset_index(drop=True)
        
        # If student data is completely missing, borrow a training trip temporarily
        if st.empty and not tr.empty:
            print("\n" + "!"*60)
            print("WARNING: Student-test trips not found in the candidate file.")
            print("TEMPORARY WORKAROUND: Borrowing one training trip for testing.")
            print("Remember to re-run this with the real student data for final submission!")
            print("!"*60 + "\n")
            temp_test_trip = tr["trip_id"].iloc[-1]
            st = tr[tr["trip_id"] == temp_test_trip].copy()
            tr = tr[tr["trip_id"] != temp_test_trip].copy()
    # -------------------------------

    print(f"train trips found {tr['trip_id'].nunique()}/{len(train_ids)} | rows {len(tr):,} | "
          f"positives {tr['label'].mean():.3%}")
    print(f"student trips found {st['trip_id'].nunique()}/{len(student_ids)} | rows {len(st):,}")
    
    if tr.empty:
        sys.exit("Train frame is empty - check trip ID format in splits file vs parquet.")
    if st.empty:
        sys.exit("Student-test frame is empty and fallback failed.")

    X_tr, y_tr, g_tr = tr[feature_cols].to_numpy(), tr["label"].to_numpy().astype(int), tr["trip_id"].to_numpy()
    X_st = st[feature_cols].to_numpy()

    zoo = model_zoo()
    if args.models == "default":
        names = DEFAULT_MODELS
    elif args.models == "all":
        names = list(zoo)
    else:
        names = [m.strip() for m in args.models.split(",")]
    unknown = [n for n in names if n not in zoo]
    assert not unknown, f"Unknown model names: {unknown}"

    n_splits = min(args.cv_folds, tr["trip_id"].nunique())
    gkf = GroupKFold(n_splits=n_splits)

    meta = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "seed": SEED, "threshold": THRESHOLD, "cv_folds": n_splits,
        "feature_cols": feature_cols, "n_train_trips": int(tr["trip_id"].nunique()),
        "n_student_trips": int(st["trip_id"].nunique()),
        "python": platform.python_version(), "sklearn": sklearn.__version__,
        "candidates_file": str(args.candidates), "student_candidates_file": str(args.student_candidates),
        "models_run": names, "smoke": bool(args.smoke),
    }
    json.dump(meta, open(out / "run_meta.json", "w"), indent=2)

    rows = []
    for name in names:
        family, factory, config = zoo[name]
        print(f"\n=== {name} ({family}) ===", flush=True)

        fold_f1, fold_top1, fold_ll = [], [], []
        t0 = time.perf_counter()
        for k, (a, b) in enumerate(gkf.split(X_tr, y_tr, g_tr)):
            m = factory()
            m.fit(X_tr[a], y_tr[a])
            r = evaluate(tr.iloc[b], proba_pos(m, X_tr[b]))
            fold_f1.append(r["macro_f1"]); fold_top1.append(r["top1_acc"]); fold_ll.append(r["log_loss"])
            print(f"  fold {k}: macro_f1={r['macro_f1']:.4f} top1={r['top1_acc']:.4f} logloss={r['log_loss']:.4f}", flush=True)
        cv_time = time.perf_counter() - t0

        m = factory()
        t0 = time.perf_counter()
        m.fit(X_tr, y_tr)
        train_time = time.perf_counter() - t0
        train_res = evaluate(tr, proba_pos(m, X_tr))

        t0 = time.perf_counter()
        p_st = proba_pos(m, X_st)
        infer_time = time.perf_counter() - t0
        st_res = evaluate(st, p_st)

        row = {
            "model": name, "family": family, "config": json.dumps(config), "seed": SEED,
            "cv_macro_f1_mean": np.mean(fold_f1), "cv_macro_f1_std": np.std(fold_f1),
            "cv_top1_mean": np.mean(fold_top1), "cv_log_loss_mean": np.mean(fold_ll),
            "train_macro_f1": train_res["macro_f1"], "train_top1": train_res["top1_acc"],
            "train_log_loss": train_res["log_loss"],
            "student_top1": st_res["top1_acc"], "student_top1_given_hit": st_res["top1_acc_given_hit"],
            "student_macro_f1": st_res["macro_f1"], "student_macro_precision": st_res["macro_precision"],
            "student_macro_recall": st_res["macro_recall"], "student_accuracy_cand": st_res["accuracy_cand"],
            "student_log_loss": st_res["log_loss"], "student_candidate_miss_rate": st_res["candidate_miss_rate"],
            "cv_time_s": cv_time, "train_time_s": train_time, "infer_time_s": infer_time,
            "infer_ms_per_1k_candidates": infer_time / len(X_st) * 1e6,
            "timestamp": datetime.now().isoformat(timespec="seconds"), "smoke": bool(args.smoke),
        }
        rows.append(row)
        print(f"  CV macro F1 {row['cv_macro_f1_mean']:.4f} +/- {row['cv_macro_f1_std']:.4f} | "
              f"student top1 {row['student_top1']:.4f} macro F1 {row['student_macro_f1']:.4f} "
              f"logloss {row['student_log_loss']:.4f} | train {train_time:.1f}s infer {infer_time:.2f}s", flush=True)

        joblib.dump(m, out / "models" / f"{name}.joblib")
        pred = st[KEYS + ["arc_id", "label"]].copy()
        pred["p"] = p_st
        pred.to_parquet(out / "student_predictions" / f"{name}.parquet")

        log_path = out / "experiment_log.csv"
        pd.DataFrame([row]).to_csv(log_path, mode="a", header=not log_path.exists(), index=False)

    res = pd.DataFrame(rows).sort_values("cv_macro_f1_mean", ascending=False)
    res.to_csv(out / "comparison_table.csv", index=False)
    short = res[["model", "family", "cv_macro_f1_mean", "cv_macro_f1_std", "cv_top1_mean", "student_top1",
                 "student_macro_f1", "student_log_loss", "student_candidate_miss_rate", "train_time_s",
                 "infer_ms_per_1k_candidates"]]
    print("\n" + short.round(4).to_string(index=False))
    try:
        (out / "comparison_table.md").write_text(short.round(4).to_markdown(index=False))
    except ImportError:
        pass
    best = res.iloc[0]
    print(f"\nBest by CV macro F1: {best['model']} ({best['cv_macro_f1_mean']:.4f}). "
          f"Judge the final pick on CV + student-test + log loss + inference time, not on this alone.")

if __name__ == "__main__":
    main()
>>>>>>> origin/main
