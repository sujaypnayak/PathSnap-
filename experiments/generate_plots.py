"""Regenerate recorded PathSnap model-comparison figures.

Run from the repository root:
    python -m experiments.generate_plots
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"
OUTPUT = ROOT / "reports" / "figures"


def plot_bar(df, column, title, ylabel, filename, error_column=None):
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(10, 5))
    errors = df[error_column].to_numpy() if error_column else None
    ax.bar(x, df[column].to_numpy(), yerr=errors, capsize=4)
    ax.set_xticks(x, df["model"].astype(str).tolist(), rotation=20, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUTPUT / filename, dpi=300)
    plt.close(fig)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    comparison = pd.read_csv(RESULTS / "comparison_table.csv")
    comparison = comparison.sort_values("cv_macro_f1_mean", ascending=False)

    plot_bar(
        comparison, "cv_macro_f1_mean",
        "Grouped Cross-Validation Macro F1 by Model",
        "CV Macro F1 (mean ± std)", "cv_macro_f1.png",
        error_column="cv_macro_f1_std",
    )

    # Student-test metrics belong to experiment_log.csv, not comparison_table.csv.
    log = pd.read_csv(RESULTS / "experiment_log.csv")
    log = log.dropna(subset=["student_top1", "student_macro_f1", "student_log_loss"])
    log = log.sort_values("timestamp").drop_duplicates("model", keep="last")
    log = log.set_index("model").reindex(comparison["model"].tolist())
    log = log.dropna(subset=["student_top1", "student_macro_f1", "student_log_loss"]).reset_index()
    if log.empty:
        raise ValueError("No recorded student-test metrics in experiment_log.csv")

    x = np.arange(len(log))
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.38
    ax.bar(x - width / 2, log["student_top1"], width, label="Top-1")
    ax.bar(x + width / 2, log["student_macro_f1"], width, label="Macro F1")
    ax.set_xticks(x, log["model"], rotation=20, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Student-Test Metrics (Earlier Experiment Protocol)")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUTPUT / "student_test_comparison.png", dpi=300)
    plt.close(fig)

    plot_bar(
        log, "student_log_loss",
        "Student-Test Log Loss (Earlier Experiment Protocol)",
        "Log loss (lower is better)", "log_loss_comparison.png",
    )
    print(f"Saved three figures in {OUTPUT}")


if __name__ == "__main__":
    main()
