import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

# --- Configuration ---
# Update this path if your CSV is somewhere else
CSV_PATH = "experiments/results/comparison_table.csv" 
OUT_DIR = Path("reports/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Set a nice visual style
sns.set_theme(style="whitegrid")

def main():
    if not os.path.exists(CSV_PATH):
        print(f"Error: Could not find {CSV_PATH}. Run your experiments first!")
        return

    df = pd.read_csv(CSV_PATH)
    
    # Sort by CV Macro F1 for better visual flow
    df = df.sort_values("cv_macro_f1_mean", ascending=False)

    # ---------------------------------------------------------
    # Plot 1: Cross-Validation Macro F1 Comparison
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=df, 
        x="model", 
        y="cv_macro_f1_mean", 
        palette="viridis", 
        errorbar=None # We'll add the std dev manually below
    )
    
    # Add error bars from the standard deviation column
    for i, row in enumerate(df.itertuples()):
        plt.errorbar(
            x=i, y=row.cv_macro_f1_mean, 
            yerr=row.cv_macro_f1_std, 
            fmt='none', c='black', capsize=5
        )

    plt.title("Cross-Validation Macro F1 Score by Model", fontsize=14, pad=15)
    plt.ylabel("CV Macro F1 (Mean ± Std)", fontsize=12)
    plt.xlabel("Model", fontsize=12)
    plt.ylim(0.6, 1.0)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "cv_macro_f1.png", dpi=300)
    plt.close()
    print("Saved cv_macro_f1.png")

    # ---------------------------------------------------------
    # Plot 2: Student-Test Top-1 Accuracy vs Macro F1
    # ---------------------------------------------------------
    # Melt the dataframe to make it easy for Seaborn to plot grouped bars
    df_melted = df.melt(
        id_vars="model", 
        value_vars=["student_top1", "student_macro_f1"], 
        var_name="Metric", 
        value_name="Score"
    )
    
    # Rename the metrics for the legend
    df_melted['Metric'] = df_melted['Metric'].map({
        "student_top1": "Top-1 Accuracy",
        "student_macro_f1": "Macro F1"
    })

    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=df_melted, 
        x="model", 
        y="Score", 
        hue="Metric", 
        palette="mako"
    )
    plt.title("Student-Test Performance: Top-1 Accuracy vs Macro F1", fontsize=14, pad=15)
    plt.ylabel("Score", fontsize=12)
    plt.xlabel("Model", fontsize=12)
    plt.ylim(0.6, 1.0)
    plt.legend(title="Metric", loc="upper right")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "student_test_comparison.png", dpi=300)
    plt.close()
    print("Saved student_test_comparison.png")

    # ---------------------------------------------------------
    # Plot 3: Log Loss Comparison
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=df, 
        x="model", 
        y="student_log_loss", 
        palette="rocket"
    )
    plt.title("Student-Test Log Loss by Model (Lower is Better)", fontsize=14, pad=15)
    plt.ylabel("Log Loss", fontsize=12)
    plt.xlabel("Model", fontsize=12)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "log_loss_comparison.png", dpi=300)
    plt.close()
    print("Saved log_loss_comparison.png")

    print(f"\nAll plots successfully saved to {OUT_DIR}/")

if __name__ == "__main__":
    main()
