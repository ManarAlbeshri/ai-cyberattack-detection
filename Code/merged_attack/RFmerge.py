"""
=============================================================
Random Forest Classifier — Merged Dataset — 100 Iterations
Dataset: CICDDoS2019 (Merged Parquet Files)
Metrics:
    • Accuracy
    • Precision
    • Recall
    • F1-Score
    • Confusion Matrix
=============================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# =========================================================
# Settings
# =========================================================
BENIGN_LABEL = "Benign"
N_RUNS = 100
RANDOM_STATE = 42

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "results_RandomForest_Merged"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================================
# Read merged parquet files
# =========================================================
train_path = os.path.join(BASE_DIR, "merged-training.parquet")
test_path  = os.path.join(BASE_DIR, "merged-testing.parquet")

if not os.path.exists(train_path):
    raise FileNotFoundError(
        "merged-training.parquet not found."
    )

if not os.path.exists(test_path):
    raise FileNotFoundError(
        "merged-testing.parquet not found."
    )

print("\nReading merged dataset files...")

merged_train = pd.read_parquet(train_path)
merged_test  = pd.read_parquet(test_path)

if (
    "Label" not in merged_train.columns
    or
    "Label" not in merged_test.columns
):
    raise ValueError(
        "Label column not found in merged files."
    )

print("\nMerged training rows:", len(merged_train))
print("Merged testing rows :", len(merged_test))

# =========================================================
# Prepare Features and Labels
# =========================================================
X_train = merged_train.drop(columns=["Label"])
y_train = (
    merged_train["Label"] != BENIGN_LABEL
).astype(int)

X_test = merged_test.drop(columns=["Label"])
y_test = (
    merged_test["Label"] != BENIGN_LABEL
).astype(int)

# Replace inf/nan
X_train = (
    X_train
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0)
)

X_test = (
    X_test
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0)
)

# Ensure matching columns
X_test = X_test.reindex(
    columns=X_train.columns,
    fill_value=0
)

print("\nNumber of features:", X_train.shape[1])

# =========================================================
# Save Dataset Information
# =========================================================
dataset_info = pd.DataFrame([{
    "Train Rows":
        len(merged_train),

    "Test Rows":
        len(merged_test),

    "Train Attack %":
        round(float(y_train.mean()) * 100, 2),

    "Test Attack %":
        round(float(y_test.mean()) * 100, 2),

    "Number of Features":
        X_train.shape[1]
}])

dataset_info_path = os.path.join(
    OUTPUT_DIR,
    "RF_Merged_dataset_info.csv"
)

dataset_info.to_csv(
    dataset_info_path,
    index=False
)

print(
    f"\nSaved dataset info → "
    f"{dataset_info_path}"
)

# =========================================================
# 100 Iterations
# =========================================================
iter_rows = []

print(
    "\n─── Random Forest "
    "Merged Dataset — 100 Iterations ───"
)

for run_idx in range(1, N_RUNS + 1):

    rs = RANDOM_STATE + run_idx

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=rs,
        n_jobs=-1,
        class_weight="balanced"
    )

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)

    prec = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    rec = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    # Confusion Matrix
    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    iter_rows.append({
        "Iteration":
            run_idx,

        "Accuracy":
            round(acc, 6),

        "Precision":
            round(prec, 6),

        "Recall":
            round(rec, 6),

        "F1":
            round(f1, 6),

        "TN":
            tn,

        "FP":
            fp,

        "FN":
            fn,

        "TP":
            tp
    })

    if run_idx % 10 == 0:
        print(
            f"Completed iteration "
            f"{run_idx}/{N_RUNS}"
        )

# =========================================================
# Save All Iterations
# =========================================================
iter_df = pd.DataFrame(iter_rows)

iter_csv = os.path.join(
    OUTPUT_DIR,
    "RF_Merged_100_iterations_all_metrics.csv"
)

iter_df.to_csv(iter_csv, index=False)

print(
    f"\nSaved iterations CSV → "
    f"{iter_csv}"
)

# =========================================================
# Summary Statistics
# =========================================================
summary_df = (
    iter_df[
        ["Accuracy", "Precision", "Recall", "F1"]
    ]
    .agg(["mean", "std", "min", "max"])
    .T
    .reset_index()
)

summary_df.columns = [
    "Metric",
    "Mean",
    "Std",
    "Min",
    "Max"
]

summary_csv = os.path.join(
    OUTPUT_DIR,
    "RF_Merged_100_iterations_summary.csv"
)

summary_df.to_csv(
    summary_csv,
    index=False
)

print(
    f"Saved summary CSV → "
    f"{summary_csv}"
)

# =========================================================
# Confusion Matrix (Last Run)
# =========================================================
cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1]
)

disp = ConfusionMatrixDisplay(
    cm,
    display_labels=["Benign", "Attack"]
)

fig, ax = plt.subplots(figsize=(5, 4))

disp.plot(
    ax=ax,
    values_format="d",
    colorbar=False
)

ax.set_title(
    "Random Forest — "
    "Merged Dataset Confusion Matrix"
)

plt.tight_layout()

cm_path = os.path.join(
    OUTPUT_DIR,
    "RF_Merged_confusion_matrix.png"
)

plt.savefig(
    cm_path,
    dpi=150,
    bbox_inches="tight"
)

plt.close()

print(
    f"Saved confusion matrix → "
    f"{cm_path}"
)

# =========================================================
# Mean Metrics Bar Chart
# =========================================================
mean_metrics = iter_df[
    ["Accuracy", "Precision", "Recall", "F1"]
].mean()

plt.figure(figsize=(8, 5))

plt.bar(
    mean_metrics.index,
    mean_metrics.values
)

plt.ylim(0, 1.05)

plt.title(
    "Random Forest — "
    "Mean Metrics over 100 Runs"
)

plt.ylabel("Score")
plt.xlabel("Metrics")

for i, value in enumerate(mean_metrics.values):

    plt.text(
        i,
        value + 0.01,
        f"{value:.4f}",
        ha="center"
    )

plt.tight_layout()

bar_path = os.path.join(
    OUTPUT_DIR,
    "RF_Merged_mean_metrics_bar.png"
)

plt.savefig(
    bar_path,
    dpi=150
)

plt.close()

print(
    f"Saved bar chart → "
    f"{bar_path}"
)

# =========================================================
# Final Results
# =========================================================
print(
    "\n═══ Random Forest "
    "Merged Dataset Summary ═══"
)

print(summary_df.to_string(index=False))

print("\n✓ Random Forest Merged — Done!")