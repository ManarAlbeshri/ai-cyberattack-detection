"""
=============================================================
Logistic Regression — Merged Dataset — 100 Runs
Dataset: CICDDoS2019 (Merged Parquet Files)
Method: merged-training.parquet + merged-testing.parquet
Metrics: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
=============================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ─────────────────────────────────────────
# Settings
# ─────────────────────────────────────────
BENIGN_LABEL = "Benign"
RANDOM_STATE = 42
N_RUNS = 100
MAX_ITER = 2000

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "results_LogisticRegression_Merged")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────
# Load merged training / testing files
# ─────────────────────────────────────────
train_path = os.path.join(BASE_DIR, "merged-training.parquet")
test_path  = os.path.join(BASE_DIR, "merged-testing.parquet")

if not os.path.exists(train_path):
    raise FileNotFoundError("merged-training.parquet not found.")

if not os.path.exists(test_path):
    raise FileNotFoundError("merged-testing.parquet not found.")

print("\nReading merged dataset files...")

train_df = pd.read_parquet(train_path)
test_df  = pd.read_parquet(test_path)

if "Label" not in train_df.columns or "Label" not in test_df.columns:
    raise ValueError("Label column not found in merged files.")

print("\nMerged training rows:", len(train_df))
print("Merged testing rows :", len(test_df))

# ─────────────────────────────────────────
# Prepare X / y
# ─────────────────────────────────────────
X_train = train_df.drop(columns=["Label"]).select_dtypes(include=[np.number])
X_test  = test_df.drop(columns=["Label"]).select_dtypes(include=[np.number])

y_train = (train_df["Label"] != BENIGN_LABEL).astype(int)
y_test  = (test_df["Label"] != BENIGN_LABEL).astype(int)

# ─────────────────────────────────────────
# Clean data
# ─────────────────────────────────────────
X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
X_test  = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)

X_train = (
    X_train
    .apply(pd.to_numeric, errors="coerce")
    .fillna(0)
    .astype(np.float64)
)

X_test = (
    X_test
    .apply(pd.to_numeric, errors="coerce")
    .fillna(0)
    .astype(np.float64)
)

X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

X_train = X_train.clip(lower=-1e6, upper=1e6)
X_test  = X_test.clip(lower=-1e6, upper=1e6)

print("\nNumber of features:", X_train.shape[1])

# ─────────────────────────────────────────
# Scaling
# ─────────────────────────────────────────
print("\nScaling features...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ─────────────────────────────────────────
# Save dataset info
# ─────────────────────────────────────────
dataset_info = pd.DataFrame([{
    "Train Rows": len(train_df),
    "Test Rows": len(test_df),
    "Train Attack %": round(float(y_train.mean()) * 100, 2),
    "Test Attack %": round(float(y_test.mean()) * 100, 2),
    "Number of Features": X_train.shape[1]
}])

dataset_info_path = os.path.join(
    OUTPUT_DIR,
    "LR_Merged_dataset_info.csv"
)

dataset_info.to_csv(dataset_info_path, index=False)

print(f"\nSaved dataset info → {dataset_info_path}")

# ─────────────────────────────────────────
# PART A — 100 Runs
# ─────────────────────────────────────────
iter_rows = []

print("\n─── Logistic Regression Merged Dataset — 100 Runs ───")

for run_idx in range(1, N_RUNS + 1):

    rs = RANDOM_STATE + run_idx

    model = LogisticRegression(
        max_iter=MAX_ITER,
        solver="saga",
        class_weight="balanced",
        random_state=rs,
        n_jobs=-1
    )

    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    iter_rows.append({
        "Run": run_idx,
        "Accuracy": round(acc, 6),
        "Precision": round(prec, 6),
        "Recall": round(rec, 6),
        "F1": round(f1, 6),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    })

    if run_idx % 10 == 0:
        print(f"Completed run {run_idx}/{N_RUNS}")

iter_df = pd.DataFrame(iter_rows)

iter_csv = os.path.join(
    OUTPUT_DIR,
    "LR_Merged_100_runs_all_metrics.csv"
)

iter_df.to_csv(iter_csv, index=False)

print(f"\nSaved runs CSV → {iter_csv}")

# ─────────────────────────────────────────
# Summary
# ─────────────────────────────────────────
summary_df = (
    iter_df[["Accuracy", "Precision", "Recall", "F1"]]
    .agg(["mean", "std", "min", "max"])
    .T
    .reset_index()
)

summary_df.columns = ["Metric", "Mean", "Std", "Min", "Max"]

summary_csv = os.path.join(
    OUTPUT_DIR,
    "LR_Merged_100_runs_summary.csv"
)

summary_df.to_csv(summary_csv, index=False)

print(f"Saved summary CSV → {summary_csv}")

# ─────────────────────────────────────────
# Confusion Matrix from last run
# ─────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

disp = ConfusionMatrixDisplay(
    cm,
    display_labels=["Benign", "Attack"]
)

fig, ax = plt.subplots(figsize=(5, 4))
disp.plot(ax=ax, values_format="d", colorbar=False)
ax.set_title("Logistic Regression — Merged Dataset Confusion Matrix", fontsize=11)

plt.tight_layout()

cm_path = os.path.join(
    OUTPUT_DIR,
    "LR_Merged_confusion_matrix.png"
)

plt.savefig(cm_path, dpi=150, bbox_inches="tight")
plt.close()

print(f"Saved confusion matrix → {cm_path}")

# ─────────────────────────────────────────
# Bar chart — mean metrics
# ─────────────────────────────────────────
mean_metrics = iter_df[["Accuracy", "Precision", "Recall", "F1"]].mean()

plt.figure(figsize=(8, 5))
plt.bar(mean_metrics.index, mean_metrics.values)
plt.ylim(0, 1.05)

plt.title("Logistic Regression — Mean Metrics over 100 Runs")
plt.ylabel("Score")
plt.xlabel("Metrics")

for i, value in enumerate(mean_metrics.values):
    plt.text(i, value + 0.01, f"{value:.4f}", ha="center")

plt.tight_layout()

bar_path = os.path.join(
    OUTPUT_DIR,
    "LR_Merged_mean_metrics_bar.png"
)

plt.savefig(bar_path, dpi=150)
plt.close()

print(f"\nBar chart saved → {bar_path}")

# ─────────────────────────────────────────
# Final Print
# ─────────────────────────────────────────
print("\n═══ Logistic Regression Merged Dataset Summary ═══")
print(summary_df.to_string(index=False))

print("\n✓ Logistic Regression Merged — Done!")