"""
=============================================================
Logistic Regression — 100 Runs per Dataset
Dataset: CICDDoS2019 (Parquet files)
Metrics: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
=============================================================
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
)

# ─────────────────────────────────────────
# Settings
# ─────────────────────────────────────────
BENIGN_LABEL = "Benign"
RANDOM_STATE = 42
N_RUNS = 100

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "results_LogisticRegression")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────
# Load all training / testing pairs
# ─────────────────────────────────────────
train_files = sorted(glob.glob(os.path.join(BASE_DIR, "*-training.parquet")))
pairs = []

for train_path in train_files:
    prefix = os.path.basename(train_path).replace("-training.parquet", "")
    test_path = os.path.join(BASE_DIR, f"{prefix}-testing.parquet")

    if os.path.exists(test_path):
        pairs.append((prefix, train_path, test_path))

if not pairs:
    raise FileNotFoundError("No training/testing parquet pairs found.")

print(f"\nFound {len(pairs)} dataset pairs:")
for p, _, _ in pairs:
    print(f"  • {p}")

# ─────────────────────────────────────────
# Cache datasets
# ─────────────────────────────────────────
datasets = {}

for attack_name, train_path, test_path in pairs:
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    if "Label" not in train_df.columns or "Label" not in test_df.columns:
        print(f"[SKIP] {attack_name}: no Label column")
        continue

    X_train = train_df.drop(columns=["Label"]).select_dtypes(include=[np.number])
    X_test = test_df.drop(columns=["Label"]).select_dtypes(include=[np.number])

    y_train = (train_df["Label"] != BENIGN_LABEL).astype(int)
    y_test = (test_df["Label"] != BENIGN_LABEL).astype(int)

    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    datasets[attack_name] = {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "train_attack_pct": float(y_train.mean()) * 100,
        "test_attack_pct": float(y_test.mean()) * 100,
    }

print(f"\nCached {len(datasets)} datasets.")

# ─────────────────────────────────────────
# PART A — 100 Runs
# ─────────────────────────────────────────
iter_rows = []

for run_idx in range(1, N_RUNS + 1):
    rs = RANDOM_STATE + run_idx

    for attack_name, data in datasets.items():
        model = LogisticRegression(
            max_iter=2000,
            solver="saga",
            class_weight="balanced",
            random_state=rs,
            n_jobs=-1
        )

        model.fit(data["X_train"], data["y_train"])
        y_pred = model.predict(data["X_test"])

        acc = accuracy_score(data["y_test"], y_pred)
        prec = precision_score(data["y_test"], y_pred, zero_division=0)
        rec = recall_score(data["y_test"], y_pred, zero_division=0)
        f1 = f1_score(data["y_test"], y_pred, zero_division=0)

        cm = confusion_matrix(data["y_test"], y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        iter_rows.append({
            "Run": run_idx,
            "Attack/File": attack_name,
            "Accuracy": round(acc, 6),
            "Precision": round(prec, 6),
            "Recall": round(rec, 6),
            "F1": round(f1, 6),
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp,
        })

    if run_idx % 10 == 0:
        print(f"  Completed run {run_idx}/{N_RUNS}")

iter_df = pd.DataFrame(iter_rows)

iter_csv = os.path.join(OUTPUT_DIR, "LR_100_runs_all_metrics.csv")
iter_df.to_csv(iter_csv, index=False)
print(f"\nSaved runs CSV → {iter_csv}")

summary_df = (
    iter_df.groupby("Attack/File")[["Accuracy", "Precision", "Recall", "F1"]]
    .agg(["mean", "std", "min", "max"])
    .reset_index()
    .sort_values(("F1", "mean"), ascending=False)
)

summary_csv = os.path.join(OUTPUT_DIR, "LR_100_runs_summary.csv")
summary_df.to_csv(summary_csv, index=False)
print(f"Saved summary CSV → {summary_csv}")

# ─────────────────────────────────────────
# PART B — Single Run Confusion Matrices
# ─────────────────────────────────────────
print("\n─── Generating Confusion Matrices (single run) ───")
results = []

for attack_name, data in datasets.items():
    model = LogisticRegression(
        max_iter=2000,
        solver="saga",
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    model.fit(data["X_train"], data["y_train"])
    y_pred = model.predict(data["X_test"])

    acc = accuracy_score(data["y_test"], y_pred)
    prec = precision_score(data["y_test"], y_pred, zero_division=0)
    rec = recall_score(data["y_test"], y_pred, zero_division=0)
    f1 = f1_score(data["y_test"], y_pred, zero_division=0)

    cm = confusion_matrix(data["y_test"], y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    disp = ConfusionMatrixDisplay(cm, display_labels=["Benign", "Attack"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, values_format="d", colorbar=False)
    ax.set_title(f"Logistic Regression — {attack_name}", fontsize=11)

    plt.tight_layout()
    cm_path = os.path.join(OUTPUT_DIR, f"LR_CM_{attack_name}.png")
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  [Saved] {cm_path}")

    results.append({
        "Attack/File": attack_name,
        "Train Rows": data["train_rows"],
        "Test Rows": data["test_rows"],
        "Train Attack %": round(data["train_attack_pct"], 2),
        "Test Attack %": round(data["test_attack_pct"], 2),
        "Accuracy": round(acc, 6),
        "Precision": round(prec, 6),
        "Recall": round(rec, 6),
        "F1": round(f1, 6),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    })

results_df = pd.DataFrame(results).sort_values("F1", ascending=False)

final_csv = os.path.join(OUTPUT_DIR, "LR_final_comparison.csv")
results_df.to_csv(final_csv, index=False)

print("\n═══ Logistic Regression — Final Comparison (single run) ═══")
print(results_df.to_string(index=False))
print(f"\nSaved final comparison → {final_csv}")

# ─────────────────────────────────────────
# Bar chart — mean metrics
# ─────────────────────────────────────────
mean_df = iter_df.groupby("Attack/File")[["Accuracy", "Precision", "Recall", "F1"]].mean()

mean_df.plot(kind="bar", figsize=(14, 5), ylim=(0, 1.05))
plt.title("Logistic Regression — Mean Metrics over 100 Runs per Dataset")
plt.ylabel("Score")
plt.xlabel("Dataset")
plt.xticks(rotation=30, ha="right")
plt.legend(loc="lower right")
plt.tight_layout()

bar_path = os.path.join(OUTPUT_DIR, "LR_mean_metrics_bar.png")
plt.savefig(bar_path, dpi=150)
plt.close()

print(f"\nBar chart saved → {bar_path}")
print("\n✓ Logistic Regression — Done!")