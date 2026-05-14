"""
=============================================================
Random Forest Classifier — 100 Iterations with Hyperparameter Tuning
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

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
)

BENIGN_LABEL  = "Benign"
N_ESTIMATORS  = 100
RANDOM_STATE  = 42
N_RUNS        = 100

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR    = os.path.join(BASE_DIR, "results_RandomForest_Tuned")
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

datasets = {}

for attack_name, train_path, test_path in pairs:
    train_df = pd.read_parquet(train_path)
    test_df  = pd.read_parquet(test_path)

    if "Label" not in train_df.columns or "Label" not in test_df.columns:
        print(f"[SKIP] {attack_name}: no Label column")
        continue

    X_train = train_df.drop(columns=["Label"])
    y_train = (train_df["Label"] != BENIGN_LABEL).astype(int)

    X_test = test_df.drop(columns=["Label"])
    y_test = (test_df["Label"] != BENIGN_LABEL).astype(int)

    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_test  = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)

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
# Hyperparameter Tuning
# ─────────────────────────────────────────
param_grid = {
    "n_estimators": [100],
    "max_depth": [10, 20, 30, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"],
    "class_weight": ["balanced", "balanced_subsample"],
    "criterion": ["gini", "entropy"]
}

best_params_rows = []
best_models = {}

print("\n─── Hyperparameter Tuning per Dataset ───")

for attack_name, data in datasets.items():
    print(f"\nTuning Random Forest for: {attack_name}")

    base_model = RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=-1,
        bootstrap=True
    )

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_grid,
        n_iter=20,
        scoring="recall",
        cv=3,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    search.fit(data["X_train"], data["y_train"])

    best_models[attack_name] = search.best_estimator_

    best_params_rows.append({
        "Attack/File": attack_name,
        "Best Recall Score CV": round(search.best_score_, 6),
        "Best Params": search.best_params_
    })

    print(f"Best params for {attack_name}:")
    print(search.best_params_)

best_params_df = pd.DataFrame(best_params_rows)
best_params_csv = os.path.join(OUTPUT_DIR, "RF_best_hyperparameters.csv")
best_params_df.to_csv(best_params_csv, index=False)

print(f"\nSaved best hyperparameters → {best_params_csv}")

# ─────────────────────────────────────────
# PART A — 100 Iterations using tuned parameters
# ─────────────────────────────────────────
iter_rows = []

for run_idx in range(1, N_RUNS + 1):
    rs = RANDOM_STATE + run_idx

    for attack_name, data in datasets.items():
        best_params = best_models[attack_name].get_params()

        model = RandomForestClassifier(
            n_estimators=best_params["n_estimators"],
            max_depth=best_params["max_depth"],
            min_samples_split=best_params["min_samples_split"],
            min_samples_leaf=best_params["min_samples_leaf"],
            max_features=best_params["max_features"],
            class_weight=best_params["class_weight"],
            criterion=best_params["criterion"],
            bootstrap=True,
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
            "Iteration": run_idx,
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
        print(f"  Completed iteration {run_idx}/{N_RUNS}")

iter_df = pd.DataFrame(iter_rows)

iter_csv = os.path.join(OUTPUT_DIR, "RF_Tuned_100_iterations_all_metrics.csv")
iter_df.to_csv(iter_csv, index=False)
print(f"\nSaved iterations CSV → {iter_csv}")

summary_df = (
    iter_df.groupby("Attack/File")[["Accuracy", "Precision", "Recall", "F1"]]
    .agg(["mean", "std", "min", "max"])
    .reset_index()
    .sort_values(("F1", "mean"), ascending=False)
)

summary_csv = os.path.join(OUTPUT_DIR, "RF_Tuned_100_iterations_summary.csv")
summary_df.to_csv(summary_csv, index=False)
print(f"Saved summary CSV → {summary_csv}")

# ─────────────────────────────────────────
# PART B — Single Run Confusion Matrices
# ─────────────────────────────────────────
print("\n─── Generating Confusion Matrices (single run) ───")
results = []

for attack_name, data in datasets.items():
    model = best_models[attack_name]

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
    ax.set_title(f"Tuned Random Forest — {attack_name}", fontsize=11)

    plt.tight_layout()
    cm_path = os.path.join(OUTPUT_DIR, f"RF_Tuned_CM_{attack_name}.png")
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

final_csv = os.path.join(OUTPUT_DIR, "RF_Tuned_final_comparison.csv")
results_df.to_csv(final_csv, index=False)

print("\n═══ Tuned Random Forest — Final Comparison ═══")
print(results_df.to_string(index=False))
print(f"\nSaved final comparison → {final_csv}")

# ─────────────────────────────────────────
# Bar Chart
# ─────────────────────────────────────────
mean_df = iter_df.groupby("Attack/File")[["Accuracy", "Precision", "Recall", "F1"]].mean()

mean_df.plot(kind="bar", figsize=(14, 5), ylim=(0, 1.05))
plt.title("Tuned Random Forest — Mean Metrics over 100 Runs per Dataset")
plt.ylabel("Score")
plt.xlabel("Dataset")
plt.xticks(rotation=30, ha="right")
plt.legend(loc="lower right")
plt.tight_layout()

bar_path = os.path.join(OUTPUT_DIR, "RF_Tuned_mean_metrics_bar.png")
plt.savefig(bar_path, dpi=150)
plt.close()

print(f"\nBar chart saved → {bar_path}")
print("\n✓ Tuned Random Forest — Done!")