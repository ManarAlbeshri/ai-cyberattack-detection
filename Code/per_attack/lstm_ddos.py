"""
=============================================================
LSTM — 100 Runs per Dataset
Dataset: CICDDoS2019 (Parquet files)
Method: Per-Attack Evaluation
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
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization

# ─────────────────────────────────────────
# Settings
# ─────────────────────────────────────────
BENIGN_LABEL = "Benign"
RANDOM_STATE = 42
N_RUNS = 100
EPOCHS_PER_RUN = 15
BATCH_SIZE = 512
TIME_STEPS = 1

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "results_LSTM")
os.makedirs(OUTPUT_DIR, exist_ok=True)

tf.random.set_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# ─────────────────────────────────────────
# Load pairs — exclude merged files
# ─────────────────────────────────────────
train_files = sorted(glob.glob(os.path.join(BASE_DIR, "*-training.parquet")))

train_files = [
    f for f in train_files
    if not os.path.basename(f).startswith("merged-")
]

pairs = []

for train_path in train_files:
    prefix = os.path.basename(train_path).replace("-training.parquet", "")

    if prefix.lower() == "merged":
        continue

    test_path = os.path.join(BASE_DIR, f"{prefix}-testing.parquet")

    if os.path.exists(test_path):
        pairs.append((prefix, train_path, test_path))

if not pairs:
    raise FileNotFoundError("No training/testing parquet pairs found.")

print(f"\nFound {len(pairs)} dataset pairs:")
for p, _, _ in pairs:
    print(f"  • {p}")

# ─────────────────────────────────────────
# Model
# ─────────────────────────────────────────
def build_model(n_features):
    model = Sequential([
        LSTM(64, input_shape=(TIME_STEPS, n_features)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model

# ─────────────────────────────────────────
# Main Loop
# ─────────────────────────────────────────
iter_rows = []
final_rows = []

for attack_name, train_path, test_path in pairs:

    print(f"\n===== {attack_name} =====")

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    if "Label" not in train_df.columns or "Label" not in test_df.columns:
        print(f"[SKIP] {attack_name}: no Label column")
        continue

    X_train = train_df.drop(columns=["Label"])
    X_test = test_df.drop(columns=["Label"])

    y_train = (train_df["Label"] != BENIGN_LABEL).astype(int)
    y_test = (test_df["Label"] != BENIGN_LABEL).astype(int)

    # Clean data
    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)

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
    X_test = X_test.clip(lower=-1e6, upper=1e6)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    n_features = X_train.shape[1]

    X_train = X_train.reshape(-1, TIME_STEPS, n_features)
    X_test = X_test.reshape(-1, TIME_STEPS, n_features)

    for run in range(1, N_RUNS + 1):

        tf.keras.backend.clear_session()
        tf.random.set_seed(RANDOM_STATE + run)
        np.random.seed(RANDOM_STATE + run)

        model = build_model(n_features)

        model.fit(
            X_train,
            y_train,
            epochs=EPOCHS_PER_RUN,
            batch_size=BATCH_SIZE,
            verbose=0
        )

        y_prob = model.predict(X_test, verbose=0).ravel()
        y_pred = (y_prob >= 0.5).astype(int)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        iter_rows.append({
            "Run": run,
            "Attack/File": attack_name,
            "Accuracy": round(acc, 6),
            "Precision": round(prec, 6),
            "Recall": round(rec, 6),
            "F1": round(f1, 6),
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp
        })

        if run % 10 == 0:
            print(f"  Run {run}/100")

    final_rows.append({
        "Attack/File": attack_name,
        "Accuracy": round(acc, 6),
        "Precision": round(prec, 6),
        "Recall": round(rec, 6),
        "F1": round(f1, 6),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    })

    # Confusion matrix image for this attack
    disp = ConfusionMatrixDisplay(cm, display_labels=["Benign", "Attack"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, values_format="d", colorbar=False)
    ax.set_title(f"LSTM — {attack_name}", fontsize=11)
    plt.tight_layout()

    cm_path = os.path.join(OUTPUT_DIR, f"LSTM_CM_{attack_name}.png")
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()

# ─────────────────────────────────────────
# Save Results
# ─────────────────────────────────────────
iter_df = pd.DataFrame(iter_rows)

iter_csv = os.path.join(OUTPUT_DIR, "LSTM_100_runs_all_metrics.csv")
iter_df.to_csv(iter_csv, index=False)

summary_df = (
    iter_df.groupby("Attack/File")[["Accuracy", "Precision", "Recall", "F1"]]
    .agg(["mean", "std", "min", "max"])
    .reset_index()
    .sort_values(("F1", "mean"), ascending=False)
)

summary_csv = os.path.join(OUTPUT_DIR, "LSTM_100_runs_summary.csv")
summary_df.to_csv(summary_csv, index=False)

final_df = pd.DataFrame(final_rows).sort_values("F1", ascending=False)
final_csv = os.path.join(OUTPUT_DIR, "LSTM_final_comparison.csv")
final_df.to_csv(final_csv, index=False)

mean_df = iter_df.groupby("Attack/File")[["Accuracy", "Precision", "Recall", "F1"]].mean()

mean_df.plot(kind="bar", figsize=(14, 5), ylim=(0, 1.05))
plt.title("LSTM — Mean Metrics over 100 Runs per Dataset")
plt.ylabel("Score")
plt.xlabel("Dataset")
plt.xticks(rotation=30, ha="right")
plt.legend(loc="lower right")
plt.tight_layout()

bar_path = os.path.join(OUTPUT_DIR, "LSTM_mean_metrics_bar.png")
plt.savefig(bar_path, dpi=150)
plt.close()

print(f"\nSaved iterations CSV → {iter_csv}")
print(f"Saved summary CSV → {summary_csv}")
print(f"Saved final comparison → {final_csv}")
print(f"Saved bar chart → {bar_path}")

print("\n✓ LSTM — Done!")