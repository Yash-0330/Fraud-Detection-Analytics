
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    average_precision_score,
)

# Project folders
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
CHART_DIR = BASE_DIR / "charts"
REPORT_DIR = BASE_DIR / "reports"

CHART_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

# Load prepared data
prepared = joblib.load(MODEL_DIR / "prepared_data.joblib")

X_test = prepared["X_test"]
X_test_scaled = prepared["X_test_scaled"]
y_test = prepared["y_test"]

# Load trained models
logistic_model = joblib.load(
    MODEL_DIR / "logistic_regression.joblib"
)
random_forest_model = joblib.load(
    MODEL_DIR / "random_forest.joblib"
)

# Each model must receive the same feature representation
models = {
    "Logistic Regression": (
        logistic_model,
        X_test_scaled,
    ),
    "Random Forest": (
        random_forest_model,
        X_test,
    ),
}

# Thresholds to compare
thresholds = [
    0.10, 0.15, 0.20, 0.25, 0.30,
    0.35, 0.40, 0.45, 0.50, 0.60, 0.70
]

results = []

for model_name, (model, X) in models.items():
    # Probability of the positive class: fraud (Class = 1)
    probabilities = model.predict_proba(X)[:, 1]

    # Average precision summarizes the precision-recall curve
    avg_precision = average_precision_score(
        y_test, probabilities
    )

    for threshold in thresholds:
        predictions = (
            probabilities >= threshold
        ).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1],
        ).ravel()

        precision = precision_score(
            y_test, predictions, zero_division=0
        )
        recall = recall_score(
            y_test, predictions, zero_division=0
        )
        f1 = f1_score(
            y_test, predictions, zero_division=0
        )
        f2 = fbeta_score(
            y_test, predictions, beta=2,
            zero_division=0
        )

        results.append({
            "Model": model_name,
            "Threshold": threshold,
            "True Positives": int(tp),
            "False Positives": int(fp),
            "Missed Frauds": int(fn),
            "True Negatives": int(tn),
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "F2 Score": f2,
            "Average Precision": avg_precision,
        })

# Save all results
results_df = pd.DataFrame(results)

csv_path = REPORT_DIR / "threshold_evaluation.csv"
results_df.to_csv(csv_path, index=False)

# Find the threshold with the highest F2 score per model.
# F2 gives more weight to recall than precision.
best_rows = results_df.loc[
    results_df.groupby("Model")["F2 Score"].idxmax()
].copy()

best_rows.to_csv(
    REPORT_DIR / "recommended_thresholds.csv",
    index=False,
)

# Display key results
display_columns = [
    "Model",
    "Threshold",
    "True Positives",
    "False Positives",
    "Missed Frauds",
    "Precision",
    "Recall",
    "F2 Score",
]

print("\nTHRESHOLD COMPARISON")
print(
    results_df[display_columns].round(4).to_string(
        index=False
    )
)

print("\nBEST THRESHOLD BY F2 SCORE")
print(
    best_rows[display_columns].round(4).to_string(
        index=False
    )
)

# Plot recall and false positives against threshold
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for model_name in results_df["Model"].unique():
    model_results = results_df[
        results_df["Model"] == model_name
    ]

    axes[0].plot(
        model_results["Threshold"],
        model_results["Recall"],
        marker="o",
        label=model_name,
    )

    axes[1].plot(
        model_results["Threshold"],
        model_results["False Positives"],
        marker="o",
        label=model_name,
    )

axes[0].set_title("Fraud Recall by Threshold")
axes[0].set_xlabel("Decision Threshold")
axes[0].set_ylabel("Recall")
axes[0].grid(True)
axes[0].legend()

axes[1].set_title("False Positives by Threshold")
axes[1].set_xlabel("Decision Threshold")
axes[1].set_ylabel("False Positive Count")
axes[1].grid(True)
axes[1].legend()

plt.tight_layout()

chart_path = CHART_DIR / "threshold_analysis.png"
plt.savefig(chart_path, dpi=150)
plt.close()

print(f"\nResults saved to: {csv_path}")
print(
    "Recommended thresholds saved to: "
    f"{REPORT_DIR / 'recommended_thresholds.csv'}"
)
print(f"Chart saved to: {chart_path}")

print(
    "\nNote: This initial analysis uses the test set. "
    "For a final unbiased evaluation, select the threshold "
    "on validation data and evaluate it once on a separate "
    "test set."
)
