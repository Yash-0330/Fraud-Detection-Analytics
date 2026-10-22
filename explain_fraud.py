
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# Project paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
CHART_DIR = BASE_DIR / "charts"
REPORT_DIR = BASE_DIR / "reports"

CHART_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

# Load prepared data and trained Random Forest
prepared = joblib.load(MODEL_DIR / "prepared_data.joblib")
model = joblib.load(MODEL_DIR / "random_forest.joblib")

feature_names = prepared["feature_names"]

# Use a small sample to keep SHAP computation manageable
X_train = prepared["X_train"]
X_test = prepared["X_test"]

if isinstance(X_train, pd.DataFrame):
    background = X_train.sample(
        n=min(100, len(X_train)),
        random_state=42,
    )
else:
    rng = np.random.default_rng(42)
    indices = rng.choice(
        len(X_train),
        size=min(100, len(X_train)),
        replace=False,
    )
    background = X_train[indices]

if isinstance(X_test, pd.DataFrame):
    X_explain = X_test.iloc[:200].copy()
else:
    X_explain = pd.DataFrame(
        X_test[:200],
        columns=feature_names,
    )

# Convert background data to a DataFrame for clear feature labels
if not isinstance(background, pd.DataFrame):
    background = pd.DataFrame(
        background,
        columns=feature_names,
    )

print("Calculating SHAP values...")
print("Explaining up to 200 test transactions.")

# Explain the Random Forest's predicted fraud probability
explainer = shap.TreeExplainer(
    model,
    data=background,
    feature_perturbation="interventional",
    model_output="probability",
)

raw_values = explainer.shap_values(
    X_explain,
    check_additivity=False,
)

# SHAP versions differ in how they return multiclass values.
# Select class 1, which represents fraud.
if isinstance(raw_values, list):
    shap_values = np.asarray(raw_values[1])
else:
    shap_values = np.asarray(raw_values)
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

expected_value = np.asarray(explainer.expected_value)
base_value = (
    float(expected_value[1])
    if expected_value.ndim > 0
    else float(expected_value)
)

# 1. Global feature importance
plt.figure()
shap.summary_plot(
    shap_values,
    X_explain,
    plot_type="bar",
    show=False,
)
plt.tight_layout()
plt.savefig(
    CHART_DIR / "shap_feature_importance.png",
    dpi=150,
    bbox_inches="tight",
)
plt.close()

# 2. SHAP beeswarm plot: direction and size of contributions
plt.figure()
shap.summary_plot(
    shap_values,
    X_explain,
    show=False,
)
plt.tight_layout()
plt.savefig(
    CHART_DIR / "shap_beeswarm.png",
    dpi=150,
    bbox_inches="tight",
)
plt.close()

# 3. Explain one individual transaction
row_number = 0
row = X_explain.iloc[row_number]

probability = model.predict_proba(
    X_explain.iloc[[row_number]]
)[0, 1]

threshold = 0.50
prediction = int(probability >= threshold)

explanation = shap.Explanation(
    values=shap_values[row_number],
    base_values=base_value,
    data=row.to_numpy(),
    feature_names=list(X_explain.columns),
)

plt.figure()
shap.plots.waterfall(
    explanation,
    max_display=12,
    show=False,
)
plt.tight_layout()
plt.savefig(
    CHART_DIR / "shap_transaction_explanation.png",
    dpi=150,
    bbox_inches="tight",
)
plt.close()

# 4. Save the largest feature contributions for this transaction
contributions = pd.DataFrame({
    "Feature": X_explain.columns,
    "Feature Value": row.to_numpy(),
    "SHAP Value": shap_values[row_number],
})

contributions["Absolute SHAP"] = (
    contributions["SHAP Value"].abs()
)

contributions = contributions.sort_values(
    "Absolute SHAP",
    ascending=False,
)

contributions.to_csv(
    REPORT_DIR / "shap_transaction_contributions.csv",
    index=False,
)

# 5. Print a simple explanation
print("\nINDIVIDUAL TRANSACTION EXPLANATION")
print("---------------------------------")
print(f"Test transaction row: {row_number}")
print(f"Predicted fraud probability: {probability:.2%}")
print(f"Decision threshold: {threshold:.2f}")
print(
    "Prediction:",
    "FRAUD ALERT" if prediction == 1 else "NOT FLAGGED AS FRAUD",
)

print("\nTop feature contributions:")
print(
    contributions[
        ["Feature", "Feature Value", "SHAP Value"]
    ].head(10).round(5).to_string(index=False)
)

print("\nFiles generated:")
print("charts/shap_feature_importance.png")
print("charts/shap_beeswarm.png")
print("charts/shap_transaction_explanation.png")
print("reports/shap_transaction_contributions.csv")

print(
    "\nInterpretation: positive SHAP values push the model's "
    "fraud probability higher; negative values push it lower. "
    "These explain model behavior, not proof of actual fraud."
)
