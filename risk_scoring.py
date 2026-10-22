
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"
REPORT_DIR.mkdir(exist_ok=True)

# Load the trained Random Forest model and test data
model = joblib.load(MODEL_DIR / "random_forest.joblib")
prepared = joblib.load(MODEL_DIR / "prepared_data.joblib")

X_test = prepared["X_test"]
y_test = np.asarray(prepared["y_test"])

# Calculate fraud probabilities
probabilities = model.predict_proba(X_test)[:, 1]

# Convert probabilities to scores between 0 and 100
risk_scores = np.round(probabilities * 100, 2)

# Example business rules for a demonstration
def assign_risk_level(score):
    if score >= 80:
        return "Critical"
    elif score >= 60:
        return "High"
    elif score >= 30:
        return "Medium"
    return "Low"

risk_levels = [
    assign_risk_level(score)
    for score in risk_scores
]

# Use a separate classification threshold
decision_threshold = 0.50
predictions = (
    probabilities >= decision_threshold
).astype(int)

# Build the risk report
report = pd.DataFrame({
    "Actual_Class": y_test,
    "Fraud_Probability": np.round(probabilities, 5),
    "Risk_Score": risk_scores,
    "Risk_Level": risk_levels,
    "Predicted_Class": predictions,
})

output_path = REPORT_DIR / "fraud_risk_report.csv"
report.to_csv(output_path, index=False)

print("\nFRAUD RISK REPORT")
print("-" * 45)
print(f"Total transactions evaluated: {len(report)}")
print(f"Transactions flagged: {int(predictions.sum())}")
print(f"Critical risk: {(report['Risk_Level'] == 'Critical').sum()}")
print(f"High risk: {(report['Risk_Level'] == 'High').sum()}")
print(f"Medium risk: {(report['Risk_Level'] == 'Medium').sum()}")
print(f"Low risk: {(report['Risk_Level'] == 'Low').sum()}")

print("\nSample results:")
print(report.head(10).to_string(index=False))

print(f"\nReport saved to: {output_path}")
print(
    "\nNote: Risk categories are illustrative business rules. "
    "They should be validated against operational costs and "
    "fraud investigation capacity before real-world use."
)
