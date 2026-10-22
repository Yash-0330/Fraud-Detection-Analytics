
import joblib
import json

from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    average_precision_score,
    precision_recall_curve,
    auc
)

# 1. Load the prepared data
data = joblib.load("models/prepared_data.joblib")

X_train = data["X_train_scaled"]
X_test = data["X_test_scaled"]
y_train = data["y_train"]
y_test = data["y_test"]

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

# 2. Create the Logistic Regression model
# class_weight="balanced" helps account for rare fraud cases.
model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=42
)

# 3. Train the model
print("\nTraining Logistic Regression...")
model.fit(X_train, y_train)
print("Training completed!")

# 4. Predict fraud probabilities on unseen test data
fraud_probabilities = model.predict_proba(X_test)[:, 1]

# 5. Convert probabilities into class predictions
# Default decision threshold = 0.50
threshold = 0.50
y_pred = (fraud_probabilities >= threshold).astype(int)

# 6. Calculate evaluation metrics
precision = precision_score(
    y_test, y_pred, zero_division=0
)
recall = recall_score(
    y_test, y_pred, zero_division=0
)

# Average Precision summarizes the precision-recall curve.
average_precision = average_precision_score(
    y_test, fraud_probabilities
)

# Trapezoidal area under the precision-recall curve.
pr_precision, pr_recall, _ = precision_recall_curve(
    y_test, fraud_probabilities
)
pr_auc = auc(pr_recall[::-1], pr_precision[::-1])

print("\n--- LOGISTIC REGRESSION RESULTS ---")
print(f"Threshold: {threshold:.2f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"PR-AUC: {pr_auc:.4f}")
print(f"Average Precision (AP): {average_precision:.4f}")

# 7. Display confusion matrix
print("\n--- CONFUSION MATRIX ---")
print(confusion_matrix(y_test, y_pred))

print("\n--- CLASSIFICATION REPORT ---")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Genuine", "Fraud"],
    zero_division=0
))

# 8. Save the trained model
Path("models").mkdir(exist_ok=True)

joblib.dump(model, "models/logistic_regression.joblib")

# Save metrics for later comparison
metrics = {
    "model": "Logistic Regression",
    "threshold": threshold,
    "precision": float(precision),
    "recall": float(recall),
    "pr_auc": float(pr_auc),
    "average_precision": float(average_precision)
}

with open("models/logistic_metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)

print("\nModel saved successfully!")
print("File: models/logistic_regression.joblib")
print("Metrics saved to models/logistic_metrics.json")
