
import json
import time
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    average_precision_score,
    precision_recall_curve,
    auc,
    confusion_matrix,
    classification_report,
)

# 1. Load the same train/test split used by Logistic Regression
data = joblib.load("models/prepared_data.joblib")

# Random Forest uses the original, unscaled features
X_train = data["X_train"]
X_test = data["X_test"]
y_train = data["y_train"]
y_test = data["y_test"]

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

# 2. Create the Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=12,
    min_samples_leaf=2,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1,
)

# 3. Train the model
print("\nTraining Random Forest...")
start_time = time.time()

model.fit(X_train, y_train)

print(f"Training completed in {time.time() - start_time:.1f} seconds.")

# 4. Predict fraud probabilities
fraud_probabilities = model.predict_proba(X_test)[:, 1]

# 5. Convert probabilities into predictions
threshold = 0.50
y_pred = (fraud_probabilities >= threshold).astype(int)

# 6. Calculate evaluation metrics
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
average_precision = average_precision_score(
    y_test, fraud_probabilities
)

pr_precision, pr_recall, _ = precision_recall_curve(
    y_test, fraud_probabilities
)
pr_auc = auc(pr_recall[::-1], pr_precision[::-1])

print("\n--- RANDOM FOREST RESULTS ---")
print(f"Threshold: {threshold:.2f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"PR-AUC: {pr_auc:.4f}")
print(f"Average Precision (AP): {average_precision:.4f}")

print("\n--- CONFUSION MATRIX ---")
print(confusion_matrix(y_test, y_pred))

print("\n--- CLASSIFICATION REPORT ---")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Genuine", "Fraud"],
    zero_division=0,
))

# 7. Save the trained model and its metrics
Path("models").mkdir(exist_ok=True)

joblib.dump(model, "models/random_forest.joblib")

metrics = {
    "model": "Random Forest",
    "threshold": threshold,
    "precision": float(precision),
    "recall": float(recall),
    "pr_auc": float(pr_auc),
    "average_precision": float(average_precision),
}

with open("models/random_forest_metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)

print("\nRandom Forest model saved!")
print("File: models/random_forest.joblib")
print("Metrics: models/random_forest_metrics.json")
