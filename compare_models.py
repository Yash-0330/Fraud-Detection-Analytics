
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# Load saved metrics from both models
with open("models/logistic_metrics.json", "r") as file:
    logistic = json.load(file)

with open("models/random_forest_metrics.json", "r") as file:
    forest = json.load(file)

# Create a comparison table
comparison = pd.DataFrame([logistic, forest])

comparison = comparison[
    [
        "model",
        "precision",
        "recall",
        "pr_auc",
        "average_precision",
    ]
]

comparison.columns = [
    "Model",
    "Precision",
    "Recall",
    "PR-AUC",
    "Average Precision",
]

print("\n===== MODEL COMPARISON =====")
print(comparison.round(4).to_string(index=False))

# Save comparison results
Path("charts").mkdir(exist_ok=True)
Path("models").mkdir(exist_ok=True)

comparison.to_csv("models/model_comparison.csv", index=False)

# Plot metrics side by side
metrics = [
    "Precision",
    "Recall",
    "PR-AUC",
    "Average Precision",
]

ax = comparison.set_index("Model")[metrics].plot(
    kind="bar",
    figsize=(10, 6),
    ylim=(0, 1),
    rot=0,
)

ax.set_title("Fraud Detection Model Comparison")
ax.set_ylabel("Score")
ax.set_xlabel("Model")
ax.legend(loc="lower right")
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("charts/model_comparison.png", dpi=150)
plt.show()

print("\nComparison saved:")
print("models/model_comparison.csv")
print("charts/model_comparison.png")
