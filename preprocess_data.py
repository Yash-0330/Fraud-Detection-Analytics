
import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Load the dataset
df = pd.read_csv("data/creditcard.csv")

print("Original dataset shape:", df.shape)

# 2. Separate features (X) and target (y)
X = df.drop(columns=["Class"])
y = df["Class"]

print("Features shape:", X.shape)
print("Target shape:", y.shape)

# 3. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n--- SPLIT RESULTS ---")
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print("Fraud cases in training:", int(y_train.sum()))
print("Fraud cases in testing:", int(y_test.sum()))

# 4. Scale numerical features
# Fit the scaler ONLY on training data.
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n--- SCALING RESULTS ---")
print("Training data shape:", X_train_scaled.shape)
print("Testing data shape:", X_test_scaled.shape)

# 5. Save everything for the next steps
Path("models").mkdir(exist_ok=True)

prepared_data = {
    "X_train": X_train,
    "X_test": X_test,
    "y_train": y_train,
    "y_test": y_test,
    "X_train_scaled": X_train_scaled,
    "X_test_scaled": X_test_scaled,
    "scaler": scaler,
    "feature_names": list(X.columns)
}

joblib.dump(prepared_data, "models/prepared_data.joblib")

print("\nPreprocessing completed successfully!")
print("Saved to models/prepared_data.joblib")
