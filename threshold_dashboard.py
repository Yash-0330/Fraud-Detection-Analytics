
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import confusion_matrix, precision_score, recall_score

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

st.set_page_config(
    page_title="Fraud Threshold Evaluation",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ Fraud Detection: Threshold Evaluation")
st.write(
    "Compare how the decision threshold affects fraud detection "
    "and false alarms."
)

@st.cache_resource
def load_prepared_data():
    return joblib.load(MODEL_DIR / "prepared_data.joblib")

@st.cache_resource
def load_models():
    return {
        "Logistic Regression": joblib.load(
            MODEL_DIR / "logistic_regression.joblib"
        ),
        "Random Forest": joblib.load(
            MODEL_DIR / "random_forest.joblib"
        ),
    }

prepared = load_prepared_data()
models = load_models()

model_name = st.selectbox(
    "Select model",
    list(models.keys()),
)

threshold = st.slider(
    "Decision threshold",
    min_value=0.01,
    max_value=0.99,
    value=0.50,
    step=0.01,
)

# Use the matching feature representation for each model
if model_name == "Logistic Regression":
    X_test = prepared["X_test_scaled"]
else:
    X_test = prepared["X_test"]

y_test = prepared["y_test"]
model = models[model_name]

probabilities = model.predict_proba(X_test)[:, 1]
predictions = (probabilities >= threshold).astype(int)

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

col1, col2, col3, col4 = st.columns(4)

col1.metric("Frauds Detected", int(tp))
col2.metric("False Alarms", int(fp))
col3.metric("Missed Frauds", int(fn))
col4.metric("Recall", f"{recall:.2%}")

st.subheader("Confusion Matrix")

matrix = pd.DataFrame(
    [[tn, fp], [fn, tp]],
    index=["Actual Legitimate", "Actual Fraud"],
    columns=["Predicted Legitimate", "Predicted Fraud"],
)

st.dataframe(matrix, use_container_width=True)

st.subheader("Additional Metrics")

metric_col1, metric_col2 = st.columns(2)
metric_col1.metric("Precision", f"{precision:.2%}")
metric_col2.metric("True Negatives", int(tn))

st.caption(
    "These results use the saved test set for exploration. "
    "Repeatedly choosing a threshold based on test results can "
    "bias final evaluation. Prefer a validation set for threshold "
    "selection, followed by one final test evaluation."
)
