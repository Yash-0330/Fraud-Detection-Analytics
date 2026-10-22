
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Fraud Detection | SHAP Dashboard",
    page_icon="🔍",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATA_PATH = MODELS_DIR / "prepared_data.joblib"
MODEL_PATH = MODELS_DIR / "random_forest.joblib"


# =========================================================
# LOAD MODEL AND PREPARED DATA
# =========================================================

@st.cache_resource
def load_project():
    """Load the saved Random Forest model and prepared data."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Prepared data not found: {DATA_PATH}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Random Forest model not found: {MODEL_PATH}"
        )

    prepared_data = joblib.load(DATA_PATH)
    model = joblib.load(MODEL_PATH)

    return prepared_data, model


def prepare_dataframe(data, columns=None):
    """Convert prepared data into a DataFrame safely."""

    if isinstance(data, pd.DataFrame):
        return data.copy()

    array = np.asarray(data)

    if columns is None:
        raise ValueError(
            "Feature names are missing from prepared_data.joblib. "
            "Check preprocess_data.py."
        )

    if array.ndim != 2 or array.shape[1] != len(columns):
        raise ValueError(
            f"Data shape {array.shape} does not match "
            f"{len(columns)} feature names."
        )

    return pd.DataFrame(array, columns=columns)


# =========================================================
# SHAP HELPERS
# =========================================================

def create_explainer(model, background):
    """Create a TreeExplainer for the saved tree-based model."""

    import shap

    return shap.TreeExplainer(
        model,
        data=background,
    )


def get_fraud_shap_values(shap_result, n_features):
    """Extract SHAP values for the fraud class (class 1)."""

    if isinstance(shap_result, list):
        if len(shap_result) > 1:
            values = np.asarray(shap_result[1])
        else:
            values = np.asarray(shap_result[0])

    elif hasattr(shap_result, "values"):
        values = np.asarray(shap_result.values)

    else:
        values = np.asarray(shap_result)

    if values.ndim == 3:
        # Samples x features x classes
        if values.shape[-1] > 1:
            values = values[:, :, 1]
        # Classes x samples x features
        elif values.shape[0] > 1:
            values = values[1]

    if values.ndim == 1:
        values = values.reshape(1, -1)

    if values.ndim != 2 or values.shape[1] != n_features:
        raise ValueError(
            f"Unexpected SHAP output shape: {values.shape}. "
            f"Expected samples x {n_features} features."
        )

    return values


def get_fraud_base_value(explainer):
    """Return the expected output for the fraud class."""

    expected = np.asarray(explainer.expected_value).ravel()

    if len(expected) > 1:
        return float(expected[1])

    if len(expected) == 1:
        return float(expected[0])

    raise ValueError("SHAP did not provide a base value.")


# =========================================================
# MAIN DASHBOARD
# =========================================================

st.title("🔍 Fraud Detection & Transaction Risk Analytics")
st.caption(
    "Model insights, transaction risk, and explainable AI using SHAP"
)

try:
    prepared_data, model = load_project()

    if "X_test" not in prepared_data:
        raise KeyError(
            "'X_test' is missing from prepared_data.joblib."
        )

    if "y_test" not in prepared_data:
        raise KeyError(
            "'y_test' is missing from prepared_data.joblib."
        )

    saved_names = prepared_data.get("feature_names")

    # Recover feature names from a saved DataFrame when possible.
    if isinstance(prepared_data["X_test"], pd.DataFrame):
        feature_names = list(prepared_data["X_test"].columns)
    elif saved_names is not None:
        feature_names = list(saved_names)
    else:
        raise ValueError(
            "Feature names are missing. Please check "
            "preprocess_data.py and regenerate prepared_data.joblib."
        )

    X_test_df = prepare_dataframe(
        prepared_data["X_test"],
        feature_names,
    )

    y_test = np.asarray(prepared_data["y_test"]).reshape(-1)

    if len(X_test_df) != len(y_test):
        raise ValueError(
            "The number of test transactions does not match "
            "the number of test labels."
        )

    if hasattr(model, "n_features_in_"):
        if model.n_features_in_ != len(feature_names):
            raise ValueError(
                "The saved model and prepared data have different "
                "numbers of features. Use the matching model/data pair."
            )

    # Keep data available for all dashboard sections.
    st.session_state["model"] = model
    st.session_state["X_test_df"] = X_test_df
    st.session_state["feature_names"] = feature_names

    # Prefer unscaled training data as the background.
    if "X_train" in prepared_data:
        X_train_df = prepare_dataframe(
            prepared_data["X_train"],
            feature_names,
        )
        background = X_train_df.sample(
            n=min(100, len(X_train_df)),
            random_state=42,
        )
    else:
        background = X_test_df.sample(
            n=min(100, len(X_test_df)),
            random_state=42,
        )

except Exception as exc:
    st.error("Could not load the fraud detection project.")
    st.exception(exc)
    st.stop()


# =========================================================
# OVERVIEW AND KEY METRICS
# =========================================================

total_transactions = len(X_test_df)
fraud_count = int(np.sum(y_test == 1))
normal_count = int(np.sum(y_test == 0))
fraud_rate = (
    fraud_count / total_transactions * 100
    if total_transactions else 0
)

col1, col2, col3, col4 = st.columns(4)

col1.metric("Test Transactions", f"{total_transactions:,}")
col2.metric("Fraud Transactions", f"{fraud_count:,}")
col3.metric("Normal Transactions", f"{normal_count:,}")
col4.metric("Fraud Rate", f"{fraud_rate:.3f}%")


# =========================================================
# FRAUD DISTRIBUTION
# =========================================================

st.divider()
st.subheader("1. Test Data Distribution")

distribution = pd.DataFrame({
    "Transaction Type": ["Normal", "Fraud"],
    "Count": [normal_count, fraud_count],
})

col1, col2 = st.columns(2)

with col1:
    st.dataframe(
        distribution,
        hide_index=True,
        width="stretch",
    )

with col2:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(
        distribution["Transaction Type"],
        distribution["Count"],
    )
    ax.set_title("Normal vs Fraud Transactions")
    ax.set_ylabel("Number of transactions")
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)


# =========================================================
# INDIVIDUAL TRANSACTION EXPLANATION
# =========================================================

st.divider()
st.subheader("2. Individual Transaction SHAP Explanation")

st.write(
    "Select a test transaction to examine which features influenced "
    "the Random Forest model's prediction. SHAP values explain the "
    "model output; they do not prove that a transaction is fraudulent."
)

transaction_index = st.number_input(
    "Test transaction index",
    min_value=0,
    max_value=max(0, len(X_test_df) - 1),
    value=0,
    step=1,
)

if st.button("Generate SHAP explanation", type="primary"):
    try:
        import shap

        transaction = X_test_df.iloc[[int(transaction_index)]].copy()

        with st.spinner("Calculating SHAP values..."):
            explainer = create_explainer(model, background)

            shap_result = explainer.shap_values(
                transaction,
                check_additivity=False,
            )

            values = get_fraud_shap_values(
                shap_result,
                len(feature_names),
            )[0]

            base_value = get_fraud_base_value(explainer)

            # Obtain the predicted fraud probability.
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(transaction)[0]
                classes = list(model.classes_)

                if 1 in classes:
                    fraud_probability = float(
                        probabilities[classes.index(1)]
                    )
                else:
                    fraud_probability = float(probabilities[-1])
            else:
                fraud_probability = float(
                    model.predict(transaction)[0] == 1
                )

            predicted_class = int(
                fraud_probability >= 0.5
            )

            st.metric(
                "Predicted Fraud Probability",
                f"{fraud_probability:.2%}",
            )

            st.write(
                "Model prediction:",
                "Fraud" if predicted_class == 1 else "Normal",
            )

            st.write(
                "Actual test label:",
                "Fraud" if y_test[int(transaction_index)] == 1
                else "Normal",
            )

            st.info(
                "Positive SHAP values increase the model output "
                "relative to its baseline; negative values decrease it. "
                "The precise output scale depends on the explainer."
            )

            explanation = shap.Explanation(
                values=values,
                base_values=base_value,
                data=transaction.iloc[0].to_numpy(),
                feature_names=feature_names,
            )

            st.markdown("**Feature contribution waterfall**")

            shap.plots.waterfall(
                explanation,
                max_display=12,
                show=False,
            )

            fig = plt.gcf()
            st.pyplot(fig, width="stretch")
            plt.close(fig)

            contributions = pd.DataFrame({
                "Feature": feature_names,
                "Feature Value": transaction.iloc[0].to_numpy(),
                "SHAP Value": values,
            })

            contributions["Absolute Impact"] = (
                contributions["SHAP Value"].abs()
            )

            contributions = contributions.sort_values(
                "Absolute Impact",
                ascending=False,
            )

            st.markdown("**Most influential features**")

            st.dataframe(
                contributions.head(10).round(6),
                hide_index=True,
                width="stretch",
            )

            st.download_button(
                "Download individual SHAP explanation",
                data=contributions.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name="transaction_shap_explanation.csv",
                mime="text/csv",
                width="stretch",
            )

    except Exception as exc:
        st.error("Individual SHAP explanation failed.")
        st.exception(exc)


# =========================================================
# GLOBAL FEATURE IMPORTANCE
# =========================================================

st.divider()
st.subheader("3. Global Feature Importance")

st.write(
    "This chart summarizes which features influence model predictions "
    "across a sample of test transactions. It describes model behavior, "
    "not necessarily causal relationships."
)

if st.button("Generate global SHAP summary"):
    try:
        import shap

        sample = X_test_df.sample(
            n=min(100, len(X_test_df)),
            random_state=42,
        )

        with st.spinner("Calculating global feature importance..."):
            explainer = create_explainer(model, background)

            shap_result = explainer.shap_values(
                sample,
                check_additivity=False,
            )

            global_values = get_fraud_shap_values(
                shap_result,
                len(feature_names),
            )

            fig, ax = plt.subplots(figsize=(10, 6))

            shap.summary_plot(
                global_values,
                sample,
                plot_type="bar",
                max_display=15,
                show=False,
            )

            fig = plt.gcf()
            st.pyplot(fig, width="stretch")
            plt.close(fig)

            # Numeric global importance table.
            importance = np.abs(global_values).mean(axis=0)

            importance_df = pd.DataFrame({
                "Feature": feature_names,
                "Mean Absolute SHAP": importance,
            }).sort_values(
                "Mean Absolute SHAP",
                ascending=False,
            )

            st.markdown("**Global feature importance table**")

            st.dataframe(
                importance_df.head(15).round(6),
                hide_index=True,
                width="stretch",
            )

            st.download_button(
                "Download global feature importance",
                data=importance_df.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name="global_shap_feature_importance.csv",
                mime="text/csv",
                width="stretch",
            )

    except Exception as exc:
        st.error("Global SHAP analysis failed.")
        st.exception(exc)


# =========================================================
# FOOTER
# =========================================================

st.divider()
st.caption(
    "Fraud Detection & Transaction Risk Analytics | "
    "Python • Pandas • Scikit-learn • Streamlit • SHAP"
)
