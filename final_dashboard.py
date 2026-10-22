
from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# =========================================================
# CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Fraud Detection & Risk Analytics",
    page_icon="🔍",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

PREPARED_PATH = MODELS_DIR / "prepared_data.joblib"
MODEL_PATH = MODELS_DIR / "random_forest.joblib"
DATA_PATH = DATA_DIR / "creditcard.csv"
COMPARISON_PATH = MODELS_DIR / "model_comparison.csv"
METRICS_PATH = MODELS_DIR / "random_forest_metrics.json"


# =========================================================
# LOAD EXISTING MODEL AND DATA
# =========================================================

@st.cache_resource
def load_project():
    if not PREPARED_PATH.exists():
        raise FileNotFoundError(f"Missing file: {PREPARED_PATH}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing file: {MODEL_PATH}")

    prepared = joblib.load(PREPARED_PATH)
    model = joblib.load(MODEL_PATH)

    names = prepared.get("feature_names")

    if isinstance(prepared.get("X_test"), pd.DataFrame):
        names = list(prepared["X_test"].columns)

    if names is None:
        raise ValueError(
            "Feature names are missing from prepared_data.joblib. "
            "Check preprocess_data.py and regenerate the prepared data."
        )

    names = list(names)

    def as_frame(key):
        if key not in prepared:
            return None

        values = prepared[key]

        if isinstance(values, pd.DataFrame):
            return values.copy().loc[:, names]

        return pd.DataFrame(values, columns=names)

    return (
        model,
        names,
        as_frame("X_train"),
        as_frame("X_test"),
        np.asarray(prepared["y_test"]).reshape(-1),
    )


try:
    model, feature_names, X_train_df, X_test_df, y_test = load_project()
except Exception as exc:
    st.error("The model or prepared data could not be loaded.")
    st.exception(exc)
    st.stop()

if X_test_df is None:
    st.error("X_test is missing from the prepared data.")
    st.stop()

if len(X_test_df) != len(y_test):
    st.error("Test features and labels have different lengths.")
    st.stop()

if hasattr(model, "n_features_in_"):
    if model.n_features_in_ != len(feature_names):
        st.error("The saved model and feature list do not match.")
        st.stop()


# =========================================================
# REUSABLE HELPERS
# =========================================================

@st.cache_data
def load_analytics_data(path_string):
    path = Path(path_string)

    if not path.exists():
        return None

    # Load only the columns needed for transaction analytics.
    return pd.read_csv(
        path,
        usecols=lambda col: col in {"Time", "Amount", "Class"},
    )


def fraud_probability(dataframe):
    """Return the probability of class 1 (fraud)."""

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(dataframe)
        classes = list(model.classes_)

        if 1 not in classes:
            raise ValueError("The model has no fraud class (Class=1).")

        return probabilities[:, classes.index(1)]

    raise ValueError(
        "This model does not support predict_proba()."
    )


def risk_category(probability):
    if probability >= 0.80:
        return "Critical"
    if probability >= 0.50:
        return "High"
    if probability >= 0.20:
        return "Medium"
    return "Low"


def get_shap_values(result, n_features):
    """Normalize common SHAP output formats for binary classifiers."""

    if isinstance(result, list):
        values = np.asarray(result[1] if len(result) > 1 else result[0])
    elif hasattr(result, "values"):
        values = np.asarray(result.values)
    else:
        values = np.asarray(result)

    if values.ndim == 3:
        if values.shape[-1] > 1:
            values = values[:, :, 1]
        elif values.shape[0] > 1:
            values = values[1]

    if values.ndim == 1:
        values = values.reshape(1, -1)

    if values.ndim != 2 or values.shape[1] != n_features:
        raise ValueError(
            f"Unsupported SHAP output shape {values.shape}; "
            f"expected samples x {n_features} features."
        )

    return values


def get_shap_base_value(explainer):
    value = np.asarray(explainer.expected_value).ravel()

    if len(value) > 1:
        return float(value[1])
    if len(value) == 1:
        return float(value[0])

    raise ValueError("The SHAP base value is unavailable.")


@st.cache_resource
def create_shap_explainer(background):
    import shap

    return shap.TreeExplainer(model, data=background)


def prepare_upload(uploaded_file):
    uploaded = pd.read_csv(uploaded_file)

    # Remove labels if the supplied dataset includes them.
    X = uploaded.drop(
        columns=["Class", "Prediction", "Fraud Probability", "Risk Level"],
        errors="ignore",
    )

    missing = [c for c in feature_names if c not in X.columns]
    extra = [c for c in X.columns if c not in feature_names]

    if missing:
        raise ValueError(
            "Missing required model features: "
            + ", ".join(missing[:10])
        )

    if extra:
        st.warning(
            "Extra columns were ignored: " + ", ".join(extra[:10])
        )

    X = X.loc[:, feature_names]

    if X.isna().any().any():
        raise ValueError("The uploaded data contains missing values.")

    if not all(pd.api.types.is_numeric_dtype(X[c]) for c in X.columns):
        raise ValueError("All model input features must be numeric.")

    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("The uploaded data contains infinite values.")

    return uploaded, X


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("Dashboard Controls")

threshold = st.sidebar.slider(
    "Fraud alert threshold",
    min_value=0.01,
    max_value=0.99,
    value=0.50,
    step=0.01,
    help=(
        "Transactions with fraud probability at or above this value "
        "are flagged. Choose the threshold based on validation results."
    ),
)

st.sidebar.caption(
    "Risk categories are descriptive probability bands. "
    "The alert threshold is configurable."
)


# =========================================================
# HEADER
# =========================================================

st.title("🔍 Fraud Detection & Transaction Risk Analytics")
st.write(
    "Explore transaction patterns, review model performance, "
    "score transaction risk, and explain predictions with SHAP."
)

overview_tab, risk_tab, predict_tab, shap_tab = st.tabs(
    [
        "📊 Analytics",
        "🚦 Risk Scoring",
        "🧪 Predictions",
        "🧠 SHAP Explanations",
    ]
)


# =========================================================
# TAB 1: ANALYTICS
# =========================================================

with overview_tab:
    st.subheader("Transaction Analytics")

    analytics = load_analytics_data(str(DATA_PATH))

    if analytics is None:
        st.warning(
            "data/creditcard.csv was not found. Dataset-wide charts "
            "are unavailable, but the prepared test data can still be used."
        )
    elif "Class" not in analytics.columns:
        st.warning("The source dataset has no Class column.")
    else:
        total = len(analytics)
        frauds = int((analytics["Class"] == 1).sum())
        normals = int((analytics["Class"] == 0).sum())
        rate = 100 * frauds / total if total else 0

        a, b, c, d = st.columns(4)
        a.metric("Transactions", f"{total:,}")
        b.metric("Fraud cases", f"{frauds:,}")
        c.metric("Normal cases", f"{normals:,}")
        d.metric("Fraud rate", f"{rate:.3f}%")

        left, right = st.columns(2)

        with left:
            counts = analytics["Class"].value_counts().reindex(
                [0, 1], fill_value=0
            )
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(["Normal", "Fraud"], counts.values)
            ax.set_title("Transaction Class Distribution")
            ax.set_ylabel("Transactions")
            st.pyplot(fig, width="stretch")
            plt.close(fig)

        with right:
            if "Amount" in analytics.columns:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.hist(
                    analytics["Amount"].dropna().sample(
                        n=min(20000, analytics["Amount"].notna().sum()),
                        random_state=42,
                    ),
                    bins=40,
                )
                ax.set_title("Transaction Amount Distribution")
                ax.set_xlabel("Amount")
                ax.set_ylabel("Frequency")
                st.pyplot(fig, width="stretch")
                plt.close(fig)

        if "Amount" in analytics.columns:
            st.markdown("**Average transaction amount by class**")
            amount_summary = (
                analytics.groupby("Class")["Amount"]
                .agg(["count", "mean", "median", "max"])
                .rename(index={0: "Normal", 1: "Fraud"})
            )
            st.dataframe(
                amount_summary.round(2),
                width="stretch",
            )

    st.divider()
    st.subheader("Model Comparison")

    if COMPARISON_PATH.exists():
        comparison = pd.read_csv(COMPARISON_PATH)
        st.dataframe(comparison, width="stretch", hide_index=True)
    else:
        st.info(
            "Model comparison report not found. Run "
            "`python compare_models.py` to generate it."
        )

    if METRICS_PATH.exists():
        try:
            with open(METRICS_PATH, encoding="utf-8") as file:
                metrics = json.load(file)
            with st.expander("Random Forest evaluation metrics"):
                st.json(metrics)
        except (OSError, json.JSONDecodeError) as exc:
            st.warning(f"Could not read model metrics: {exc}")


# =========================================================
# TAB 2: RISK SCORING
# =========================================================

with risk_tab:
    st.subheader("Test-Set Risk Scoring")
    st.write(
        "This section scores the held-out test transactions. "
        "The displayed alert threshold is adjustable."
    )

    if st.button("Calculate test-set risk scores"):
        try:
            probabilities = fraud_probability(X_test_df)

            risk_report = X_test_df.copy()
            risk_report["Actual Class"] = y_test
            risk_report["Fraud Probability"] = probabilities
            risk_report["Risk Level"] = [
                risk_category(float(p)) for p in probabilities
            ]
            risk_report["Flagged"] = probabilities >= threshold

            critical = int((risk_report["Risk Level"] == "Critical").sum())
            high = int((risk_report["Risk Level"] == "High").sum())
            medium = int((risk_report["Risk Level"] == "Medium").sum())
            low = int((risk_report["Risk Level"] == "Low").sum())
            flagged = int(risk_report["Flagged"].sum())

            a, b, c, d, e = st.columns(5)
            a.metric("Flagged", f"{flagged:,}")
            b.metric("Critical", f"{critical:,}")
            c.metric("High", f"{high:,}")
            d.metric("Medium", f"{medium:,}")
            e.metric("Low", f"{low:,}")

            st.dataframe(
                risk_report.sort_values(
                    "Fraud Probability", ascending=False
                ).head(100),
                width="stretch",
                hide_index=True,
            )

            st.download_button(
                "Download risk-scoring report",
                data=risk_report.to_csv(index=False).encode("utf-8"),
                file_name="fraud_risk_report.csv",
                mime="text/csv",
            )

        except Exception as exc:
            st.error("Risk scoring failed.")
            st.exception(exc)


# =========================================================
# TAB 3: CSV PREDICTIONS
# =========================================================

with predict_tab:
    st.subheader("Predict Fraud Risk from a CSV")
    st.write(
        "Upload a CSV with the same numeric feature columns used to train "
        "the Random Forest model. The Class label is optional."
    )

    st.caption(
        "For this credit-card dataset, the model usually expects Time, "
        "V1–V28, and Amount. Use the exact feature list shown below."
    )

    with st.expander("Required input columns"):
        st.code(", ".join(feature_names))

    uploaded_file = st.file_uploader(
        "Upload transaction CSV",
        type=["csv"],
        key="prediction_upload",
    )

    if uploaded_file is not None:
        try:
            uploaded, X_upload = prepare_upload(uploaded_file)

            st.write(f"Transactions loaded: {len(X_upload):,}")

            if st.button("Predict transaction risk", type="primary"):
                probabilities = fraud_probability(X_upload)

                results = uploaded.copy()
                results["Fraud Probability"] = probabilities
                results["Risk Level"] = [
                    risk_category(float(p)) for p in probabilities
                ]
                results["Prediction"] = np.where(
                    probabilities >= threshold,
                    "Fraud Alert",
                    "Not Flagged",
                )

                flagged = int(
                    (probabilities >= threshold).sum()
                )

                a, b, c = st.columns(3)
                a.metric("Transactions", f"{len(results):,}")
                b.metric("Flagged", f"{flagged:,}")
                c.metric(
                    "Flagged rate",
                    f"{100 * flagged / len(results):.2f}%"
                    if len(results) else "0%",
                )

                st.dataframe(
                    results.sort_values(
                        "Fraud Probability",
                        ascending=False,
                    ),
                    width="stretch",
                    hide_index=True,
                )

                st.download_button(
                    "Download prediction results",
                    data=results.to_csv(index=False).encode("utf-8"),
                    file_name="fraud_predictions.csv",
                    mime="text/csv",
                )

                st.session_state["last_prediction_X"] = X_upload.copy()
                st.session_state["last_prediction_results"] = results.copy()

        except Exception as exc:
            st.error("Could not process the uploaded CSV.")
            st.exception(exc)


# =========================================================
# TAB 4: SHAP EXPLANATIONS
# =========================================================

with shap_tab:
    st.subheader("Explainable AI with SHAP")

    background_source = (
        X_train_df
        if X_train_df is not None and not X_train_df.empty
        else X_test_df
    )
    background = background_source.sample(
        n=min(100, len(background_source)),
        random_state=42,
    )

    st.write(
        "SHAP explains how model features influence predictions. "
        "It does not establish causation or prove fraud."
    )

    shap_source = st.radio(
        "Choose transactions to explain",
        ["Test dataset", "Last uploaded prediction"],
        horizontal=True,
    )

    if (
        shap_source == "Last uploaded prediction"
        and "last_prediction_X" in st.session_state
    ):
        explanation_data = st.session_state["last_prediction_X"]
    else:
        explanation_data = X_test_df

        if shap_source == "Last uploaded prediction":
            st.info(
                "No uploaded predictions are available yet. "
                "Using test transactions instead."
            )

    if len(explanation_data) == 0:
        st.warning("No transactions are available for explanation.")
    else:
        index = st.number_input(
            "Transaction index",
            min_value=0,
            max_value=len(explanation_data) - 1,
            value=0,
            step=1,
            key="shap_transaction_index",
        )

        if st.button("Explain selected transaction"):
            try:
                import shap

                background_data = background

                with st.spinner("Generating SHAP explanation..."):
                    explainer = create_shap_explainer(background_data)
                    transaction = explanation_data.iloc[[int(index)]]

                    shap_result = explainer.shap_values(
                        transaction,
                        check_additivity=False,
                    )

                    values = get_shap_values(
                        shap_result,
                        len(feature_names),
                    )[0]

                    base_value = get_shap_base_value(explainer)

                    explanation = shap.Explanation(
                        values=values,
                        base_values=base_value,
                        data=transaction.iloc[0].to_numpy(),
                        feature_names=feature_names,
                    )

                    st.markdown("**Individual feature contributions**")
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

                    st.dataframe(
                        contributions.head(15).round(6),
                        width="stretch",
                        hide_index=True,
                    )

                    st.download_button(
                        "Download SHAP explanation",
                        data=contributions.to_csv(
                            index=False
                        ).encode("utf-8"),
                        file_name="individual_shap_explanation.csv",
                        mime="text/csv",
                    )

            except Exception as exc:
                st.error("Individual SHAP explanation failed.")
                st.exception(exc)

    st.divider()
    st.markdown("**Global feature importance**")

    if st.button("Generate global SHAP importance"):
        try:
            import shap

            sample = X_test_df.sample(
                n=min(100, len(X_test_df)),
                random_state=42,
            )

            with st.spinner("Calculating global feature importance..."):
                explainer = create_shap_explainer(background)
                result = explainer.shap_values(
                    sample,
                    check_additivity=False,
                )

                values = get_shap_values(
                    result,
                    len(feature_names),
                )

                importance = np.abs(values).mean(axis=0)

                importance_df = pd.DataFrame({
                    "Feature": feature_names,
                    "Mean Absolute SHAP": importance,
                }).sort_values(
                    "Mean Absolute SHAP",
                    ascending=False,
                )

                fig, ax = plt.subplots(figsize=(9, 6))
                ax.barh(
                    importance_df.head(15)["Feature"][::-1],
                    importance_df.head(15)["Mean Absolute SHAP"][::-1],
                )
                ax.set_title("Global SHAP Feature Importance")
                ax.set_xlabel("Mean absolute SHAP value")
                fig.tight_layout()
                st.pyplot(fig, width="stretch")
                plt.close(fig)

                st.dataframe(
                    importance_df.head(15).round(6),
                    width="stretch",
                    hide_index=True,
                )

                st.download_button(
                    "Download global feature importance",
                    data=importance_df.to_csv(
                        index=False
                    ).encode("utf-8"),
                    file_name="global_shap_importance.csv",
                    mime="text/csv",
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
