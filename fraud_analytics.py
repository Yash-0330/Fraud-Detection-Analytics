
from pathlib import Path
import json

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="Fraud Risk Analytics",
    page_icon="🔎",
    layout="wide",
)

st.title("Fraud Detection & Transaction Risk Analytics")
st.caption(
    "Interactive portfolio dashboard | Python | Pandas | "
    "Scikit-learn | Plotly | Streamlit"
)

# =====================================================
# 2. CHECK REQUIRED FILES
# =====================================================
required_files = [
    "data/creditcard.csv",
    "models/prepared_data.joblib",
    "models/logistic_regression.joblib",
    "models/random_forest.joblib",
    "models/logistic_metrics.json",
    "models/random_forest_metrics.json",
]

missing_files = [
    file for file in required_files
    if not Path(file).exists()
]

if missing_files:
    st.error("Some required project files are missing:")
    for file in missing_files:
        st.write(f"- {file}")
    st.info(
        "Run your data preparation and model training scripts "
        "before starting the dashboard."
    )
    st.stop()

# =====================================================
# 3. LOAD DATA AND MODELS
# =====================================================
@st.cache_data
def load_dataset():
    # Read only the columns needed for analytics.
    # The original dataset remains unchanged.
    return pd.read_csv(
        "data/creditcard.csv",
        usecols=["Time", "Amount", "Class"],
    )


@st.cache_resource
def load_models():
    prepared = joblib.load("models/prepared_data.joblib")
    logistic = joblib.load("models/logistic_regression.joblib")
    forest = joblib.load("models/random_forest.joblib")
    return prepared, logistic, forest


@st.cache_data
def load_metrics():
    with open("models/logistic_metrics.json") as file:
        logistic_metrics = json.load(file)

    with open("models/random_forest_metrics.json") as file:
        forest_metrics = json.load(file)

    return logistic_metrics, forest_metrics


with st.spinner("Loading dashboard data and models..."):
    df = load_dataset()
    prepared, logistic_model, forest_model = load_models()
    logistic_metrics, forest_metrics = load_metrics()

feature_names = list(prepared["feature_names"])

# =====================================================
# 4. SIDEBAR FILTERS
# =====================================================
st.sidebar.title("Dashboard Controls")
st.sidebar.header("Transaction Filters")

class_options = st.sidebar.multiselect(
    "Transaction class",
    options=[0, 1],
    default=[0, 1],
    format_func=lambda value: (
        "Genuine" if value == 0 else "Fraud"
    ),
)

amount_min = float(df["Amount"].min())
amount_max = float(df["Amount"].max())

amount_range = st.sidebar.slider(
    "Transaction amount range",
    min_value=amount_min,
    max_value=amount_max,
    value=(amount_min, amount_max),
)

time_min = float(df["Time"].min())
time_max = float(df["Time"].max())

time_range = st.sidebar.slider(
    "Transaction time range (seconds)",
    min_value=time_min,
    max_value=time_max,
    value=(time_min, time_max),
)

# Apply filters to all available rows.
filtered = df.loc[
    df["Class"].isin(class_options)
    & df["Amount"].between(
        amount_range[0], amount_range[1]
    )
    & df["Time"].between(
        time_range[0], time_range[1]
    )
]

# =====================================================
# 5. TRANSACTION OVERVIEW
# =====================================================
st.header("1. Transaction Overview")

total_transactions = len(filtered)
fraud_count = int(filtered["Class"].sum())
genuine_count = total_transactions - fraud_count

fraud_rate = (
    fraud_count / total_transactions * 100
    if total_transactions else 0.0
)

total_amount = float(filtered["Amount"].sum())
average_amount = (
    float(filtered["Amount"].mean())
    if total_transactions else 0.0
)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Filtered Transactions",
    f"{total_transactions:,}",
)
col2.metric(
    "Genuine Transactions",
    f"{genuine_count:,}",
)
col3.metric(
    "Fraud Transactions",
    f"{fraud_count:,}",
)
col4.metric(
    "Fraud Rate",
    f"{fraud_rate:.3f}%",
)

col5, col6 = st.columns(2)
col5.metric("Total Transaction Amount", f"{total_amount:,.2f}")
col6.metric("Average Transaction Amount", f"{average_amount:,.2f}")

# =====================================================
# 6. INTERACTIVE CHARTS
# =====================================================
st.header("2. Interactive Charts")

if filtered.empty:
    st.warning(
        "No transactions match your filters. "
        "Change the filters in the sidebar."
    )
else:
    chart_left, chart_right = st.columns(2)

    with chart_left:
        class_summary = pd.DataFrame({
            "Transaction Class": ["Genuine", "Fraud"],
            "Count": [genuine_count, fraud_count],
        })

        fig_class = px.bar(
            class_summary,
            x="Transaction Class",
            y="Count",
            color="Transaction Class",
            title="Genuine vs Fraud Transactions",
            text="Count",
        )

        fig_class.update_layout(
            showlegend=False,
            height=380,
        )

        st.plotly_chart(
            fig_class,
            use_container_width=True,
        )

    with chart_right:
        # Sample rows only for the visualization.
        chart_data = filtered.sample(
            n=min(len(filtered), 20000),
            random_state=42,
        )

        fig_amount = px.histogram(
            chart_data,
            x="Amount",
            color="Class",
            nbins=25,
            title="Transaction Amount Distribution",
            labels={
                "Class": "Class (0 = Genuine, 1 = Fraud)",
                "Amount": "Transaction Amount",
            },
            color_discrete_sequence=["#2563eb", "#ef4444"],
        )

        fig_amount.update_layout(height=380)

        st.plotly_chart(
            fig_amount,
            use_container_width=True,
        )

    st.subheader("Fraud Rate by Transaction Amount")

    amount_bins = pd.cut(
        filtered["Amount"],
        bins=[
            -0.01, 50, 100, 500, 2000,
            float("inf"),
        ],
        labels=[
            "0–50", "50–100", "100–500",
            "500–2000", "2000+",
        ],
    )

    band_data = pd.DataFrame({
        "Amount Band": amount_bins,
        "Class": filtered["Class"],
    })

    band_summary = (
        band_data.groupby(
            "Amount Band",
            observed=False,
        )
        .agg(
            Transactions=("Class", "size"),
            FraudCount=("Class", "sum"),
            FraudRate=("Class", "mean"),
        )
        .reset_index()
    )

    band_summary["Fraud Rate (%)"] = (
        band_summary["FraudRate"] * 100
    )

    fig_band = px.bar(
        band_summary,
        x="Amount Band",
        y="Fraud Rate (%)",
        hover_data=["Transactions", "FraudCount"],
        title="Fraud Rate by Amount Band",
        labels={
            "Amount Band": "Transaction Amount Range",
        },
    )

    fig_band.update_layout(height=380)

    st.plotly_chart(
        fig_band,
        use_container_width=True,
    )

    st.subheader("Filtered Transaction Records")

    st.dataframe(
        filtered.head(100),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Showing up to 100 rows. KPIs and report exports "
        "use all matching transactions."
    )

# =====================================================
# 7. DOWNLOAD REPORTS
# =====================================================
st.header("3. Download Reports")

filtered_csv = filtered.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    "Download Filtered Transactions CSV",
    data=filtered_csv,
    file_name="filtered_transactions.csv",
    mime="text/csv",
)

kpi_summary = pd.DataFrame([{
    "Filtered Transactions": total_transactions,
    "Genuine Transactions": genuine_count,
    "Fraud Transactions": fraud_count,
    "Fraud Rate Percent": fraud_rate,
    "Total Transaction Amount": total_amount,
    "Average Transaction Amount": average_amount,
}])

st.download_button(
    "Download KPI Summary CSV",
    data=kpi_summary.to_csv(index=False).encode("utf-8"),
    file_name="fraud_kpi_summary.csv",
    mime="text/csv",
)

# =====================================================
# 8. MODEL PERFORMANCE COMPARISON
# =====================================================
st.header("4. Model Performance Comparison")

comparison = pd.DataFrame([
    {
        "Model": "Logistic Regression",
        "Precision": logistic_metrics["precision"],
        "Recall": logistic_metrics["recall"],
        "PR-AUC": logistic_metrics["pr_auc"],
        "Average Precision": logistic_metrics["average_precision"],
    },
    {
        "Model": "Random Forest",
        "Precision": forest_metrics["precision"],
        "Recall": forest_metrics["recall"],
        "PR-AUC": forest_metrics["pr_auc"],
        "Average Precision": forest_metrics["average_precision"],
    },
])

st.dataframe(
    comparison.style.format({
        "Precision": "{:.4f}",
        "Recall": "{:.4f}",
        "PR-AUC": "{:.4f}",
        "Average Precision": "{:.4f}",
    }),
    use_container_width=True,
    hide_index=True,
)

fig_metrics = px.bar(
    comparison,
    x="Model",
    y=[
        "Precision",
        "Recall",
        "PR-AUC",
        "Average Precision",
    ],
    barmode="group",
    title="Comparison of Fraud Detection Models",
)

fig_metrics.update_layout(height=420)

st.plotly_chart(
    fig_metrics,
    use_container_width=True,
)

st.download_button(
    "Download Model Comparison CSV",
    data=comparison.to_csv(index=False).encode("utf-8"),
    file_name="model_comparison.csv",
    mime="text/csv",
)

# =====================================================
# 9. CSV UPLOAD AND FRAUD PREDICTIONS
# =====================================================
st.header("5. Predict Fraud Risk from a CSV")

st.write(
    "Upload transaction data to estimate fraud probabilities "
    "using one of your trained models."
)

st.info(
    "Your file must contain Time, V1–V28, and Amount. "
    "The V1–V28 features are anonymized numerical variables "
    "from the original dataset, not ordinary card details."
)

model_choice = st.selectbox(
    "Select prediction model",
    ["Random Forest", "Logistic Regression"],
)

threshold = st.slider(
    "Flag transactions at or above this fraud probability",
    min_value=0.05,
    max_value=0.95,
    value=0.50,
    step=0.05,
)

uploaded_file = st.file_uploader(
    "Upload compatible transaction CSV",
    type=["csv"],
)

if uploaded_file is not None:
    try:
        upload_df = pd.read_csv(uploaded_file)

        if upload_df.empty:
            st.warning("The uploaded CSV has no rows.")
        else:
            missing_features = [
                name for name in feature_names
                if name not in upload_df.columns
            ]

            if missing_features:
                st.error(
                    "Missing required columns: "
                    + ", ".join(missing_features)
                )

            elif len(upload_df) > 100000:
                st.error(
                    "This demo accepts up to 100,000 rows per upload. "
                    "Please split your file into smaller CSV files."
                )

            else:
                prediction_input = upload_df[
                    feature_names
                ].copy()

                # Validate numeric inputs.
                for column in feature_names:
                    prediction_input[column] = pd.to_numeric(
                        prediction_input[column],
                        errors="coerce",
                    )

                numeric_values = prediction_input.to_numpy()

                import numpy as np

                if not np.isfinite(numeric_values).all():
                    st.error(
                        "Required features contain missing, "
                        "non-numeric, or infinite values. "
                        "Clean the CSV and upload it again."
                    )
                else:
                    if model_choice == "Logistic Regression":
                        prediction_model = logistic_model
                        model_input = prepared["scaler"].transform(
                            prediction_input
                        )
                    else:
                        prediction_model = forest_model
                        model_input = prediction_input

                    with st.spinner(
                        "Calculating transaction risk..."
                    ):
                        probabilities = (
                            prediction_model.predict_proba(
                                model_input
                            )[:, 1]
                        )

                    results = upload_df.copy()

                    results["Fraud Probability"] = probabilities
                    results["Predicted Class"] = (
                        probabilities >= threshold
                    ).astype(int)

                    results["Prediction"] = (
                        results["Predicted Class"]
                        .map({
                            0: "Not flagged",
                            1: "Flagged for review",
                        })
                    )

                    st.subheader("Prediction Summary")

                    pred_col1, pred_col2, pred_col3 = (
                        st.columns(3)
                    )

                    pred_col1.metric(
                        "Rows Scored",
                        f"{len(results):,}",
                    )

                    flagged_count = int(
                        results["Predicted Class"].sum()
                    )

                    pred_col2.metric(
                        "Flagged for Review",
                        f"{flagged_count:,}",
                    )

                    pred_col3.metric(
                        "Decision Threshold",
                        f"{threshold:.0%}",
                    )

                    st.dataframe(
                        results.head(100),
                        use_container_width=True,
                        hide_index=True,
                    )

                    st.download_button(
                        "Download Fraud Predictions CSV",
                        data=results.to_csv(
                            index=False
                        ).encode("utf-8"),
                        file_name="fraud_predictions.csv",
                        mime="text/csv",
                    )

                    st.caption(
                        "A flagged transaction is a model-generated "
                        "risk signal, not proof of fraud. This project "
                        "is a demonstration, not a production banking system."
                    )

    except (
        ValueError,
        pd.errors.ParserError,
        UnicodeDecodeError,
    ) as error:
        st.error(f"Could not process the uploaded CSV: {error}")

# =====================================================
# 10. FOOTER
# =====================================================
st.divider()

st.caption(
    "Fraud Detection & Transaction Risk Analytics | "
    "Portfolio project | Educational use only"
)
