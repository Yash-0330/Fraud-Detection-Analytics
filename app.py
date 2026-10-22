from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# 1. CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="FraudShield | Risk Intelligence",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

DATA_PATH = DATA_DIR / "creditcard.csv"
PREPARED_PATH = MODELS_DIR / "prepared_data.joblib"
LOGISTIC_PATH = MODELS_DIR / "logistic_regression.joblib"
FOREST_PATH = MODELS_DIR / "random_forest.joblib"
LOGISTIC_METRICS_PATH = MODELS_DIR / "logistic_metrics.json"
FOREST_METRICS_PATH = MODELS_DIR / "random_forest_metrics.json"

BG = "#0b1220"
PANEL = "#131f33"
CYAN = "#22d3ee"
BLUE = "#3b82f6"
GREEN = "#34d399"
ORANGE = "#fb923c"
RED = "#f87171"
MUTED = "#94a3b8"


# ============================================================
# 2. PROFESSIONAL UI THEME
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    .stApp {
        background: linear-gradient(135deg, #0b1220 0%, #101b2d 60%, #0b1220 100%);
        color: #e5edf8;
        font-family: 'Inter', sans-serif;
    }

    .block-container {
        max-width: 1550px;
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
    }

    section[data-testid="stSidebar"] {
        background: #091323;
        border-right: 1px solid #26364d;
    }

    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #eaf4ff !important;
    }

    h1, h2, h3 {
        color: #f8fafc !important;
        letter-spacing: -0.4px;
        font-weight: 750 !important;
    }

    p, label, .stMarkdown {
        color: #d4deed;
    }

    .hero {
        background: linear-gradient(115deg, #132a46 0%, #172554 55%, #164e63 100%);
        border: 1px solid #2c4a68;
        border-radius: 22px;
        padding: 30px 32px;
        margin-bottom: 24px;
        box-shadow: 0 12px 35px rgba(0,0,0,.20);
    }

    .hero-kicker {
        color: #67e8f9;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 2.5px;
        text-transform: uppercase;
    }

    .hero-title {
        color: #ffffff;
        font-size: clamp(27px, 3vw, 40px);
        line-height: 1.2;
        font-weight: 800;
        margin: 12px 0 10px 0;
    }

    .hero-subtitle {
        color: #c5d6e9;
        font-size: 14px;
        line-height: 1.8;
        margin: 0;
    }

    .section-label {
        color: #67e8f9;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 1.8px;
        text-transform: uppercase;
        margin: 8px 0;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #17263c, #111c2e);
        border: 1px solid #2b405c;
        border-radius: 16px;
        padding: 19px 20px;
        min-height: 120px;
        box-shadow: 0 6px 18px rgba(0,0,0,.14);
        transition: border-color .2s ease, transform .2s ease;
    }

    div[data-testid="stMetric"]:hover {
        border-color: #22d3ee;
        transform: translateY(-2px);
    }

    div[data-testid="stMetricLabel"] {
        color: #a9bad0 !important;
        font-size: 12px !important;
    }

    div[data-testid="stMetricValue"] {
        color: #f8fafc !important;
        font-weight: 800 !important;
        font-size: 26px !important;
    }

    div[data-testid="stMetricDelta"] {
        font-size: 12px;
    }

    button[data-baseweb="tab"] {
        color: #9fb1c8;
        font-weight: 700;
        padding: 12px 16px;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #67e8f9 !important;
        border-bottom-color: #22d3ee !important;
    }

    .stButton > button,
    .stDownloadButton > button {
        background: linear-gradient(100deg, #0891b2, #2563eb);
        color: #ffffff !important;
        border: 1px solid #2873b7;
        border-radius: 10px;
        font-weight: 700;
        min-height: 42px;
        transition: all .2s ease;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        border-color: #67e8f9;
        filter: brightness(1.12);
    }

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        background-color: #111e32;
        border-color: #334963;
        border-radius: 9px;
    }

    div[data-testid="stPlotlyChart"] {
        background: #111d30;
        border: 1px solid #263b55;
        border-radius: 14px;
        padding: 8px;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid #2b405c;
        border-radius: 12px;
        overflow: hidden;
    }

    [data-testid="stFileUploader"] {
        background: #132238;
        border: 1px dashed #3b82a0;
        border-radius: 14px;
        padding: 12px;
    }

    div[data-testid="stAlert"] {
        border-radius: 12px;
    }

    hr {
        border-color: #283a53;
    }

    .footer {
        color: #8194ad;
        text-align: center;
        font-size: 11px;
        line-height: 1.8;
        padding: 24px 0 8px 0;
        margin-top: 25px;
        border-top: 1px solid #26364d;
    }

    @media (max-width: 768px) {
        .block-container {
            padding-left: .8rem;
            padding-right: .8rem;
        }

        .hero {
            padding: 22px 18px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 3. HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-kicker">Financial Intelligence · ML Analytics</div>
        <div class="hero-title">🔐 FraudShield Analytics</div>
        <p class="hero-subtitle">
            Transaction intelligence, machine learning risk detection,
            model evaluation and explainable AI — in one workspace.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 4. DATA AND MODEL LOADING
# ============================================================

@st.cache_data(show_spinner="Loading transaction data...")
def load_dataset(path_string):
    path = Path(path_string)
    if not path.exists():
        return None

    return pd.read_csv(
        path,
        usecols=lambda col: col in {"Time", "Amount", "Class"},
    )


@st.cache_resource(show_spinner="Loading trained models...")
def load_models():
    required = [PREPARED_PATH, LOGISTIC_PATH, FOREST_PATH]
    missing = [str(p) for p in required if not p.exists()]

    if missing:
        raise FileNotFoundError(
            "Missing required project files:\n" + "\n".join(missing)
        )

    prepared = joblib.load(PREPARED_PATH)
    logistic = joblib.load(LOGISTIC_PATH)
    forest = joblib.load(FOREST_PATH)

    return prepared, logistic, forest


@st.cache_data
def load_metrics(path_string):
    path = Path(path_string)
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def to_dataframe(values, columns):
    if values is None:
        return None

    if isinstance(values, pd.DataFrame):
        missing = [c for c in columns if c not in values.columns]
        if missing:
            raise ValueError(f"Missing saved features: {missing}")
        return values.loc[:, columns].copy()

    array = np.asarray(values)

    if array.ndim != 2 or array.shape[1] != len(columns):
        raise ValueError(
            f"Data shape {array.shape} does not match "
            f"{len(columns)} expected features."
        )

    return pd.DataFrame(array, columns=columns)


try:
    prepared, logistic_model, forest_model = load_models()

    if "X_test" not in prepared or "y_test" not in prepared:
        raise ValueError(
            "prepared_data.joblib must contain X_test and y_test."
        )

    saved_names = prepared.get("feature_names")

    if isinstance(prepared["X_test"], pd.DataFrame):
        feature_names = list(prepared["X_test"].columns)
    elif saved_names is not None:
        feature_names = list(saved_names)
    else:
        raise ValueError(
            "Feature names are missing from prepared_data.joblib."
        )

    X_test_df = to_dataframe(prepared["X_test"], feature_names)
    y_test = np.asarray(prepared["y_test"]).reshape(-1)
    X_train_df = to_dataframe(prepared.get("X_train"), feature_names)

    if len(X_test_df) != len(y_test):
        raise ValueError("X_test and y_test row counts do not match.")

    if len(X_test_df) == 0:
        raise ValueError("The saved test dataset is empty.")

    if hasattr(forest_model, "n_features_in_"):
        if forest_model.n_features_in_ != len(feature_names):
            raise ValueError(
                "Random Forest feature count does not match the saved data."
            )

    scaler = prepared.get("scaler")
    df = load_dataset(str(DATA_PATH))

    logistic_metrics = load_metrics(str(LOGISTIC_METRICS_PATH))
    forest_metrics = load_metrics(str(FOREST_METRICS_PATH))

except Exception as exc:
    st.error("Unable to load the project files.")
    st.exception(exc)
    st.stop()


# ============================================================
# 5. HELPER FUNCTIONS
# ============================================================

def fraud_probability(model, X):
    if not hasattr(model, "predict_proba"):
        raise ValueError("The selected model has no predict_proba method.")

    probabilities = model.predict_proba(X)
    classes = list(model.classes_)

    if 1 not in classes:
        raise ValueError("The trained model does not contain fraud class 1.")

    return probabilities[:, classes.index(1)]


def risk_level(probability):
    if probability >= 0.80:
        return "Critical"
    if probability >= 0.50:
        return "High"
    if probability >= 0.20:
        return "Medium"
    return "Low"


def validate_csv(uploaded_file):
    if uploaded_file is None:
        raise ValueError("Please upload a CSV file.")

    # Streamlit may reuse the same UploadedFile object across reruns.
    # Always rewind so the CSV is parsed from the beginning every time.
    try:
        uploaded_file.seek(0)
    except Exception:
        pass

    df = pd.read_csv(uploaded_file)

    if df.empty:
        raise ValueError("The uploaded CSV has no rows.")

    # Remove accidental spaces around column names
    df.columns = df.columns.astype(str).str.strip()

    required_features = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

    missing = [col for col in required_features if col not in df.columns]
    if missing:
        raise ValueError(
        "Missing required model features: " + ", ".join(missing)
        + ". "
        f"Columns found: {df.columns.tolist()}. "
        "The model needs Time, V1–V28, and Amount. "
        "Files that only contain Time/Amount/Class "
        "(such as a filtered download from the Overview tab of the "
        "previous app version) cannot be scored — re-download after "
        "updating the app, or use the original creditcard.csv."
    )

    prediction_input = df[required_features].copy()

    for col in required_features:
        prediction_input[col] = pd.to_numeric(
            prediction_input[col], errors="coerce"
        )

    if not np.isfinite(prediction_input.to_numpy(dtype=float)).all():
        raise ValueError(
            "The CSV contains missing, non-numeric, or infinite values."
        )

    # Ensure correct feature ordering for the model
    if list(feature_names) != required_features:
        prediction_input = prediction_input[feature_names]

    return df, prediction_input


def get_metric(metrics, key):
    value = metrics.get(key)
    if value is None:
        return np.nan

    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def style_chart(fig, height=390):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial, sans-serif", color="#dbeafe", size=12),
        title_font=dict(size=17, color="#f8fafc"),
        height=height,
        margin=dict(l=20, r=20, t=65, b=35),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
        ),
    )
    fig.update_xaxes(gridcolor="#263b55", zerolinecolor="#334963")
    fig.update_yaxes(gridcolor="#263b55", zerolinecolor="#334963")
    return fig


def section_heading(kicker, heading, description=None):
    st.markdown(
        f'<div class="section-label">{kicker}</div>'
        f'<h2 style="margin-top:0">{heading}</h2>',
        unsafe_allow_html=True,
    )
    if description:
        st.caption(description)


def risk_distribution_chart(results):
    counts = (
        results["Risk Level"]
        .value_counts()
        .reindex(["Critical", "High", "Medium", "Low"], fill_value=0)
        .rename_axis("Risk Level")
        .reset_index(name="Transactions")
    )

    fig = px.bar(
        counts,
        x="Risk Level",
        y="Transactions",
        color="Risk Level",
        text="Transactions",
        color_discrete_map={
            "Critical": "#ef4444",
            "High": "#f97316",
            "Medium": "#eab308",
            "Low": "#22c55e",
        },
        title="Transaction Risk Distribution",
    )

    return style_chart(fig, 320)


# ============================================================
# 6. SIDEBAR
# ============================================================

st.sidebar.markdown(
    """
    <div style="padding:8px 0 16px 0">
        <div style="font-size:11px;color:#67e8f9;
                    font-weight:800;letter-spacing:2px">
            FRAUDSHIELD
        </div>
        <div style="font-size:21px;color:#f8fafc;
                    font-weight:800;margin-top:5px">
            Control Center
        </div>
        <div style="font-size:12px;color:#94a3b8;margin-top:5px">
            Transaction risk workspace
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.divider()
st.sidebar.subheader("⚙️ Detection Settings")

threshold = st.sidebar.slider(
    "Fraud alert threshold",
    min_value=0.01,
    max_value=0.99,
    value=0.50,
    step=0.01,
    help="Flag a transaction when its estimated fraud probability meets this threshold.",
)

st.sidebar.caption(
    "The threshold and risk bands are illustrative. "
    "Select operational thresholds using validation data."
)

if df is not None:
    st.sidebar.divider()
    st.sidebar.subheader("🔍 Analytics Filters")

    class_options = st.sidebar.multiselect(
        "Transaction class",
        options=[0, 1],
        default=[0, 1],
        format_func=lambda v: "Genuine" if v == 0 else "Fraud",
    )

    amount_min = float(df["Amount"].min())
    amount_max = float(df["Amount"].max())

    if amount_min == amount_max:
        amount_range = (amount_min, amount_max)
    else:
        amount_range = st.sidebar.slider(
            "Amount range",
            min_value=amount_min,
            max_value=amount_max,
            value=(amount_min, amount_max),
        )

    time_min = float(df["Time"].min())
    time_max = float(df["Time"].max())

    if time_min == time_max:
        time_range = (time_min, time_max)
    else:
        time_range = st.sidebar.slider(
            "Time range (seconds)",
            min_value=time_min,
            max_value=time_max,
            value=(time_min, time_max),
        )

    filtered = df[
        df["Class"].isin(class_options)
        & df["Amount"].between(*amount_range)
        & df["Time"].between(*time_range)
    ].copy()

else:
    filtered = None
    st.sidebar.warning("Dataset not found: data/creditcard.csv")


st.sidebar.divider()
st.sidebar.markdown(
    """
    <div style="font-size:11px;color:#8295ad;line-height:1.8">
        <b style="color:#cbd5e1">TECH STACK</b><br>
        Python · Pandas · Scikit-learn<br>
        Plotly · Streamlit · SHAP
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 7. NAVIGATION
# ============================================================

overview_tab, performance_tab, risk_tab, prediction_tab, shap_tab = st.tabs(
    [
        "📊  Overview",
        "📈  Model Lab",
        "🚦  Risk Scoring",
        "🧪  Predictions",
        "🧠  Explain AI",
    ]
)


# ============================================================
# TAB 1 — ANALYTICS OVERVIEW
# ============================================================

with overview_tab:
    section_heading(
        "LIVE DATA EXPLORER",
        "Transaction Overview",
        "Explore transaction patterns and the distribution of fraud in the selected dataset.",
    )

    if filtered is None:
        st.warning("Place creditcard.csv inside your project's data folder.")

    elif filtered.empty:
        st.warning("No transactions match these filters. Adjust the sidebar.")

    else:
        total = len(filtered)
        fraud_count = int((filtered["Class"] == 1).sum())
        genuine_count = int((filtered["Class"] == 0).sum())
        fraud_rate = 100 * fraud_count / total if total else 0
        total_amount = float(filtered["Amount"].sum())
        avg_amount = float(filtered["Amount"].mean())

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Transactions", f"{total:,}")
        c2.metric("Genuine Transactions", f"{genuine_count:,}")
        c3.metric("Fraud Transactions", f"{fraud_count:,}")
        c4.metric("Observed Fraud Rate", f"{fraud_rate:.3f}%")

        c5, c6 = st.columns(2)
        c5.metric("Total Transaction Amount", f"{total_amount:,.2f}")
        c6.metric("Average Transaction Amount", f"{avg_amount:,.2f}")

        st.caption(
            "All KPIs reflect the selected filters. "
            "The dataset's Class column supplies the observed labels."
        )

        st.divider()

        left, right = st.columns(2)

        with left:
            class_summary = pd.DataFrame({
                "Class": ["Genuine", "Fraud"],
                "Transactions": [genuine_count, fraud_count],
            })

            fig = px.pie(
                class_summary,
                names="Class",
                values="Transactions",
                hole=0.68,
                color="Class",
                color_discrete_map={
                    "Genuine": "#22d3ee",
                    "Fraud": "#fb7185",
                },
                title="Transaction Class Composition",
            )
            fig.update_traces(
                textposition="outside",
                textinfo="label+percent",
            )
            st.plotly_chart(style_chart(fig), width="stretch")

        with right:
            fig = px.histogram(
                filtered,
                x="Amount",
                color="Class",
                nbins=45,
                barmode="overlay",
                opacity=0.8,
                color_discrete_map={
                    0: "#22d3ee",
                    1: "#fb7185",
                },
                title="Transaction Amount Distribution",
                labels={
                    "Amount": "Transaction Amount",
                    "Class": "Class (0 = Genuine, 1 = Fraud)",
                },
            )
            st.plotly_chart(style_chart(fig), width="stretch")

        st.divider()
        section_heading(
            "PATTERN ANALYSIS",
            "Fraud Rate by Amount Band",
            "Observed fraud rate for transactions in each amount range.",
        )

        bands = pd.cut(
            filtered["Amount"],
            bins=[-0.01, 50, 100, 500, 2000, float("inf")],
            labels=["0–50", "50–100", "100–500", "500–2000", "2000+"],
        )

        band_summary = (
            filtered.assign(AmountBand=bands)
            .groupby("AmountBand", observed=False)
            .agg(
                Transactions=("Class", "size"),
                FraudCount=("Class", "sum"),
                FraudRate=("Class", "mean"),
            )
            .reset_index()
        )

        band_summary["Fraud Rate (%)"] = band_summary["FraudRate"] * 100

        fig = px.bar(
            band_summary,
            x="AmountBand",
            y="Fraud Rate (%)",
            color="Fraud Rate (%)",
            color_continuous_scale=["#164e63", "#0891b2", "#67e8f9"],
            text_auto=".3f",
            hover_data=["Transactions", "FraudCount"],
            title="Observed Fraud Rate Across Amount Bands",
        )
        st.plotly_chart(style_chart(fig, 360), width="stretch")

        with st.expander("View filtered transaction records"):
            st.dataframe(
                filtered.head(500),
                width="stretch",
                hide_index=True,
            )
            st.caption("Showing a maximum of 500 records for performance.")

        c1, c2 = st.columns(2)

        c1.download_button(
            "⬇ Download Filtered Transactions",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name="filtered_transactions.csv",
            mime="text/csv",
            key="download_filtered",
        )

        kpi_summary = pd.DataFrame([{
            "Transactions": total,
            "Genuine": genuine_count,
            "Fraud": fraud_count,
            "Fraud Rate Percent": fraud_rate,
            "Total Amount": total_amount,
            "Average Amount": avg_amount,
        }])

        c2.download_button(
            "⬇ Download KPI Summary",
            data=kpi_summary.to_csv(index=False).encode("utf-8"),
            file_name="fraud_kpi_summary.csv",
            mime="text/csv",
            key="download_kpi",
        )


# ============================================================
# TAB 2 — MODEL PERFORMANCE
# ============================================================

with performance_tab:
    section_heading(
        "MODEL EVALUATION",
        "Model Performance Lab",
        "Compare classification metrics for your saved machine learning models.",
    )

    comparison = pd.DataFrame([
        {
            "Model": "Logistic Regression",
            "Precision": get_metric(logistic_metrics, "precision"),
            "Recall": get_metric(logistic_metrics, "recall"),
            "F1 Score": get_metric(logistic_metrics, "f1"),
            "PR-AUC": get_metric(logistic_metrics, "pr_auc"),
            "Average Precision": get_metric(
                logistic_metrics, "average_precision"
            ),
        },
        {
            "Model": "Random Forest",
            "Precision": get_metric(forest_metrics, "precision"),
            "Recall": get_metric(forest_metrics, "recall"),
            "F1 Score": get_metric(forest_metrics, "f1"),
            "PR-AUC": get_metric(forest_metrics, "pr_auc"),
            "Average Precision": get_metric(
                forest_metrics, "average_precision"
            ),
        },
    ])

    metric_columns = [
        "Precision", "Recall", "F1 Score", "PR-AUC", "Average Precision"
    ]

    st.dataframe(
        comparison.style.format(
            {column: "{:.4f}" for column in metric_columns},
            na_rep="N/A",
        ),
        width="stretch",
        hide_index=True,
    )

    selected_metric = st.selectbox(
        "Choose a metric",
        metric_columns,
        key="comparison_metric",
    )

    plot_data = comparison.dropna(subset=[selected_metric])

    if not plot_data.empty:
        fig = px.bar(
            plot_data,
            x="Model",
            y=selected_metric,
            color="Model",
            text_auto=".4f",
            color_discrete_sequence=[CYAN, BLUE],
            title=f"{selected_metric} Comparison",
        )
        fig.update_yaxes(rangemode="tozero")
        st.plotly_chart(style_chart(fig, 360), width="stretch")
    else:
        st.info(
            "No saved values are available for this metric. "
            "Check the metrics JSON files."
        )

    st.download_button(
        "⬇ Download Model Comparison",
        data=comparison.to_csv(index=False).encode("utf-8"),
        file_name="model_comparison.csv",
        mime="text/csv",
        key="download_comparison",
    )

    st.warning(
        "Fraud datasets are often highly imbalanced. Accuracy alone can "
        "be misleading. Evaluate precision, recall, F1 and PR-AUC on "
        "held-out data. These metrics are only as reliable as the evaluation "
        "process used to create the JSON files."
    )


# ============================================================
# TAB 3 — RISK SCORING
# ============================================================

with risk_tab:
    section_heading(
        "TRANSACTION INVESTIGATION",
        "Risk Scoring Center",
        "Score the held-out test set with Random Forest and flag transactions for review.",
    )

    st.caption(
        f"Current alert threshold: {threshold:.2f}. "
        "Risk labels are illustrative categories."
    )

    if st.button(
        "⚡ Calculate Test-Set Risk Scores",
        type="primary",
        key="calculate_risk",
    ):
        try:
            probabilities = fraud_probability(forest_model, X_test_df)

            report = X_test_df.copy()
            report["Actual Class"] = y_test
            report["Fraud Probability"] = probabilities
            report["Risk Level"] = [
                risk_level(float(p)) for p in probabilities
            ]
            report["Flagged"] = probabilities >= threshold

            st.session_state["risk_report"] = report

        except Exception as exc:
            st.error("Risk scoring failed.")
            st.exception(exc)

    if "risk_report" in st.session_state:
        report = st.session_state["risk_report"].copy()

        # Refresh the flag whenever the threshold changes.
        report["Flagged"] = report["Fraud Probability"] >= threshold

        total_flagged = int(report["Flagged"].sum())
        confirmed_fraud = int((report["Actual Class"] == 1).sum())
        correctly_flagged = int(
            ((report["Actual Class"] == 1) & report["Flagged"]).sum()
        )

        a, b, c, d = st.columns(4)
        a.metric("Scored Transactions", f"{len(report):,}")
        b.metric("Flagged for Review", f"{total_flagged:,}")
        c.metric("Actual Fraud Cases", f"{confirmed_fraud:,}")
        d.metric("Fraud Cases Flagged", f"{correctly_flagged:,}")

        st.plotly_chart(
            risk_distribution_chart(report),
            width="stretch",
        )

        flagged_report = report[report["Flagged"]].sort_values(
            "Fraud Probability",
            ascending=False,
        )

        st.subheader("🚨 Highest-Risk Transactions")

        if flagged_report.empty:
            st.info("No transactions meet the selected alert threshold.")
        else:
            st.dataframe(
                flagged_report.head(100),
                width="stretch",
                hide_index=True,
            )

        st.download_button(
            "⬇ Download Complete Risk Report",
            data=report.to_csv(index=False).encode("utf-8"),
            file_name="fraud_risk_report.csv",
            mime="text/csv",
            key="download_risk_report",
        )

        st.caption(
            "This is a retrospective test-set analysis, not a live payment "
            "monitoring service. Do not use the test set to tune the final "
            "threshold or as a substitute for independent evaluation."
        )


# ============================================================
# TAB 4 — CSV PREDICTIONS
# ============================================================

with prediction_tab:
    section_heading(
        "BATCH PREDICTION",
        "Transaction Risk Assessment",
        "Upload a compatible feature CSV to estimate fraud probability for each row.",
    )

    st.info(
        "Your CSV must contain the exact model feature columns and compatible "
        "units. The Class column is optional and is never used as an input."
    )

    with st.expander("📋 Required feature columns"):
        st.code(", ".join(feature_names))

    required = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
    template_df = pd.DataFrame(
        [dict.fromkeys(required, 0.0)],
        columns=required,
    )
    st.download_button(
        "⬇ Download CSV Template",
        data=template_df.to_csv(index=False).encode("utf-8"),
        file_name="prediction_template.csv",
        mime="text/csv",
        key="download_prediction_template",
    )
    st.caption(
        "Fill this template with real feature values from your source system. "
        "Do not fill V-features with zeros — the model was trained on real "
        "PCA values, and zero-filling will produce meaningless predictions."
    )

    model_choice = st.selectbox(
        "Prediction model",
        ["Random Forest", "Logistic Regression"],
        key="prediction_model_choice",
    )

    uploaded_file = st.file_uploader(
        "Upload a transaction CSV",
        type=["csv"],
        key="prediction_csv",
        help="Use a CSV with the same feature definitions as the training data.",
    )

    if uploaded_file is not None:
        # ---- STEP 1: validate the upload ----
        try:
            uploaded_df, prediction_input = validate_csv(uploaded_file)
        except (ValueError, KeyError, TypeError) as e:
            st.error(f"Invalid CSV file: {e}")
            st.info(
                "Upload a CSV containing all required model features: "
                "Time, V1–V28, and Amount."
            )
        else:
            # ---- STEP 2: main processing (own try/except) ----
            try:
                st.success(
                    f"File validated successfully · {len(prediction_input):,} rows"
                )

                with st.expander("Preview uploaded data"):
                    st.dataframe(
                        uploaded_df.head(10),
                        width="stretch",
                        hide_index=True,
                    )

                current_signature = (
                    uploaded_file.name,
                    uploaded_file.size,
                    model_choice,
                )

                if st.button(
                    "🔎 Predict Fraud Risk",
                    type="primary",
                    key="run_prediction",
                ):
                    with st.spinner("Scoring transactions..."):
                        if model_choice == "Logistic Regression":
                            if scaler is None:
                                raise ValueError(
                                    "Saved scaler not found in "
                                    "prepared_data.joblib."
                                )
                            prediction_model = logistic_model
                            model_input = scaler.transform(prediction_input)
                        else:
                            prediction_model = forest_model
                            model_input = prediction_input

                        probabilities = fraud_probability(
                            prediction_model,
                            model_input,
                        )

                        results = uploaded_df.copy()
                        results["Fraud Probability"] = probabilities
                        results["Risk Level"] = [
                            risk_level(float(p)) for p in probabilities
                        ]
                        results["Predicted Class"] = (
                            probabilities >= threshold
                        ).astype(int)
                        results["Decision"] = results[
                            "Predicted Class"
                        ].map({
                            0: "Not flagged",
                            1: "Flagged for review",
                        })

                        st.session_state["prediction_results"] = results
                        st.session_state["prediction_features"] = (
                            prediction_input.copy()
                        )
                        st.session_state["prediction_model_used"] = model_choice
                        st.session_state["prediction_signature"] = (
                            current_signature
                        )

                if "prediction_results" in st.session_state:
                    if (
                        st.session_state.get("prediction_signature")
                        != current_signature
                    ):
                        # Stale results from a previous upload/model.
                        st.session_state.pop("prediction_results", None)
                        st.session_state.pop("prediction_features", None)
                        st.session_state.pop("prediction_model_used", None)
                        st.info(
                            "Click Predict Fraud Risk to score this upload."
                        )
                    else:
                        results = st.session_state[
                            "prediction_results"
                        ].copy()

                        # Apply current threshold without rescoring.
                        results["Predicted Class"] = (
                            results["Fraud Probability"] >= threshold
                        ).astype(int)
                        results["Decision"] = results[
                            "Predicted Class"
                        ].map({
                            0: "Not flagged",
                            1: "Flagged for review",
                        })

                        st.caption(
                            "Model used: "
                            + str(
                                st.session_state.get(
                                    "prediction_model_used", "Unknown"
                                )
                            )
                        )

                        total_rows = len(results)
                        flagged_rows = int(results["Predicted Class"].sum())
                        flagged_rate = (
                            100 * flagged_rows / total_rows
                            if total_rows
                            else 0
                        )

                        a, b, c = st.columns(3)
                        a.metric("Transactions Scored", f"{total_rows:,}")
                        b.metric("Flagged for Review", f"{flagged_rows:,}")
                        c.metric("Flagged Rate", f"{flagged_rate:.2f}%")

                        st.plotly_chart(
                            risk_distribution_chart(results),
                            width="stretch",
                        )

                        st.subheader("Prediction Results")

                        st.dataframe(
                            results.head(500),
                            width="stretch",
                            hide_index=True,
                        )

                        st.download_button(
                            "⬇ Download Prediction Results",
                            data=results.to_csv(index=False).encode("utf-8"),
                            file_name="fraud_predictions.csv",
                            mime="text/csv",
                            key="download_predictions",
                        )

                        st.caption(
                            "A flagged transaction is a model-generated signal "
                            "for human review, not proof of fraud."
                        )
            except Exception as exc:
                st.error("Could not process this CSV.")
                st.exception(exc)


# ============================================================
# TAB 5 — SHAP EXPLAINABILITY
# ============================================================

with shap_tab:
    section_heading(
        "EXPLAINABLE AI",
        "Understand Model Behavior",
        "Inspect feature contributions for a Random Forest prediction.",
    )

    st.info(
        "SHAP explanations below are for Random Forest, using unscaled "
        "features. Contributions explain the model output and do not "
        "establish causation."
    )

    try:
        import shap
        import matplotlib.pyplot as plt

        background_source = (
            X_train_df
            if X_train_df is not None and not X_train_df.empty
            else X_test_df
        )

        background = background_source.sample(
            n=min(100, len(background_source)),
            random_state=42,
        )

        def build_explainer():
            return shap.TreeExplainer(
                forest_model,
                data=background,
            )

        def normalize_shap_values(result, n_features):
            if isinstance(result, list):
                values = np.asarray(
                    result[1] if len(result) > 1 else result[0]
                )
            elif hasattr(result, "values"):
                values = np.asarray(result.values)
            else:
                values = np.asarray(result)

            # SHAP versions may return (samples, features, classes).
            if values.ndim == 3:
                if values.shape[-1] > 1:
                    values = values[:, :, 1]
                elif values.shape[0] > 1:
                    values = values[1]

            if values.ndim == 1:
                values = values.reshape(1, -1)

            if values.ndim != 2 or values.shape[1] != n_features:
                raise ValueError(
                    f"Unexpected SHAP array shape {values.shape}."
                )

            return values

        def base_value_for_fraud(explainer):
            expected = np.asarray(explainer.expected_value).reshape(-1)

            if len(expected) > 1:
                return float(expected[1])
            if len(expected) == 1:
                return float(expected[0])

            raise ValueError("SHAP base value is unavailable.")

        explain_source = st.radio(
            "Explanation dataset",
            ["Test dataset", "Last uploaded CSV"],
            horizontal=True,
            key="shap_source",
        )

        if (
            explain_source == "Last uploaded CSV"
            and "prediction_features" in st.session_state
        ):
            explain_df = st.session_state["prediction_features"]
        else:
            explain_df = X_test_df

            if explain_source == "Last uploaded CSV":
                st.warning(
                    "No scored CSV is available yet. Showing test-set "
                    "transactions."
                )

        if explain_df.empty:
            st.warning("No transactions are available to explain.")
        else:
            explain_index = st.number_input(
                "Transaction row index",
                min_value=0,
                max_value=len(explain_df) - 1,
                value=0,
                step=1,
                key="shap_row_index",
            )

            if st.button(
                "🧠 Explain Selected Transaction",
                type="primary",
                key="run_individual_shap",
            ):
                with st.spinner("Calculating feature contributions..."):
                    transaction = explain_df.iloc[[int(explain_index)]].copy()

                    explainer = build_explainer()
                    shap_result = explainer.shap_values(
                        transaction,
                        check_additivity=False,
                    )

                    values = normalize_shap_values(
                        shap_result,
                        len(feature_names),
                    )[0]

                    base_value = base_value_for_fraud(explainer)

                    explanation = shap.Explanation(
                        values=values,
                        base_values=base_value,
                        data=transaction.iloc[0].to_numpy(),
                        feature_names=feature_names,
                    )

                    st.subheader("Feature Contribution Waterfall")

                    shap.plots.waterfall(
                        explanation,
                        max_display=12,
                        show=False,
                    )

                    fig = plt.gcf()
                    # st.pyplot accepts use_container_width (not width)
                    st.pyplot(fig, use_container_width=True)
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

                    st.subheader("Most Influential Features")
                    st.dataframe(
                        contributions.head(15).round(6),
                        width="stretch",
                        hide_index=True,
                    )

                    st.download_button(
                        "⬇ Download Individual Explanation",
                        data=contributions.to_csv(index=False).encode("utf-8"),
                        file_name="individual_shap_explanation.csv",
                        mime="text/csv",
                        key="download_individual_shap",
                    )

            st.divider()
            st.subheader("Global Feature Importance")

            if st.button(
                "📊 Calculate Global Importance",
                key="run_global_shap",
            ):
                with st.spinner("Calculating global feature importance..."):
                    sample = X_test_df.sample(
                        n=min(100, len(X_test_df)),
                        random_state=42,
                    )

                    explainer = build_explainer()
                    shap_result = explainer.shap_values(
                        sample,
                        check_additivity=False,
                    )

                    global_values = normalize_shap_values(
                        shap_result,
                        len(feature_names),
                    )

                    importance_df = pd.DataFrame({
                        "Feature": feature_names,
                        "Mean Absolute SHAP": np.abs(
                            global_values
                        ).mean(axis=0),
                    }).sort_values(
                        "Mean Absolute SHAP",
                        ascending=False,
                    )

                    st.session_state["global_shap_importance"] = importance_df

            if "global_shap_importance" in st.session_state:
                importance_df = st.session_state["global_shap_importance"]

                top_features = importance_df.head(15).sort_values(
                    "Mean Absolute SHAP",
                    ascending=True,
                )

                fig = go.Figure(
                    go.Bar(
                        x=top_features["Mean Absolute SHAP"],
                        y=top_features["Feature"],
                        orientation="h",
                        marker=dict(
                            color=top_features["Mean Absolute SHAP"],
                            colorscale=[
                                [0, "#164e63"],
                                [0.5, "#0891b2"],
                                [1, "#67e8f9"],
                            ],
                        ),
                    )
                )

                fig.update_layout(
                    title="Top 15 Global Feature Contributions",
                    xaxis_title="Mean Absolute SHAP Value",
                    yaxis_title="Feature",
                )

                st.plotly_chart(
                    style_chart(fig, 480),
                    width="stretch",
                )

                st.dataframe(
                    importance_df.head(15).round(6),
                    width="stretch",
                    hide_index=True,
                )

                st.download_button(
                    "⬇ Download Global Importance",
                    data=importance_df.to_csv(index=False).encode("utf-8"),
                    file_name="global_shap_importance.csv",
                    mime="text/csv",
                    key="download_global_shap",
                )

    except ImportError:
        st.warning(
            "SHAP is not installed in this Python environment. "
            "Install it with: python -m pip install shap matplotlib"
        )

    except Exception as exc:
        st.error("SHAP analysis could not be completed.")
        st.exception(exc)


# ============================================================
# 8. FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        <b style="color:#cbd5e1">FRAUDSHIELD ANALYTICS</b><br>
        Python · Pandas · Scikit-learn · Plotly · Streamlit · SHAP<br>
        Educational portfolio project. Model scores are estimates, not proof
        of fraud. Do not use them as the sole basis for real financial decisions.
    </div>
    """,
    unsafe_allow_html=True,
)