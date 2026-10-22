# 🛡️ FraudShield Analytics
### Credit Card Fraud Detection & Transaction Risk Analytics

An interactive machine learning application built with Python and Streamlit to analyze credit card transactions, identify potentially fraudulent activity, and assess transaction risk using trained classification models.

<p align="center">
  <a href="https://github.com/Yash-0330/Fraud-Detection-Analytics">
    <img src="https://img.shields.io/badge/GitHub-Project-black?logo=github" alt="GitHub Project"/>
  </a>
  <img src="https://img.shields.io/badge/Python-3.x-blue?logo=python" alt="Python"/>
  <img src="https://img.shields.io/badge/Machine%20Learning-Scikit--learn-orange" alt="Machine Learning"/>
  <img src="https://img.shields.io/badge/Interface-Streamlit-FF4B4B?logo=streamlit" alt="Streamlit"/>
</p>

---

## 📌 Project Overview

Credit card fraud is a major challenge in digital financial transactions. Fraudulent transactions are often rare compared with legitimate transactions, making fraud detection a challenging machine learning problem.

**FraudShield Analytics** provides an interactive dashboard for exploring transaction data, evaluating trained machine learning models, predicting potential fraud, and reviewing transaction risk.

The project demonstrates how data preprocessing, machine learning, model evaluation, and interactive visualization can be combined into a practical analytics application.

## ✨ Key Features

- **📊 Interactive Dashboard:** Explore transaction statistics, fraud patterns, and analytical visualizations.
- **🤖 Machine Learning Predictions:** Use trained Logistic Regression and Random Forest models to classify transactions.
- **🚨 Transaction Risk Analysis:** Review predicted fraud probabilities and risk indicators, where supported by the selected model.
- **📁 CSV Batch Prediction:** Upload transaction data to generate predictions for multiple records.
- **📈 Model Performance Evaluation:** Review available classification metrics and compare model performance.
- **🔍 Model Explainability:** Explore feature contributions using SHAP when the required dependencies and compatible model are available.
- **⚡ Streamlit Interface:** Access project functionality through an interactive web dashboard.

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming | Python |
| Data Analysis | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Visualization | Matplotlib, Plotly, or other libraries used by the application |
| Web Dashboard | Streamlit |
| Model Explainability | SHAP |
| Data Format | CSV |
| Version Control | Git, GitHub |

*The visualization libraries listed should match the imports in your actual code.*

## 🧠 Machine Learning Approach

The project uses a supervised classification approach to distinguish potentially fraudulent transactions from legitimate ones.

1. **Data Preparation:** Load transaction data and validate the required features.
2. **Data Preprocessing:** Prepare the input features in the format expected by the trained models.
3. **Model Training:** Train and save classification models.
4. **Model Evaluation:** Examine suitable classification metrics, including precision, recall, F1-score, and ROC-AUC when available.
5. **Prediction:** Apply trained models to individual or uploaded transaction records.
6. **Interpretation:** Use supported visualizations and explainability methods to understand prediction results.

### Models

- Logistic Regression
- Random Forest Classifier

Because fraud datasets are typically imbalanced, model evaluation should consider fraud-class precision, recall, F1-score, and the confusion matrix rather than accuracy alone.

## 📂 Project Structure

```text
Fraud-Detection-Analytics/
│
├── app.py
├── final_dashboard.py
├── fraud_analytics.py
├── risk_scoring.py
├── explain_fraud.py
├── shap_dashboard.py
├── compare_models.py
├── preprocess_data.py
├── train_model.py
├── train_random_forest.py
├── threshold_dashboard.py
├── threshold_tuning.py
├── eda.py
├── explore_data.py
│
├── charts/
├── models/
├── reports/
├── data/                 # Local dataset; not committed
│
├── requirements.txt
├── .gitignore
└── README.md
```

*This structure reflects the project files currently identified. Keep only files and folders that actually exist in the repository.*

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/Yash-0330/Fraud-Detection-Analytics.git
cd Fraud-Detection-Analytics
```

### 2. Create a virtual environment

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

If a `requirements.txt` file is available:

```bash
python -m pip install -r requirements.txt
```

If it does not exist yet, create one using the dependencies imported by your application. Typical packages may include:

```text
streamlit
pandas
numpy
scikit-learn
matplotlib
plotly
joblib
shap
```

Install only the packages your code requires, and pin compatible versions after testing the application.

### 4. Prepare the dataset and model files

Place the required dataset and trained model artifacts in their expected local folders.

Expected model artifacts may include:

```text
models/
├── prepared_data.joblib
├── logistic_regression.joblib
├── random_forest.joblib
├── logistic_metrics.json
└── random_forest_metrics.json
```

These files must be available locally for the corresponding dashboard features to work. Obtain or regenerate any artifacts that are not included in the repository.

### 5. Run the dashboard

```bash
streamlit run app.py
```

Streamlit will display a local URL in the terminal. Open that URL in your browser to use the dashboard.

## 📄 Dataset

The project is designed for credit card transaction fraud detection using transaction features such as `Time`, anonymized features `V1` through `V28`, `Amount`, and the target label `Class`.

- `Class = 0`: Legitimate transaction
- `Class = 1`: Fraudulent transaction

The dataset is not included in this repository. Keep large raw datasets out of Git, and document the authorized source and download instructions before sharing the project publicly.

For CSV predictions, provide all features required by the trained model. The `Class` target column should not be used as a prediction input.

## 📊 Results & Evaluation

The dashboard is designed to present model evaluation results from the available training outputs.

Important evaluation measures include:

- Precision
- Recall
- F1-score
- Confusion matrix
- ROC-AUC, where supported

**Actual performance results:** Refer to the generated model metrics and reports. No accuracy or performance figures are claimed here because the verified metric values have not been provided.

## 🔮 Future Improvements

- Deploy the dashboard to a cloud hosting platform.
- Add automated model retraining and monitoring.
- Improve handling of class imbalance.
- Add configurable decision thresholds and risk categories.
- Expand model explainability and model comparison.
- Add automated tests and input validation.
- Improve deployment documentation and reproducibility.

## ⚠️ Disclaimer

This project is intended for educational and portfolio purposes. Predictions are probabilistic and may produce false positives or false negatives. The dashboard should not be treated as a production-grade financial fraud prevention system without further validation, security review, monitoring, and testing.

## 👨‍💻 Author

**Yash Rakhunde**  
Computer Engineering Graduate | Aspiring Data Analyst & Machine Learning Engineer

- **GitHub:** [Yash-0330](https://github.com/Yash-0330)
- **LinkedIn:** [linkedin.com/in/yashrakhunde](https://www.linkedin.com/in/yashrakhunde/)
- **Email:** [yashrakhunde5@gmail.com](mailto:yashrakhunde5@gmail.com)

---

⭐ If you find this project useful, consider starring the repository.

*Built to demonstrate practical skills in Python, data analytics, machine learning, and interactive dashboard development.*
