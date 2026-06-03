# Customer Churn Risk Calculator
### ML-Powered Streamlit App — Live Churn Prediction Tool

![Python](https://img.shields.io/badge/Python-3.13-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.58-FF4B4B?style=flat-square&logo=streamlit)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.8-F7931E?style=flat-square&logo=scikit-learn)
![Plotly](https://img.shields.io/badge/Plotly-6.7-3F4F75?style=flat-square&logo=plotly)
![Status](https://img.shields.io/badge/Status-Live-brightgreen?style=flat-square)

**Live app: [churn-risk-calculator.streamlit.app](https://churn-risk-calculator.streamlit.app)**

---

## Overview

This is an interactive web app that predicts customer churn probability in real time. Enter a customer's profile — contract type, tenure, services subscribed, payment method — and the app instantly returns a churn risk score, risk level classification, and the key factors driving that prediction.

Built on top of the Logistic Regression model from [Project 3](https://github.com/devipanjaitan/customer-churn-prediction), this app demonstrates the full data science pipeline: from training a model in a notebook to deploying it as a usable product that anyone can interact with — no installation required.

---

## What the App Does

The app takes customer attributes as input and returns three outputs:

- **Churn probability** — a percentage score from 0% to 100%
- **Risk level** — Low, Medium, or High based on probability threshold
- **Key risk factors** — the specific attributes most contributing to churn risk for that customer

The gauge chart updates in real time as you adjust the inputs, making it easy to see how changes in contract type, tenure, or number of services affect churn likelihood.

---

## Model Details

| Attribute | Value |
|---|---|
| Model | Logistic Regression |
| Training data | 7,043 Telco customers |
| AUC-ROC | 0.836 |
| Features | 33 (after one-hot encoding + feature engineering) |
| Class imbalance handling | SMOTE |
| Key engineered feature | num_services (strongest predictor per SHAP analysis) |

Logistic Regression was selected over Random Forest and XGBoost because it achieved the highest AUC-ROC (0.836) on this dataset — a reminder that simpler models often outperform complex ones on well-structured tabular data.

---

## Risk Factors the App Tracks

Based on SHAP value analysis from the model training phase, the app highlights the following as high-risk signals:

- Month-to-month contract — churn rate 42.7% vs 2.8% for two-year contracts
- Electronic check payment method
- Low tenure (under 12 months) — the highest-risk window for retention
- Fiber optic internet service
- Fewer than 3 subscribed services — low platform engagement

---

## How to Run Locally

```bash
# Clone this repository
git clone https://github.com/devipanjaitan/churn-risk-calculator.git
cd churn-risk-calculator

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## Project Structure

```
churn-risk-calculator/
├── app.py                  ← Streamlit app
├── requirements.txt        ← Dependencies
└── model/
    ├── logistic_model.pkl  ← Trained Logistic Regression model
    ├── scaler.pkl          ← StandardScaler fitted on training data
    └── features.json       ← Feature names (33 columns)
```

---

## Related Projects

- [Project 3 — Customer Churn Prediction (model training + SHAP analysis)](https://github.com/devipanjaitan/customer-churn-prediction)
- [Project 1 — E-Commerce EDA & Consumer Behavior](https://github.com/devipanjaitan/olist-ecommerce-analysis)
- [Project 2 — SQL Sales Performance Analysis](https://github.com/devipanjaitan/olist-sql-analysis)
- [Project 4 — Air Quality Analysis Southeast Asia](https://github.com/devipanjaitan/sea-air-quality-analysis)

---

## Author

**Devi Silvia Panjaitan**
- [LinkedIn](https://linkedin.com/in/devipanjaitan)
- [GitHub](https://github.com/devipanjaitan)
