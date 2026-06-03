import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import plotly.graph_objects as go

# --- Page config ---
st.set_page_config(
    page_title="Customer Churn Risk Calculator",
    page_icon="📊",
    layout="wide"
)

# --- Load model ---
@st.cache_resource
def load_model():
    with open('model/logistic_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('model/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('model/features.json', 'r') as f:
        features = json.load(f)
    return model, scaler, features

model, scaler, feature_names = load_model()

# --- Header ---
st.title("Customer Churn Risk Calculator")
st.markdown("Enter customer details to predict churn probability using a Logistic Regression model trained on 7,043 Telco customers.")
st.divider()

# --- Sidebar inputs ---
st.sidebar.header("Customer Profile")
st.sidebar.markdown("Fill in the customer details below.")

# Basic info
tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.sidebar.slider("Monthly Charges (USD)", 18, 120, 65)
total_charges = st.sidebar.number_input("Total Charges (USD)",
                                         min_value=0.0,
                                         value=float(tenure * monthly_charges))

st.sidebar.divider()

# Contract & billing
contract = st.sidebar.selectbox("Contract Type",
    ["Month-to-month", "One year", "Two year"])
payment_method = st.sidebar.selectbox("Payment Method",
    ["Electronic check", "Mailed check",
     "Bank transfer (automatic)", "Credit card (automatic)"])
paperless_billing = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])

st.sidebar.divider()

# Services
st.sidebar.markdown("**Services Subscribed**")
phone_service     = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
multiple_lines    = st.sidebar.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
internet_service  = st.sidebar.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
online_security   = st.sidebar.selectbox("Online Security", ["Yes", "No", "No internet service"])
online_backup     = st.sidebar.selectbox("Online Backup", ["Yes", "No", "No internet service"])
device_protection = st.sidebar.selectbox("Device Protection", ["Yes", "No", "No internet service"])
tech_support      = st.sidebar.selectbox("Tech Support", ["Yes", "No", "No internet service"])
streaming_tv      = st.sidebar.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
streaming_movies  = st.sidebar.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

st.sidebar.divider()

# Demographics
senior_citizen = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
partner        = st.sidebar.selectbox("Has Partner", ["Yes", "No"])
dependents     = st.sidebar.selectbox("Has Dependents", ["Yes", "No"])
gender         = st.sidebar.selectbox("Gender", ["Male", "Female"])

# --- Feature engineering (must match training) ---
def build_input(feature_names):
    # Count services
    services = [phone_service, internet_service, online_security,
                online_backup, device_protection, tech_support,
                streaming_tv, streaming_movies]
    num_services = sum([
        1 if phone_service == "Yes" else 0,
        1 if internet_service != "No" else 0,
        1 if online_security == "Yes" else 0,
        1 if online_backup == "Yes" else 0,
        1 if device_protection == "Yes" else 0,
        1 if tech_support == "Yes" else 0,
        1 if streaming_tv == "Yes" else 0,
        1 if streaming_movies == "Yes" else 0,
    ])

    charges_per_month_ratio = total_charges / (tenure + 1)
    is_senior_no_support = 1 if (senior_citizen == "Yes" and tech_support == "No") else 0

    # Build base dict with all zeros
    row = {f: 0 for f in feature_names}

    # Numeric features
    if 'tenure' in row:                       row['tenure'] = tenure
    if 'MonthlyCharges' in row:               row['MonthlyCharges'] = monthly_charges
    if 'TotalCharges' in row:                 row['TotalCharges'] = total_charges
    if 'num_services' in row:                 row['num_services'] = num_services
    if 'charges_per_month_ratio' in row:      row['charges_per_month_ratio'] = charges_per_month_ratio
    if 'is_senior_no_support' in row:         row['is_senior_no_support'] = is_senior_no_support
    if 'SeniorCitizen' in row:               row['SeniorCitizen'] = 1 if senior_citizen == "Yes" else 0

    # One-hot encoded features
    mapping = {
        'gender_Male'                              : gender == "Male",
        'Partner_Yes'                              : partner == "Yes",
        'Dependents_Yes'                           : dependents == "Yes",
        'PhoneService_Yes'                         : phone_service == "Yes",
        'MultipleLines_No phone service'           : multiple_lines == "No phone service",
        'MultipleLines_Yes'                        : multiple_lines == "Yes",
        'InternetService_Fiber optic'              : internet_service == "Fiber optic",
        'InternetService_No'                       : internet_service == "No",
        'OnlineSecurity_No internet service'       : online_security == "No internet service",
        'OnlineSecurity_Yes'                       : online_security == "Yes",
        'OnlineBackup_No internet service'         : online_backup == "No internet service",
        'OnlineBackup_Yes'                         : online_backup == "Yes",
        'DeviceProtection_No internet service'     : device_protection == "No internet service",
        'DeviceProtection_Yes'                     : device_protection == "Yes",
        'TechSupport_No internet service'          : tech_support == "No internet service",
        'TechSupport_Yes'                          : tech_support == "Yes",
        'StreamingTV_No internet service'          : streaming_tv == "No internet service",
        'StreamingTV_Yes'                          : streaming_tv == "Yes",
        'StreamingMovies_No internet service'      : streaming_movies == "No internet service",
        'StreamingMovies_Yes'                      : streaming_movies == "Yes",
        'Contract_One year'                        : contract == "One year",
        'Contract_Two year'                        : contract == "Two year",
        'PaperlessBilling_Yes'                     : paperless_billing == "Yes",
        'PaymentMethod_Credit card (automatic)'    : payment_method == "Credit card (automatic)",
        'PaymentMethod_Electronic check'           : payment_method == "Electronic check",
        'PaymentMethod_Mailed check'               : payment_method == "Mailed check",
    }

    for key, val in mapping.items():
        if key in row:
            row[key] = 1 if val else 0

    return pd.DataFrame([row])[feature_names]

# --- Predict ---
input_df    = build_input(feature_names)
input_scaled = scaler.transform(input_df)
churn_prob  = model.predict_proba(input_scaled)[0][1]
churn_pred  = model.predict(input_scaled)[0]

# --- Main display ---
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Churn Probability", f"{churn_prob*100:.1f}%")

with col2:
    risk_label = "High Risk" if churn_prob >= 0.7 else "Medium Risk" if churn_prob >= 0.4 else "Low Risk"
    risk_color = "red" if churn_prob >= 0.7 else "orange" if churn_prob >= 0.4 else "green"
    st.metric("Risk Level", risk_label)

with col3:
    st.metric("Services Subscribed",
              f"{sum([1 if phone_service=='Yes' else 0, 1 if internet_service!='No' else 0, 1 if online_security=='Yes' else 0, 1 if online_backup=='Yes' else 0, 1 if device_protection=='Yes' else 0, 1 if tech_support=='Yes' else 0, 1 if streaming_tv=='Yes' else 0, 1 if streaming_movies=='Yes' else 0])} / 8")

st.divider()

# --- Gauge chart ---
col_gauge, col_info = st.columns([1, 1])

with col_gauge:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=churn_prob * 100,
        title={'text': "Churn Risk Score"},
        number={'suffix': "%", 'font': {'size': 40}},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#E84855" if churn_prob >= 0.7 else "#F4A261" if churn_prob >= 0.4 else "#2A9D8F"},
            'steps': [
                {'range': [0, 40],  'color': '#E8F5E9'},
                {'range': [40, 70], 'color': '#FFF8E1'},
                {'range': [70, 100],'color': '#FFEBEE'},
            ],
            'threshold': {
                'line': {'color': "black", 'width': 3},
                'thickness': 0.75,
                'value': churn_prob * 100
            }
        }
    ))
    fig.update_layout(height=300, margin=dict(t=50, b=0, l=30, r=30))
    st.plotly_chart(fig, use_container_width=True)

with col_info:
    st.markdown("### Risk Interpretation")
    if churn_prob >= 0.7:
        st.error(f"**High churn risk ({churn_prob*100:.1f}%).** This customer is very likely to leave. Immediate retention action is recommended.")
    elif churn_prob >= 0.4:
        st.warning(f"**Medium churn risk ({churn_prob*100:.1f}%).** This customer shows some signs of potential churn. Monitor closely and consider a proactive offer.")
    else:
        st.success(f"**Low churn risk ({churn_prob*100:.1f}%).** This customer appears stable. Continue regular engagement.")

    st.divider()
    st.markdown("### Key Risk Factors")
    if contract == "Month-to-month":
        st.markdown("- Month-to-month contract — highest churn rate (42.7%)")
    if payment_method == "Electronic check":
        st.markdown("- Electronic check payment — associated with higher churn")
    if tenure < 12:
        st.markdown(f"- Low tenure ({tenure} months) — customers in first year are most at risk")
    if internet_service == "Fiber optic":
        st.markdown("- Fiber optic internet — higher churn due to elevated monthly cost")
    if num_services := sum([1 if phone_service=='Yes' else 0, 1 if internet_service!='No' else 0,
                            1 if online_security=='Yes' else 0, 1 if online_backup=='Yes' else 0,
                            1 if device_protection=='Yes' else 0, 1 if tech_support=='Yes' else 0,
                            1 if streaming_tv=='Yes' else 0, 1 if streaming_movies=='Yes' else 0]):
        if num_services <= 2:
            st.markdown(f"- Only {num_services} service(s) subscribed — low platform engagement")

st.divider()

# --- Customer summary ---
st.markdown("### Customer Summary")
summary_col1, summary_col2 = st.columns(2)

with summary_col1:
    st.markdown(f"""
    | Attribute | Value |
    |---|---|
    | Tenure | {tenure} months |
    | Monthly Charges | ${monthly_charges} |
    | Total Charges | ${total_charges:.0f} |
    | Contract | {contract} |
    | Payment Method | {payment_method} |
    """)

with summary_col2:
    st.markdown(f"""
    | Attribute | Value |
    |---|---|
    | Internet Service | {internet_service} |
    | Senior Citizen | {senior_citizen} |
    | Partner | {partner} |
    | Dependents | {dependents} |
    | Paperless Billing | {paperless_billing} |
    """)

st.divider()
st.caption("Model: Logistic Regression | AUC-ROC: 0.836 | Trained on Telco Customer Churn Dataset (7,043 customers) | Project by Devi Silvia Panjaitan")