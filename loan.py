"""
Loan Approval Checker Application
Streamlit web app that predicts loan approval using a trained ML model.
"""

import pandas as pd
import joblib
import streamlit as st
import os
from log import logger

# Model and data paths
MODEL_PATH = "model.joblib"
ACCURACY_PATH = "accuracy.txt"
TRAIN_DATA_PATH = "train.csv"

# Feature columns for model prediction
FEATURES = ['Married', 'Education', 'ApplicantIncome', 'CoapplicantIncome',
            'LoanAmount', 'Loan_Amount_Term', 'Credit_History']

# Configure Streamlit page
st.set_page_config(
    page_title="Loan Approval Checker",
    page_icon="🏦",
    layout="centered"
)

st.title("🏦 Loan Approval Checker")

# Log application start
logger.info("="*70)
logger.info("APPLICATION INITIALIZED")
logger.info("="*70)

logger.info(f"Loading prediction model...")

# Check if model exists
if not os.path.exists(MODEL_PATH):
    logger.error("Model not found")
    st.warning("⚠️ Model not available")
    st.info("The prediction model has not been trained yet. Please run the training script first.")
    st.stop()

# Load model with caching for performance
@st.cache_resource
def load_model(path: str):
    return joblib.load(path)

model = load_model(MODEL_PATH)
logger.info("Model loaded successfully")

# Application form
with st.form("application_form"):
    st.subheader("Applicant Information")

    applicant_income = st.number_input(
        "Applicant Income (monthly)",
        min_value=0.0,
        value=5000.0,
        step=100.0
    )

    coapplicant_income = st.number_input(
        "Coapplicant Income (monthly)",
        min_value=0.0,
        value=0.0,
        step=100.0
    )

    loan_amount = st.number_input(
        "Requested Loan Amount",
        min_value=50000.0,
        max_value=200000.0,
        value=100000.0,
        step=5000.0
    )

    loan_term = st.number_input(
        "Loan Term (months)",
        min_value=1.0,
        value=360.0,
        step=12.0
    )

    credit_history = st.selectbox(
        "Credit History",
        options=[1, 0],
        index=0,
        format_func=lambda x: "Exists (1)" if x == 1 else "Does not exist (0)"
    )

    # Marital status options
    dict_opt = {"Married": "Married", "Single": "Single"}
    married = st.selectbox(
        "Marital Status",
        options=["Married", "Single"],
        index=0,
        format_func=lambda x: dict_opt[x]
    )

    education = st.selectbox(
        "Education",
        options=["Graduate", "Not Graduate"],
        index=0
    )

    submitted = st.form_submit_button("Check Loan Eligibility")

# Process prediction request
if submitted:
    married_val = 1 if married == "Married" else 0
    edu_val = 1 if education == "Graduate" else 0

    total_income = applicant_income + coapplicant_income
    logger.info(f"Prediction request: Married={married}, Education={education}, Credit={credit_history}, Total_Income={total_income}, Loan_Amount={loan_amount}")

    # Validation rules
    if married_val == 0:
        logger.warning("Rejection: Not married")
        st.error("❌ Applicant must be married to qualify")
    elif edu_val != 1:
        logger.warning("Rejection: Not graduate")
        st.error("❌ Applicant must have graduate education to qualify")
    elif credit_history != 1:
        logger.warning("Rejection: No credit history")
        st.error("❌ Valid credit history is required")
    elif applicant_income < 4000.0  and coapplicant_income < 1000.0:
        logger.warning("Rejection: Insufficient income")
        st.error("❌ Insufficient combined income (minimum 4000 applicant or 1000 co-applicant)")
    else:
        # Prepare data for prediction
        data = pd.DataFrame([{
            "Married": married_val,
            "Education": edu_val,
            "ApplicantIncome": float(applicant_income),
            "CoapplicantIncome": float(coapplicant_income),
            "LoanAmount": float(loan_amount),
            "Loan_Amount_Term": float(loan_term),
            "Credit_History": int(credit_history)
        }])

        try:
            # Get prediction from model
            result = model.predict(data)[0]
            if result == 1:
                logger.info(f"APPROVED: Income={total_income}, Loan={loan_amount}")
                st.success("✅ Congratulations! You are approved for the loan")
            else:
                logger.info(f"REJECTED: Income={total_income}, Loan={loan_amount}")
                st.error("❌ Your application has been declined")
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            st.error(f"⚠️ An error occurred during prediction: {str(e)}")

# Display model performance metrics
if st.button("📊 View Model Performance"):
    logger.info("Accuracy request")
    try:
        if os.path.exists(ACCURACY_PATH):
            with open(ACCURACY_PATH, 'r') as f:
                acc_text = f.read().strip()
                if acc_text:
                    accuracy = float(acc_text)
                    logger.info(f"Accuracy displayed: {accuracy:.1%}")
                    st.metric("Model Accuracy", f"{accuracy:.1%}")
                else:
                    logger.warning("Accuracy file empty")
                    st.info("Accuracy data is not available")
        else:
            logger.warning("Accuracy file missing")
            st.info("Model has not been trained yet")
    except ValueError as e:
        logger.error(f"Accuracy read error: {str(e)}")
        st.error("Unable to read model performance data")