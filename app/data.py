from pathlib import Path
import json
import pickle

import numpy as np
import pandas as pd
import streamlit as st
import shap
import matplotlib.pyplot as plt


# ----------------------------
# Paths (SAFE, no ../ issues)
# ----------------------------
APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
MODELS_DIR = PROJECT_DIR / "models"

MODEL_PATH = MODELS_DIR / "xgb_model.pkl"
COLS_PATH = MODELS_DIR / "feature_columns.json"


# ----------------------------
# Streamlit config
# ----------------------------
st.set_page_config(page_title="Loan Approval Prediction", layout="wide")
st.title("Bank Loan Approval Prediction")
st.write("Predict loan approval and view SHAP explanation for your inputs.")


# ----------------------------
# Load model + feature columns
# ----------------------------
@st.cache_resource
def load_model_and_columns():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    if not COLS_PATH.exists():
        raise FileNotFoundError(f"Feature columns file not found: {COLS_PATH}")

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(COLS_PATH, "r") as f:
        feature_columns = json.load(f)

    return model, feature_columns


try:
    xgb, FEATURE_COLUMNS = load_model_and_columns()
except Exception as e:
    st.error("Could not load model files.")
    st.code(str(e))
    st.stop()


# ----------------------------
# Sidebar inputs
# ----------------------------
st.sidebar.header("Applicant Inputs")

no_of_dependents = st.sidebar.number_input(
    "Number of dependents", min_value=0, max_value=20, value=0, step=1
)

education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.sidebar.selectbox("Self employed", ["No", "Yes"])

income_annum = st.sidebar.number_input(
    "Annual income", min_value=0, value=5000000, step=100000
)
loan_amount = st.sidebar.number_input(
    "Loan amount", min_value=0, value=15000000, step=100000
)
loan_term = st.sidebar.number_input(
    "Loan term (years)", min_value=1, value=10, step=1
)

cibil_score = st.sidebar.number_input(
    "CIBIL score", min_value=300, max_value=900, value=700, step=1
)

# You redesigned model to use total_assets (not individual asset columns)
total_assets = st.sidebar.number_input(
    "Total assets value", min_value=0, value=20000000, step=100000
)


# ----------------------------
# Build features (must match training logic)
# ----------------------------
def build_features() -> pd.DataFrame:
    # Engineered features used in your final model
    emi_estimate = loan_amount / loan_term

    # Monthly EMI / Monthly income (realistic risk ratio)
    if income_annum > 0:
        loan_income_ratio = emi_estimate / (income_annum / 12)
    else:
        loan_income_ratio = 0.0

    row = {
        "no_of_dependents": no_of_dependents,
        "income_annum": income_annum,
        "loan_amount": loan_amount,
        "loan_term": loan_term,
        "cibil_score": cibil_score,
        "total_assets": total_assets,
        "emi_estimate": emi_estimate,
        "loan_income_ratio": loan_income_ratio,

        # These match your get_dummies(drop_first=True) column names
        "education_ Not Graduate": 1 if education == "Not Graduate" else 0,
        "self_employed_ Yes": 1 if self_employed == "Yes" else 0,
    }

    X_input = pd.DataFrame([row])

    # Align to training columns (add missing with 0, drop extras)
    for col in FEATURE_COLUMNS:
        if col not in X_input.columns:
            X_input[col] = 0

    X_input = X_input[FEATURE_COLUMNS]

    # Ensure everything is numeric float (SHAP + XGBoost friendly)
    X_input = X_input.astype("float64")

    return X_input


# ----------------------------
# Main layout
# ----------------------------
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Prediction")

    predict_clicked = st.button("Predict")

    if predict_clicked:
        X_input = build_features()

        # Predict probability for class 1 (Approved)
        proba_approved = float(xgb.predict_proba(X_input)[0, 1])
        pred = 1 if proba_approved >= 0.5 else 0

        if pred == 1:
            st.success(f"Approved ✅  (Probability: {proba_approved:.3f})")
        else:
            st.error(f"Rejected ❌  (Probability: {proba_approved:.3f})")

        st.markdown("### Inputs used by the model")
        st.dataframe(X_input)

        st.session_state["X_input"] = X_input
        st.session_state["proba"] = proba_approved
        st.session_state["pred"] = pred

with col2:
    st.subheader("SHAP Explanation (Waterfall)")

    if "X_input" not in st.session_state:
        st.info("Click Predict to generate SHAP explanation.")
    else:
        X_input = st.session_state["X_input"]

        # SHAP for tree model
        explainer = shap.TreeExplainer(xgb)
        shap_values = explainer.shap_values(X_input)

        base_value = explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)):
            base_value = base_value[0]

        explanation = shap.Explanation(
            values=shap_values[0],
            base_values=base_value,
            data=X_input.iloc[0],
            feature_names=X_input.columns,
        )

        fig = plt.figure()
        shap.plots.waterfall(explanation, show=False)
        st.pyplot(fig)
        plt.close(fig)

        st.caption("Red pushes towards approval, blue pushes towards rejection (relative to model base value).")
