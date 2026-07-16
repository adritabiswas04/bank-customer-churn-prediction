import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Bank Customer Churn Risk Scoring", page_icon="🏦", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "bank_churn_model.joblib"
META_PATH = BASE_DIR / "models" / "model_metadata.json"
DATA_PATH = BASE_DIR / "data" / "European_Bank.csv"
IMPORTANCE_PATH = BASE_DIR / "feature_importance.csv"
RISK_PATH = BASE_DIR / "risk_scored_customers.csv"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_data
def load_metadata():
    with open(META_PATH, "r") as f:
        return json.load(f)

@st.cache_data
def load_importance():
    return pd.read_csv(IMPORTANCE_PATH)

@st.cache_data
def load_risk_scores():
    return pd.read_csv(RISK_PATH)

def engineer_features(input_df: pd.DataFrame) -> pd.DataFrame:
    out = input_df.copy()
    out["BalanceSalaryRatio"] = out["Balance"] / (out["EstimatedSalary"] + 1.0)
    out["ProductDensity"] = out["NumOfProducts"] / (out["Tenure"] + 1.0)
    out["EngagementProductInteraction"] = out["IsActiveMember"] * out["NumOfProducts"]
    out["AgeTenureInteraction"] = out["Age"] * out["Tenure"]
    return out

def risk_band(probability: float) -> str:
    if probability < 0.30:
        return "Low"
    if probability < 0.60:
        return "Medium"
    return "High"

def risk_recommendation(band: str) -> str:
    if band == "High":
        return "Immediate retention action: relationship-manager outreach, fee review, product-fit check, and personalized offer."
    if band == "Medium":
        return "Monitor closely: trigger engagement campaign, evaluate cross-sell suitability, and check service satisfaction."
    return "Maintain loyalty: continue normal engagement and periodic satisfaction monitoring."

model = load_model()
metadata = load_metadata()
df = load_data()
importance = load_importance()
risk_scores = load_risk_scores()

st.title("🏦 Predictive Modeling and Risk Scoring for Bank Customer Churn")
st.caption("A predictive churn intelligence system for proactive retention campaigns, risk scoring, and model explainability.")

st.sidebar.header("Navigation")
page = st.sidebar.radio("Choose module", [
    "Customer Churn Risk Calculator",
    "Probability Distribution Visualization",
    "Feature Importance Dashboard",
    "What-if Scenario Simulator",
    "Portfolio Risk Table"
])

st.sidebar.markdown("---")
st.sidebar.metric("Best model", metadata.get("best_model", "Model"))
st.sidebar.metric("Decision threshold", f"{metadata.get('threshold', 0.5):.2f}")
st.sidebar.metric("Dataset churn rate", f"{metadata.get('churn_rate', 0)*100:.2f}%")

base_features = [
    "CreditScore", "Geography", "Gender", "Age", "Tenure", "Balance",
    "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"
]

def input_form(prefix="customer"):
    c1, c2, c3 = st.columns(3)
    with c1:
        credit_score = st.slider(f"Credit score", 300, 900, 650, key=f"{prefix}_credit")
        geography = st.selectbox("Geography", sorted(df["Geography"].unique()), key=f"{prefix}_geo")
        gender = st.selectbox("Gender", sorted(df["Gender"].unique()), key=f"{prefix}_gender")
        age = st.slider("Age", int(df["Age"].min()), int(df["Age"].max()), 40, key=f"{prefix}_age")
    with c2:
        tenure = st.slider("Tenure years", int(df["Tenure"].min()), int(df["Tenure"].max()), 5, key=f"{prefix}_tenure")
        balance = st.number_input("Account balance", min_value=0.0, value=float(df["Balance"].median()), step=1000.0, key=f"{prefix}_balance")
        products = st.slider("Number of products", int(df["NumOfProducts"].min()), int(df["NumOfProducts"].max()), 1, key=f"{prefix}_products")
    with c3:
        has_card = st.selectbox("Has credit card", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", key=f"{prefix}_card")
        active = st.selectbox("Is active member", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", key=f"{prefix}_active")
        salary = st.number_input("Estimated salary", min_value=0.0, value=float(df["EstimatedSalary"].median()), step=1000.0, key=f"{prefix}_salary")
    return pd.DataFrame([{ 
        "CreditScore": credit_score,
        "Geography": geography,
        "Gender": gender,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": products,
        "HasCrCard": has_card,
        "IsActiveMember": active,
        "EstimatedSalary": salary,
    }])

def predict_customer(customer_df):
    engineered = engineer_features(customer_df)
    proba = float(model.predict_proba(engineered[metadata["features"]])[:, 1][0])
    flag = int(proba >= metadata["threshold"])
    band = risk_band(proba)
    return proba, flag, band

if page == "Customer Churn Risk Calculator":
    st.header("Customer churn risk calculator")
    customer = input_form("calc")
    proba, flag, band = predict_customer(customer)
    m1, m2, m3 = st.columns(3)
    m1.metric("Churn probability", f"{proba*100:.2f}%")
    m2.metric("Risk band", band)
    m3.metric("Churn flag", "Yes" if flag else "No")
    st.info(risk_recommendation(band))
    st.subheader("Input record with engineered features")
    st.dataframe(engineer_features(customer), use_container_width=True)

elif page == "Probability Distribution Visualization":
    st.header("Portfolio probability distribution")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.hist(risk_scores["ChurnProbability"], bins=30)
    ax.axvline(0.30, linestyle="--", label="Low/Medium cutoff")
    ax.axvline(0.60, linestyle="--", label="Medium/High cutoff")
    ax.axvline(metadata["threshold"], linestyle="-", label=f"Flag threshold ({metadata['threshold']:.2f})")
    ax.set_xlabel("Predicted churn probability")
    ax.set_ylabel("Number of customers")
    ax.set_title("Predicted Churn Probability Distribution")
    ax.legend()
    st.pyplot(fig)
    st.subheader("Risk band counts")
    st.dataframe(risk_scores["RiskBand"].value_counts().rename_axis("RiskBand").reset_index(name="Customers"), use_container_width=True)

elif page == "Feature Importance Dashboard":
    st.header("Feature importance dashboard")
    top_n = st.slider("Number of top features", 5, min(20, len(importance)), 12)
    top = importance.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top["Feature"], top["Importance"])
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} churn drivers")
    st.pyplot(fig)
    st.dataframe(importance.head(top_n), use_container_width=True)
    st.caption("Interpretation: higher importance means the model relied more on that variable while separating churned and retained customers.")

elif page == "What-if Scenario Simulator":
    st.header("What-if scenario simulator")
    st.write("Compare a baseline customer profile against an adjusted scenario to see how churn probability changes.")
    base = input_form("base")
    st.markdown("### Scenario adjustments")
    s1, s2, s3 = st.columns(3)
    scenario = base.copy()
    with s1:
        scenario.loc[0, "IsActiveMember"] = st.selectbox("Scenario active member", [0, 1], index=int(base.loc[0, "IsActiveMember"]), format_func=lambda x: "Yes" if x == 1 else "No")
    with s2:
        scenario.loc[0, "NumOfProducts"] = st.slider("Scenario products", int(df["NumOfProducts"].min()), int(df["NumOfProducts"].max()), int(base.loc[0, "NumOfProducts"]))
    with s3:
        scenario.loc[0, "Balance"] = st.number_input("Scenario balance", min_value=0.0, value=float(base.loc[0, "Balance"]), step=1000.0)
    base_p, _, base_band = predict_customer(base)
    scen_p, _, scen_band = predict_customer(scenario)
    d1, d2, d3 = st.columns(3)
    d1.metric("Baseline probability", f"{base_p*100:.2f}%", base_band)
    d2.metric("Scenario probability", f"{scen_p*100:.2f}%", scen_band)
    d3.metric("Change", f"{(scen_p-base_p)*100:.2f} percentage points")
    st.subheader("Baseline vs scenario")
    compare = pd.concat([base.assign(Profile="Baseline"), scenario.assign(Profile="Scenario")], ignore_index=True)
    st.dataframe(compare[["Profile"] + base_features], use_container_width=True)

else:
    st.header("Portfolio risk table")
    st.write("Customers are sorted from highest predicted churn risk to lowest.")
    selected_band = st.multiselect("Filter risk bands", ["High", "Medium", "Low"], default=["High", "Medium"])
    table = risk_scores[risk_scores["RiskBand"].isin(selected_band)].copy()
    st.dataframe(table.head(500), use_container_width=True)
    st.download_button("Download filtered risk scores", table.to_csv(index=False), file_name="filtered_churn_risk_scores.csv", mime="text/csv")
