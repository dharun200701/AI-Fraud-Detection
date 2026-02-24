import streamlit as st
import numpy as np
import pickle
import os
import shap
import matplotlib.pyplot as plt

# =============================
# Load Model
# =============================
model_path = os.path.join("models", "xgb_fraud_model.pkl")

with open(model_path, "rb") as f:
    model = pickle.load(f)

THRESHOLD = 0.3

st.set_page_config(page_title="AI Fraud Detection Simulator", layout="centered")

st.title("🏦 AI-Based Fraud Detection Simulator")
st.markdown("Simulating Real-World Banking Fraud Analysis")

st.divider()

st.subheader("📝 Enter Transaction Details")

# =============================
# User-Friendly Inputs
# =============================

amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=100.0)

time_of_day = st.selectbox(
    "Transaction Time",
    ["Morning", "Afternoon", "Evening", "Night"]
)

international = st.selectbox(
    "Is this an International Transaction?",
    ["No", "Yes"]
)

online = st.selectbox(
    "Is this an Online Purchase?",
    ["No", "Yes"]
)

multiple_txn = st.selectbox(
    "Multiple Transactions in Last Hour?",
    ["No", "Yes"]
)

high_risk_merchant = st.selectbox(
    "High-Risk Merchant Category?",
    ["No", "Yes"]
)

# =============================
# Feature Mapping Logic
# =============================

def map_to_model_features():
    # Base values
    v17 = 0
    v14 = 0
    v12 = 0
    time = 30000

    # Time mapping
    if time_of_day == "Night":
        time = 80000
        v12 -= 1.5
    elif time_of_day == "Evening":
        time = 60000
    elif time_of_day == "Morning":
        time = 20000

    # Risk factors
    if international == "Yes":
        v14 -= 2.5
    if online == "Yes":
        v17 -= 1.8
    if multiple_txn == "Yes":
        v12 -= 2.0
    if high_risk_merchant == "Yes":
        v17 -= 2.5
        v14 -= 1.5

    return np.array([[v17, v14, v12, amount, time]])

# =============================
# Prediction
# =============================

if st.button("🔍 Analyze Transaction"):

    input_data = map_to_model_features()

    probability = model.predict_proba(input_data)[0][1]

    probability = model.predict_proba(input_data)[0][1]

    # =============================
    # Animated Risk Gauge
    # =============================

    st.subheader("📊 Fraud Risk Meter")

    risk_percent = int(probability * 100)

    if risk_percent < 20:
        color = "green"
        risk_label = "LOW RISK"
    elif risk_percent < 60:
        color = "orange"
        risk_label = "MEDIUM RISK"
    else:
        color = "red"
        risk_label = "HIGH RISK"

    progress_bar = st.progress(0)

    for i in range(risk_percent + 1):
        progress_bar.progress(i)

    st.markdown(
        f"""
        <h2 style='text-align: center; color:{color};'>
        {risk_percent}% — {risk_label}
        </h2>
        """,
        unsafe_allow_html=True
    )

    # Recommended Action
    if risk_percent >= 60:
        st.error("⚠️ Recommended Action: Block Transaction or Require OTP")
    elif risk_percent >= 20:
        st.warning("🔍 Recommended Action: Monitor Transaction")
    else:
        st.success("✅ Transaction Approved")

   # =============================
    # Simple Risk Contribution Chart
    # =============================

    st.subheader("📊 Risk Contribution Breakdown")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_data)[0]

    feature_labels = [
        "Behavior Risk",
        "Pattern Risk",
        "Activity Risk",
        "Transaction Amount",
        "Transaction Time"
    ]

    # Create dataframe
    import pandas as pd
    contrib_df = pd.DataFrame({
        "Feature": feature_labels,
        "Impact": shap_values
    })

    # Sort by impact magnitude
    contrib_df = contrib_df.sort_values(by="Impact", key=abs, ascending=True)

    fig, ax = plt.subplots()

    colors = ["red" if x > 0 else "green" for x in contrib_df["Impact"]]

    ax.barh(contrib_df["Feature"], contrib_df["Impact"], color=colors)

    ax.set_xlabel("Impact on Fraud Risk")
    ax.set_title("Feature Contribution to Risk Score")

    st.pyplot(fig)

    st.subheader("🧠 Detailed Risk Explanation")

    # Identify strongest positive and negative contributors
    top_positive = contrib_df[contrib_df["Impact"] > 0]
    top_negative = contrib_df[contrib_df["Impact"] < 0]

    st.markdown("### 📌 Why This Risk Score Was Assigned")

    if not top_positive.empty:
        st.markdown("**🔺 Factors Increasing Risk:**")
        for _, row in top_positive.iterrows():
            st.write(
                f"- {row['Feature']} showed unusual activity, increasing fraud probability."
            )

    if not top_negative.empty:
        st.markdown("**🔻 Factors Reducing Risk:**")
        for _, row in top_negative.iterrows():
            st.write(
                f"- {row['Feature']} appeared normal and reduced overall fraud risk."
            )

    st.markdown("---")