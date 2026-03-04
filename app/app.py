import streamlit as st
import numpy as np
import pickle
import os
import shap
import matplotlib.pyplot as plt
import pandas as pd

# =============================
# Page Config
# =============================
st.set_page_config(page_title="AI Fraud Detection Simulator", layout="centered")

# =============================
# Navigation State
# =============================
if "page" not in st.session_state:
    st.session_state.page = "main"

# =============================
# Header with Help Button
# =============================
col1, col2 = st.columns([6, 1])
with col1:
    st.title("🏦 AI-Based Fraud Detection Simulator")
with col2:
    if st.button("❓ Help"):
        st.session_state.page = "help"

# ==========================================================
# ======================= HELP PAGE ========================
# ==========================================================
if st.session_state.page == "help":

    st.markdown("## 🛡️ Fraud Awareness Center")

    st.markdown("""
    Financial fraud can cause:
    - Unauthorized withdrawals  
    - Account freezing  
    - Identity theft  
    - Emotional and financial stress  

    ### 🚨 What To Do If You Suspect Fraud
    1. Immediately block your card  
    2. Contact your bank  
    3. Change passwords  
    4. Enable 2FA  
    5. Monitor statements  
    """)

    st.divider()

    st.subheader("❓ Frequently Asked Questions")

    with st.expander("What increases fraud risk?"):
        st.write("Night transactions, international activity, rapid transactions, high-risk merchants.")

    with st.expander("What is a false positive?"):
        st.write("A genuine transaction incorrectly flagged as fraud.")

    with st.expander("What is a false negative?"):
        st.write("Fraud transaction incorrectly marked as safe.")

    st.divider()

    # Quiz
    st.subheader("🧠 Quick Fraud Quiz")

    score = 0
    q1 = st.radio(
        "Which scenario is most suspicious?",
        [
            "Buying groceries at 6 PM",
            "Multiple international transactions at midnight",
            "Monthly Netflix subscription"
        ]
    )

    if q1 == "Multiple international transactions at midnight":
        score += 1

    if st.button("Submit Quiz"):
        st.success(f"Your Score: {score}/1")

    if st.button("⬅ Back to App"):
        st.session_state.page = "main"

# ==========================================================
# ======================= MAIN APP =========================
# ==========================================================
else:

    st.markdown("Simulating Real-World Banking Fraud Detection")
    st.divider()

    # =============================
    # Load Model (Cached Properly)
    # =============================
    @st.cache_resource
    def load_model():
        model_path = os.path.join("models", "xgb_fraud_model.pkl")
        with open(model_path, "rb") as f:
            return pickle.load(f)

    model = load_model()

    # =============================
    # Load SHAP Explainer (NO ARGUMENTS)
    # =============================
    @st.cache_resource
    def load_explainer():
        return shap.TreeExplainer(model)

    explainer = load_explainer()

    # =============================
    # User Inputs
    # =============================
    st.subheader("📝 Enter Transaction Details")

    amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=100.0)

    time_of_day = st.selectbox(
        "Transaction Time",
        ["Morning", "Afternoon", "Evening", "Night"]
    )

    international = st.selectbox("International Transaction?", ["No", "Yes"])
    online = st.selectbox("Online Purchase?", ["No", "Yes"])
    multiple_txn = st.selectbox("Multiple Transactions in Last Hour?", ["No", "Yes"])
    high_risk_merchant = st.selectbox("High-Risk Merchant Category?", ["No", "Yes"])

    # =============================
    # Feature Mapping
    # =============================
    def map_to_model_features():
        v17 = 0
        v14 = 0
        v12 = 0
        time = 30000

        if time_of_day == "Night":
            time = 80000
            v12 -= 1.5
        elif time_of_day == "Evening":
            time = 60000
        elif time_of_day == "Morning":
            time = 20000

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
        risk_percent = int(probability * 100)

        st.subheader("📊 Fraud Risk Meter")

        if risk_percent < 20:
            color = "green"
            label = "LOW RISK"
        elif risk_percent < 60:
            color = "orange"
            label = "MEDIUM RISK"
        else:
            color = "red"
            label = "HIGH RISK"

        progress_bar = st.progress(0)
        for i in range(risk_percent + 1):
            progress_bar.progress(i)

        st.markdown(
            f"<h2 style='text-align:center; color:{color};'>{risk_percent}% — {label}</h2>",
            unsafe_allow_html=True
        )

        if risk_percent >= 60:
            st.error("⚠️ Recommended Action: Block or Require OTP")
        elif risk_percent >= 20:
            st.warning("🔍 Recommended Action: Monitor Transaction")
        else:
            st.success("✅ Transaction Approved")

        # =============================
        # SHAP Explanation
        # =============================
        st.subheader("📊 Risk Contribution Breakdown")

        try:
            shap_values = explainer.shap_values(input_data)

            if isinstance(shap_values, list):
                shap_values = shap_values[1]

            shap_values = shap_values[0]

            feature_labels = [
                "Behavior Risk",
                "Pattern Risk",
                "Activity Risk",
                "Transaction Amount",
                "Transaction Time"
            ]

            contrib_df = pd.DataFrame({
                "Feature": feature_labels,
                "Impact": shap_values
            }).sort_values(by="Impact", key=abs)

            fig, ax = plt.subplots()
            colors = ["red" if x > 0 else "green" for x in contrib_df["Impact"]]
            ax.barh(contrib_df["Feature"], contrib_df["Impact"], color=colors)
            ax.set_xlabel("Impact on Fraud Risk")
            ax.set_title("Feature Contribution")

            st.pyplot(fig)
            st.markdown("### 📘 How to Read This Chart")

            st.markdown("""
            - 🔴 **Red bars** → Increase fraud risk  
            - 🟢 **Green bars** → Decrease fraud risk  

            - The **longer the bar**, the stronger the impact on the prediction.  
            - Features at the top have the **highest influence** on the model decision.  
            - Positive values push the transaction toward **Fraud**.  
            - Negative values push the transaction toward **Safe**.  

            💡 Example:
            If *International Transaction* shows a large red bar,  
            it means that factor strongly increased fraud probability.
            """)

            st.info("This explanation uses SHAP (SHapley Additive exPlanations) to improve transparency in AI decisions.")

        except Exception as e:
            st.warning(f"SHAP explanation unavailable: {e}")
