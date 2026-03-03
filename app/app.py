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
# Top Right Help Button
# =============================
col1, col2 = st.columns([6, 1])
with col2:
    if st.button("❓ Help"):
        st.session_state.page = "help"

# =============================
# HELP PAGE
# =============================
if st.session_state.page == "help":

    st.title("🛡️ Fraud Awareness & Risk Education Center")

    st.markdown("""
    # Understanding Financial Fraud and Its Consequences

    Financial fraud is a serious financial crime that impacts customers,
    banks, and merchants worldwide.

    ## 💳 What Happens When Fraud Occurs?

    ### Immediate Financial Loss
    - Unauthorized money withdrawal
    - Account freezing
    - Emergency investigation

    ### Customer Impact
    - Emotional stress
    - Temporary financial instability
    - Identity misuse risk

    ### Bank Impact
    - Refund reimbursements
    - Investigation costs
    - Reputation damage

    ### Merchant Impact
    - Chargebacks
    - Revenue loss
    - Fraud monitoring expenses
    """)

    st.divider()

    st.subheader("🚨 What To Do If You Suspect Fraud")

    st.markdown("""
    1. Block your card immediately  
    2. Contact your bank  
    3. Change passwords  
    4. Enable two-factor authentication  
    5. Monitor account activity  
    """)

    st.divider()

    st.subheader("📞 Report Fraud – Contact Information")

    st.markdown("""
    **Simulated Fraud Support Center**

    📞 +1-800-999-SECURE  
    📧 fraudsupport@securebank.com  
    🌐 www.securebank.com/report-fraud  
    """)

    st.warning("This contact section is for project demonstration only.")

    st.divider()

    # =============================
    # FAQ Section
    # =============================
    st.subheader("❓ Frequently Asked Questions")

    with st.expander("What factors increase fraud risk the most?"):
        st.write("Unusual hours, international payments, high-risk merchants, and rapid transactions increase fraud risk.")

    with st.expander("What is a false positive?"):
        st.write("A legitimate transaction incorrectly flagged as fraud.")

    with st.expander("What is a false negative?"):
        st.write("A fraudulent transaction incorrectly classified as safe.")

    with st.expander("Why are small test transactions dangerous?"):
        st.write("Fraudsters test stolen cards with small amounts before large fraud attempts.")

    with st.expander("How does AI improve fraud detection?"):
        st.write("AI detects complex behavioral patterns beyond simple rule-based systems.")

    st.divider()

    # =============================
    # Fraud Awareness Quiz
    # =============================
    st.subheader("🧠 Test Your Fraud Awareness Knowledge")

    score = 0

    q1 = st.radio(
        "1️⃣ Most suspicious scenario?",
        [
            "Regular grocery purchase",
            "Multiple midnight international transactions",
            "Monthly subscription",
            "ATM withdrawal locally"
        ],
        key="q1"
    )
    if q1 == "Multiple midnight international transactions":
        score += 1

    q2 = st.radio(
        "2️⃣ If you receive an OTP you didn’t request?",
        [
            "Ignore it",
            "Share it",
            "Report immediately",
            "Use later"
        ],
        key="q2"
    )
    if q2 == "Report immediately":
        score += 1

    q3 = st.radio(
        "3️⃣ Why are small test transactions risky?",
        [
            "Harmless",
            "Test stolen cards",
            "Reduce risk",
            "Banks ignore them"
        ],
        key="q3"
    )
    if q3 == "Test stolen cards":
        score += 1

    q4 = st.radio(
        "4️⃣ What is a false negative?",
        [
            "Blocking real transaction",
            "Allowing fraud as safe",
            "System error",
            "Duplicate payment"
        ],
        key="q4"
    )
    if q4 == "Allowing fraud as safe":
        score += 1

    q5 = st.radio(
        "5️⃣ Which system detects complex fraud patterns?",
        [
            "Manual review",
            "Rule-only system",
            "AI & ML models",
            "Random check"
        ],
        key="q5"
    )
    if q5 == "AI & ML models":
        score += 1

    if st.button("📊 Submit Quiz"):
        st.subheader("🎯 Your Score")
        st.write(f"You scored **{score} out of 5**")

        if score == 5:
            st.success("Excellent fraud awareness!")
        elif score >= 3:
            st.info("Good understanding of fraud risks.")
        else:
            st.warning("Review the fraud awareness section again.")

    if st.button("⬅ Back to Dashboard"):
        st.session_state.page = "main"

# =============================
# MAIN DASHBOARD
# =============================
else:

    # Load Model
    model_path = os.path.join("models", "xgb_fraud_model.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    st.title("🏦 AI-Based Fraud Detection Simulator")
    st.markdown("Simulating Real-World Banking Fraud Analysis")
    st.divider()

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
            st.error("⚠️ Block Transaction or Require OTP")
        elif risk_percent >= 20:
            st.warning("🔍 Monitor Transaction")
        else:
            st.success("✅ Transaction Approved")

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

        contrib_df = pd.DataFrame({
            "Feature": feature_labels,
            "Impact": shap_values
        }).sort_values(by="Impact", key=abs)

        fig, ax = plt.subplots()
        colors = ["red" if x > 0 else "green" for x in contrib_df["Impact"]]
        ax.barh(contrib_df["Feature"], contrib_df["Impact"], color=colors)
        ax.set_xlabel("Impact on Fraud Risk")
        ax.set_title("Feature Contribution to Risk Score")

        st.pyplot(fig)

        st.markdown("---")