# 🏦 AI-Based Fraud Detection Simulator

An interactive AI-powered web application that simulates real-world banking fraud detection using Machine Learning and SHAP explainability.

Built with:
- Streamlit (Frontend + Backend)
- XGBoost (ML Model)
- SHAP (Explainable AI)
- Python

---

## 🚀 Project Overview

This project simulates how banks detect fraudulent transactions using machine learning.

Users can:

- Enter transaction details
- Get fraud probability score
- View risk level (Low / Medium / High)
- See SHAP-based feature explanations
- Learn about fraud awareness through a Help Center
- Take a small fraud awareness quiz

---

## 🧠 How It Works

1. User enters transaction details.
2. Inputs are mapped into model features.
3. XGBoost model predicts fraud probability.
4. Risk level is displayed using a visual meter.
5. SHAP explains which features influenced the decision.

---

## 📊 Input Features

| User Input | What It Represents |
|------------|-------------------|
| Transaction Amount | Value of transaction |
| Transaction Time | Morning / Afternoon / Evening / Night |
| International | Whether transaction is cross-border |
| Online Purchase | E-commerce activity |
| Multiple Transactions | Rapid activity in short time |
| High-Risk Merchant | Suspicious merchant category |

---

## 📈 Risk Output

- 🟢 Low Risk (<20%)
- 🟠 Medium Risk (20–59%)
- 🔴 High Risk (60%+)

---

## 📊 Explainable AI (SHAP)

The app uses SHAP (SHapley Additive exPlanations) to show:

- Which features increased fraud risk
- Which features reduced fraud risk
- Contribution breakdown in a bar chart

This improves transparency and trust in AI decisions.

---

## 🗂️ Project Structure

```
AI-Fraud-Detection/
│
├── app.py
├── models/
│   └── xgb_fraud_model.pkl
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation (Local Setup)

### 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/AI-Fraud-Detection.git
cd AI-Fraud-Detection
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run App

```bash
streamlit run app.py
```

---

## 📦 Requirements

```
streamlit
xgboost==1.7.6
shap==0.44.1
matplotlib
pandas
numpy
```

---

## 🌍 Deployment

This project is deployed using:

- Streamlit Community Cloud

To deploy:
1. Push code to GitHub
2. Connect repository in Streamlit Cloud
3. Select `app.py`
4. Deploy

---

## 🎯 Skills Demonstrated

- Machine Learning Deployment
- XGBoost Modeling
- Explainable AI (SHAP)
- Streamlit App Development
- Feature Engineering
- Risk Classification Logic

---

## 🔐 Real-World Application

This project simulates systems used in:

- Banking fraud detection
- Payment gateways
- Fintech risk engines
- Credit card monitoring systems

---

## 👨‍💻 Author

Developed as a Machine Learning + AI Deployment project.

---

## 📜 License

This project is for educational and portfolio purposes.