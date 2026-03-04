import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import os

# Load dataset
df = pd.read_csv("creditcard.csv.zip")

# Select important features
selected_features = ['V17', 'V14', 'V12', 'Amount', 'Time']

X = df[selected_features]
y = df['Class']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = XGBClassifier(
    n_estimators=150,
    max_depth=4,
    learning_rate=0.1,
    eval_metric='logloss',
    random_state=42
)

model.fit(X_train, y_train)

# Save model (PRODUCTION SAFE)
os.makedirs("models", exist_ok=True)
model.save_model("models/xgb_fraud_model.json")

print("✅ Model retrained and saved as JSON format.")