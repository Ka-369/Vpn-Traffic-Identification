import os
import pandas as pd
import numpy as np
import joblib
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_NAME = "consolidated_traffic_data.csv"
FULL_PATH = os.path.join(BASE_DIR, FILE_NAME)

print("Loading dataset...")
df = pd.read_csv(FULL_PATH)

target_col = df.columns[-1]
df[target_col] = df[target_col].apply(lambda x: 'VPN' if 'VPN' in str(x) else 'Non-VPN')

# THE LIVE MASTER ALIGNMENT: Match the exact 23 features expected by your live sniffer
EXPECTED_FEATURES = [
    'duration', 'total_fiat', 'total_biat', 'min_fiat', 'min_biat',
    'max_fiat', 'max_biat', 'mean_fiat', 'mean_biat', 'flowPktsPerSecond',
    'flowBytesPerSecond', 'min_flowiat', 'max_flowiat', 'mean_flowiat',
    'std_flowiat', 'min_active', 'mean_active', 'max_active', 'std_active',
    'min_idle', 'mean_idle', 'max_idle', 'std_idle'
]

# Ensure all expected features actually exist in your CSV dataset
for col in EXPECTED_FEATURES:
    if col not in df.columns:
        df[col] = 0.0

X = df[EXPECTED_FEATURES]
y = df[target_col]

# Handle infinite or missing numbers safely using median values
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median())

# Encode 'VPN' / 'Non-VPN' into 1s and 0s
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Master 23-Feature Random Forest Classifier...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Quick validation check to see performance
y_pred = model.predict(X_test)
print(f"Validation Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# Save the master binary assets out to your directory
joblib.dump(model, os.path.join(BASE_DIR, 'vpn_rf_model.pkl'))
joblib.dump(label_encoder, os.path.join(BASE_DIR, 'label_encoder.pkl'))
print("[SUCCESS] New Master 23-Feature Model saved successfully!")

print("VPN samples in test set:", sum(y_test == label_encoder.transform(['VPN'])[0]))
print("VPN correctly predicted:", sum((y_pred == label_encoder.transform(['VPN'])[0]) & (y_test == label_encoder.transform(['VPN'])[0])))
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))