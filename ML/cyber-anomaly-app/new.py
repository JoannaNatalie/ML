import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report
import joblib

# Load dataset
df = pd.read_csv("advanced_cybersecurity_data.csv")  # Make sure it's in same folder

# Preprocess
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df['Hour'] = df['Timestamp'].dt.hour

le_ip = LabelEncoder()
le_ua = LabelEncoder()
le_loc = LabelEncoder()

df['IP_Code'] = le_ip.fit_transform(df['IP_Address'])
df['UA_Code'] = le_ua.fit_transform(df['User_Agent'])
df['Loc_Code'] = le_loc.fit_transform(df['Location'])

features = df[['Hour', 'IP_Code', 'UA_Code', 'Loc_Code']]
model = IsolationForest(contamination=0.05, random_state=42)
df['Predicted_Anomaly'] = model.fit_predict(features)
df['Predicted_Anomaly'] = df['Predicted_Anomaly'].map({1: 0, -1: 1})

print(classification_report(df['Anomaly_Flag'], df['Predicted_Anomaly']))

# Save model and encoders
joblib.dump(model, "anomaly_model.pkl")
joblib.dump(le_ip, "le_ip.pkl")
joblib.dump(le_ua, "le_ua.pkl")
joblib.dump(le_loc, "le_loc.pkl")
