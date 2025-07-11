from flask import Flask, request, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# Load model and encoders
model = joblib.load("anomaly_model.pkl")
le_ip = joblib.load("le_ip.pkl")
le_ua = joblib.load("le_ua.pkl")
le_loc = joblib.load("le_loc.pkl")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            df = pd.read_csv(file)
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
            df['Hour'] = df['Timestamp'].dt.hour

            df['IP_Code'] = le_ip.transform(df['IP_Address'])
            df['UA_Code'] = le_ua.transform(df['User_Agent'])
            df['Loc_Code'] = le_loc.transform(df['Location'])

            features = df[['Hour', 'IP_Code', 'UA_Code', 'Loc_Code']]
            df['Prediction'] = model.predict(features)
            df['Prediction'] = df['Prediction'].map({1: 0, -1: 1})  # 1 = normal, -1 = anomaly → map to 0/1

            # Filter anomalies
            anomalies = df[df['Prediction'] == 1].copy()

            # Add reason and risk level
            reasons = []
            risks = []
            for _, row in anomalies.iterrows():
                reason = []
                risk = "Low"

                if row['User_Agent'].lower() in ['curl', 'python-requests', 'httpclient']:
                    reason.append("Suspicious User-Agent")
                    risk = "High"

                if row['Location'] in ['Russia', 'North Korea', 'China']:
                    reason.append("Risky Location")
                    if risk != "High":
                        risk = "Medium"

                if row['Hour'] < 6 or row['Hour'] > 22:
                    reason.append("Odd login hour")
                    if risk == "Low":
                        risk = "Medium"

                reasons.append(", ".join(reason) if reason else "Unusual Pattern")
                risks.append(risk)

            anomalies['Reason'] = reasons
            anomalies['Risk_Level'] = risks

            # Only show relevant columns
            result_df = anomalies[['Timestamp', 'IP_Address', 'User_Agent', 'Location', 'Reason', 'Risk_Level']]
            return render_template("index.html", tables=[result_df.to_html(classes='data', index=False)])
    
    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)
