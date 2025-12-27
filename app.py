from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib

# ---------------------------
# App setup
# ---------------------------
app = Flask(__name__)
CORS(app)

# ---------------------------
# Load trained model
# ---------------------------
model = joblib.load("model/rf_model.pkl")

# ---------------------------
# Home route
# ---------------------------
@app.route("/")
def home():
    return "ASD Screening Backend Running"

# ---------------------------
# Prediction route
# ---------------------------
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    # ---------------------------
    # Input dataframe (order matters)
    # ---------------------------
    input_df = pd.DataFrame([{
        "A1": int(data["A1"]),
        "A2": int(data["A2"]),
        "A3": int(data["A3"]),
        "A4": int(data["A4"]),
        "A5": int(data["A5"]),
        "A6": int(data["A6"]),
        "A7": int(data["A7"]),
        "Age_Mons": int(data["Age_Mons"]),
        "Sex": int(data["Sex"]),
        "Jaundice": int(data["Jaundice"]),
        "Family_mem_with_ASD": int(data["Family_mem_with_ASD"])
    }])

    # ---------------------------
    # Model probability
    # ---------------------------
    prob = model.predict_proba(input_df)[0][1]

    # ---------------------------
    # Binary prediction
    # ---------------------------
    if prob >= 0.5:
        prediction = "⚠️ ASD traits detected"
    else:
        prediction = "✅ No ASD traits detected"

    # ---------------------------
    # Severity level
    # ---------------------------
    if prob < 0.33:
        severity = "Low / No ASD traits"
    elif prob < 0.66:
        severity = "Mild ASD traits"
    elif prob < 0.85:
        severity = "Moderate ASD traits"
    else:
        severity = "Severe ASD traits"

    # ---------------------------
    # Simple rule-based explanation
    # ---------------------------
    reasons = []

    if input_df["A1"].values[0] == 1:
        reasons.append("Does not respond to name (A1)")
    if input_df["A2"].values[0] == 1:
        reasons.append("Poor eye contact (A2)")
    if input_df["A3"].values[0] == 1:
        reasons.append("Does not point to objects (A3)")
    if input_df["A4"].values[0] == 1:
        reasons.append("No pretend or imaginative play (A4)")
    if input_df["A5"].values[0] == 1:
        reasons.append("Does not follow gaze (A5)")
    if input_df["A6"].values[0] == 1:
        reasons.append("Difficulty understanding speech (A6)")
    if input_df["A7"].values[0] == 1:
        reasons.append("Limited use of gestures (A7)")
    if input_df["Family_mem_with_ASD"].values[0] == 1:
        reasons.append("Family history of ASD")

    # ---------------------------
    # JSON response
    # ---------------------------
    return jsonify({
        "prediction": prediction,
        "probability": round(float(prob), 3),
        "severity": severity,
        "key_factors": reasons
    })

# ---------------------------
# Run server
# ---------------------------
if __name__ == "__main__":
    app.run(debug=True)
