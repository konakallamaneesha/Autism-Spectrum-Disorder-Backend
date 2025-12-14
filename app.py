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

    # Create input dataframe (ORDER MATTERS)
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
    # Model prediction
    # ---------------------------
    prob = model.predict_proba(input_df)[0][1]

    prediction = (
        "⚠️ ASD traits detected"
        if prob > 0.5
        else "✅ No ASD traits detected"
    )

    # ---------------------------
    # Rule-based explanation (NO SHAP)
    # IMPORTANT: 1 = abnormal, 0 = normal
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
    # Response
    # ---------------------------
    return jsonify({
        "prediction": prediction,
        "probability": round(float(prob), 3),
        "key_factors": reasons
    })

# ---------------------------
# Run server
# ---------------------------
if __name__ == "__main__":
    app.run(debug=True)
