from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import time

# ---------------------------
# App setup
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder="static", static_url_path="/static")
CORS(app)

# ---------------------------
# Load trained model
# ---------------------------
model = joblib.load(os.path.join(BASE_DIR, "model", "rf_model.pkl"))

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
    try:
        data = request.json

        # ---------------------------
        # Input dataframe
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
        # Prediction
        # ---------------------------
        prob = model.predict_proba(input_df)[0][1]

        prediction = "⚠️ ASD traits detected" if prob >= 0.5 else "✅ No ASD traits detected"

        if prob < 0.4:
            risk = "Low Risk of ASD traits"
        elif prob < 0.7:
            risk = "Moderate Risk of ASD traits"
        else:
            risk = "High Risk of ASD traits"

        # ---------------------------
        # Rule-based explanation
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
        # SHAP WATERFALL (FINAL FIX)
        # ---------------------------
        shap_url = None

        try:
            explainer = shap.Explainer(model)
            shap_values = explainer(input_df)

            print("SHAP SHAPE:", shap_values.values.shape)

            os.makedirs(app.static_folder, exist_ok=True)
            filename = os.path.join(app.static_folder, f"shap_{int(time.time())}.png")

            plt.figure(figsize=(8, 5))

            # ✅ Correct extraction
            values = shap_values.values[0, :, 1]        # class 1 (ASD)
            base_value = shap_values.base_values[0, 1]  # class 1 base

            explanation = shap.Explanation(
                values=values,
                base_values=base_value,
                data=input_df.iloc[0],
                feature_names=input_df.columns
            )

            shap.plots.waterfall(
                explanation,
                max_display=8,
                show=False
            )

            plt.savefig(filename, bbox_inches='tight', dpi=300)
            plt.close()

            # Use relative URL or construct from request
            shap_url = f"/static/{os.path.basename(filename)}"

        except Exception as e:
            print("SHAP ERROR:", e)
            shap_url = None

        # ---------------------------
        # Response
        # ---------------------------
        return jsonify({
            "prediction": prediction,
            "probability": round(float(prob), 3),
            "risk": risk,
            "key_factors": reasons,
            "shap_image": shap_url
        })

    except Exception as e:
        print("MAIN ERROR:", e)
        return jsonify({"error": str(e)}), 500


# ---------------------------
# Run server
# ---------------------------
if __name__ == "__main__":
    import os
    debug_mode = os.getenv("FLASK_ENV", "production") == "development"
    app.run(debug=debug_mode, host="0.0.0.0", port=5000)