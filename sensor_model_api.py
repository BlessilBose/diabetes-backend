from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load trained model
with open("sensor_diabetes_model.pkl", "rb") as f:
    model = pickle.load(f)


# ----------------------------
# Helper function
# ----------------------------
def build_message(probability: float):
    if probability >= 0.80:
        return (
            "High",
            "High diabetes risk detected. Please consult a healthcare professional."
        )
    elif probability >= 0.55:
        return (
            "Medium",
            "Moderate diabetes risk detected. Maintain a healthy lifestyle and consider medical advice."
        )
    else:
        return (
            "Low",
            "Low diabetes risk detected. Continue healthy habits and regular monitoring."
        )


# ----------------------------
# Home Route
# ----------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "running",
        "message": "Sensor Diabetes Prediction API is live"
    })


# ----------------------------
# Health Check Route
# ----------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok"
    })


# ----------------------------
# Prediction Route
# ----------------------------
@app.route("/predict_sensor_risk", methods=["POST"])
def predict_sensor_risk():
    try:
        data = request.get_json()

        if not data:
            return jsonify({
                "error": "No JSON data received"
            }), 400

        # Read inputs
        age = float(data["age"])
        height_cm = float(data["height_cm"])
        weight_kg = float(data["weight_kg"])
        bmi = float(data["bmi"])
        family_history = int(data["family_history"])
        thirst = int(data["thirst"])
        urination = int(data["urination"])
        fatigue = int(data["fatigue"])
        bpm = float(data["bpm"])
        spo2 = float(data["spo2"])

        # Create dataframe
        input_data = pd.DataFrame([{
            "age": age,
            "height_cm": height_cm,
            "weight_kg": weight_kg,
            "bmi": bmi,
            "family_history": family_history,
            "thirst": thirst,
            "urination": urination,
            "fatigue": fatigue,
            "bpm": bpm,
            "spo2": spo2
        }])

        print("\n--- Prediction Request ---")
        print(input_data)

        # Prediction
        prediction = int(model.predict(input_data)[0])

        # Probability
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_data)[0]
            probability = float(proba[1])
        else:
            probability = 1.0 if prediction == 1 else 0.0

        # Risk message
        risk_level, message = build_message(probability)

        return jsonify({
            "success": True,
            "prediction": prediction,
            "riskScore": round(probability, 4),
            "riskLevel": risk_level,
            "message": message
        })

    except KeyError as e:
        return jsonify({
            "success": False,
            "error": f"Missing field: {str(e)}"
        }), 400

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400


# ----------------------------
# Run App
# ----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
