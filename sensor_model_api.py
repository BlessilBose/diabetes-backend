from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

with open("sensor_diabetes_model.pkl", "rb") as f:
    model = pickle.load(f)


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


@app.route("/predict_sensor_risk", methods=["POST"])
def predict_sensor_risk():
    try:
        data = request.get_json()

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
            "spo2": spo2,
        }])

        print("\n--- Incoming Prediction Request ---")
        print("Raw JSON:", data)
        print("Model Input:", input_data.to_dict(orient="records"))

        prediction = int(model.predict(input_data)[0])

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_data)[0]
            probability = float(proba[1])
            print("Predict_proba:", proba.tolist())
        else:
            probability = 1.0 if prediction == 1 else 0.0

        print("Predicted class:", prediction)
        print("Probability for class 1:", probability)

        risk_level, message = build_message(probability)

        return jsonify({
            "prediction": prediction,
            "riskScore": round(probability, 4),
            "riskLevel": risk_level,
            "message": message,
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)