from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# --------------------------------------------------
# Load Random Forest model (relative path)
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "RandomForest.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# --------------------------------------------------
# Prediction Route
# --------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Extract input features
        N = float(data["N"])
        P = float(data["P"])
        K = float(data["K"])
        temperature = float(data["temperature"])
        humidity = float(data["humidity"])
        ph = float(data["ph"])

        # Prepare input for model
        input_data = np.array([[N, P, K, temperature, humidity, ph]])

        # Make prediction
        prediction = model.predict(input_data)[0]

        return jsonify({
            "prediction": prediction
        })

    except KeyError as e:
        return jsonify({
            "error": f"Missing field: {str(e)}"
        }), 400

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 400

# --------------------------------------------------
# Run Server
# --------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
