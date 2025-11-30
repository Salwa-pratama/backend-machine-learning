from flask import Blueprint, request, jsonify
import numpy as np
import joblib
import os

naive_route = Blueprint("naive_route", __name__)

# ====== FIX PATH ABSOLUTE UNTUK VERCEL ======
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "machineLearning", "Model", "NaiveBayesWine.pkl"))
SCALER_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "machineLearning", "Model", "ScalerWinenb.pkl"))
ACCURACY_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "machineLearning", "Model", "AccuracyWinenb.pkl"))

# Debugging optional:
# print("MODEL_PATH:", MODEL_PATH)
# print("SCALER_PATH:", SCALER_PATH)
# print("ACCURACY_PATH:", ACCURACY_PATH)

# ====== LOAD MODEL & SCALER & ACCURACY ======
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    accuracy = joblib.load(ACCURACY_PATH)
except Exception as e:
    raise Exception(f"Gagal load model NB: {e}")

# ====== ROUTE ======
@naive_route.route("/predict-nb", methods=["POST"])
def predict_naive():
    try:
        data = request.json

        fitur = np.array([
            data["fixed_acidity"],
            data["volatile_acidity"],
            data["citric_acid"],
            data["residual_sugar"],
            data["chlorides"],
            data["free_sulfur_dioxide"],
            data["total_sulfur_dioxide"],
            data["density"],
            data["ph"],
            data["sulphates"],
            data["alcohol"]
        ]).reshape(1, -1)

        # scaling
        fitur_scaled = scaler.transform(fitur)

        # prediksi
        pred = model.predict(fitur)[0]

        # akurasi
        try:
            acc_value = float(accuracy)
        except:
            acc_value = accuracy.get("accuracy") if isinstance(accuracy, dict) else None

        return jsonify({
            "kode": 200,
            "status": "success",
            "prediksi": int(pred),
            "kelas": f"Wine class {pred}",
            "akurasi_model": acc_value
        })

    except Exception as e:
        return jsonify({"kode": 500, "error": str(e)})
