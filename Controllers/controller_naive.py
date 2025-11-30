from flask import Blueprint, request, jsonify
import numpy as np
import joblib
import os
import pandas as pd

naive_route = Blueprint("naive_route", __name__)

# ====== FIX PATH ABSOLUTE UNTUK VERCEL ======
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "machineLearning", "Model", "NaiveBayesWine.pkl"))
SCALER_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "machineLearning", "Model", "ScalerWinenb.pkl"))
ACCURACY_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "machineLearning", "Model", "AccuracyWinenb.pkl"))
METRICS_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "machineLearning", "Model", "Evaluation_NB.pkl"))
# Debugging optional:
# print("MODEL_PATH:", MODEL_PATH)
# print("SCALER_PATH:", SCALER_PATH)
# print("ACCURACY_PATH:", ACCURACY_PATH)

# ====== LOAD MODEL & SCALER & ACCURACY ======
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    accuracy = joblib.load(ACCURACY_PATH)
    # Metrics
    metrics = joblib.load(METRICS_PATH)
except Exception as e:
    raise Exception(f"Gagal load model NB: {e}")

# ====== ROUTE ======
@naive_route.route("/predict-nb", methods=["POST"])
def predict_naive():
    try:
        data = request.json

        fitur = pd.DataFrame([{
            "fixed acidity": data["fixed_acidity"],
            "volatile acidity": data["volatile_acidity"],
            "citric acid": data["citric_acid"],
            "residual sugar": data["residual_sugar"],
            "chlorides": data["chlorides"],
            "free sulfur dioxide": data["free_sulfur_dioxide"],
            "total sulfur dioxide": data["total_sulfur_dioxide"],
            "density": data["density"],
            "pH": data["ph"],
            "sulphates": data["sulphates"],
            "alcohol": data["alcohol"]
        }])

        # scaling
        fitur_scaled = scaler.transform(fitur)

        # prediksi
        pred = model.predict(fitur_scaled)[0]

        def predCategorical(q):
            if q <= 5:
                return "low"
            elif q == 6:
                return "medium"
            else:
                return "high"


        kategori = predCategorical(pred)


        # akurasi
        try:
            acc_value = float(accuracy)
        except:

            acc_value = accuracy.get("accuracy") if isinstance(accuracy, dict) else None


        return jsonify({
            "kode": 200,
            "status": "success",
            "prediksi": int(pred),
            "kelas": f"Wine class {kategori}",
            "akurasi_model": acc_value,
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
        })

    except Exception as e:
        return jsonify({"kode": 500, "error": str(e)})
