from flask import Blueprint, request, jsonify
import numpy as np
import pandas as pd
import joblib
import os

id3_route = Blueprint("id3_route", __name__)

# ===== PATH MODEL =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "..", "machineLearning", "Model", "ID3Wine.pkl")
ACCURACY_PATH = os.path.join(BASE_DIR, "..", "machineLearning", "Model", "ACCID3Wine.pkl")
METRICS_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "machineLearning", "Model", "Evaluation_ID3.pkl"))


MODEL_PATH = os.path.normpath(MODEL_PATH)
ACCURACY_PATH = os.path.normpath(ACCURACY_PATH)

# ===== LOAD MODEL & ACCURACY =====

try:
    model = joblib.load(MODEL_PATH)
    accuracy = joblib.load(ACCURACY_PATH)
    # Metrics
    metrics = joblib.load(METRICS_PATH)
except Exception as e:
    raise Exception(f"Gagal load model NB: {e}")


@id3_route.route("/predict-id3", methods=["POST"])
def predict_id3():
    try:
        data = request.json
        # fitur sesuai urutan training ID3 kamu
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

        # ===== prediksi =====
        pred = model.predict(fitur)[0]

        def predCategorical(q):
            if q <= 5:
                return "low"
            elif q == 6:
                return "medium"
            else:
                return "high"


        kategori = predCategorical(pred)


        # pastikan akurasi bisa dijadikan float
        try:
            acc_value = float(accuracy)
        except:
            acc_value = accuracy.get("accuracy") if isinstance(accuracy, dict) else None

        return jsonify({
                "kode": 200,
                "status": "success",
                "prediksi": int(pred),
                "kelas": f"Wine class {kategori}",
                "akurasi_model":acc_value,
                "precision": f"{round(metrics['precision'],4)}",
                "recall": f"{round(metrics['recall'],4)}",
                "f1_score": f"{round(metrics['f1_score'],4)}",
            })

    except Exception as e:
        return jsonify({"kode": 500, "error": str(e)})
