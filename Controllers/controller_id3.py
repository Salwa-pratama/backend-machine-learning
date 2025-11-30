from flask import Blueprint, request, jsonify
import numpy as np
import joblib
import os

id3_route = Blueprint("id3_route", __name__)

# ===== PATH MODEL =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "..", "machineLearning", "Model", "ID3Wine.pkl")
ACCURACY_PATH = os.path.join(BASE_DIR, "..", "machineLearning", "Model", "ACCID3Wine.pkl")

MODEL_PATH = os.path.normpath(MODEL_PATH)
ACCURACY_PATH = os.path.normpath(ACCURACY_PATH)

# ===== LOAD MODEL & ACCURACY =====
model = joblib.load(MODEL_PATH)
accuracy = joblib.load(ACCURACY_PATH)

@id3_route.route("/predict-id3", methods=["POST"])
def predict_id3():
    try:
        data = request.json

        # fitur sesuai urutan training ID3 kamu
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

        # ===== prediksi =====
        pred = model.predict(fitur)[0]

        # pastikan akurasi bisa dijadikan float
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
