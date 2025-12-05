from flask import Blueprint, request, jsonify
import joblib
import os
import pandas as pd

neural_network_route = Blueprint("neural_network", __name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "machineLearning", "Model", "nn" , "Model.pkl"))
ACC_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "machineLearning", "Model", "nn" , "Acc.pkl"))
EVALUATION_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "machineLearning", "Model", "nn" , "Evaluation.pkl"))
SCALER_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "machineLearning", "Model", "nn" , "Scaler.pkl"))

# LOAD MODEL
try:
    model = joblib.load(MODEL_PATH)
    acc = joblib.load(ACC_PATH)
    evaluation = joblib.load(EVALUATION_PATH)
    scaler = joblib.load(SCALER_PATH)
except Exception as e:
    raise Exception(f"Gagal load model RF: {e}")

@neural_network_route.route("/predict-nn-test", methods=["GET"])
def test_nn ():
    return jsonify({
        "kode" : 200,
        "status" : "hadir",
        "message" : "koneksi API Berhasil"
    })


# KITA BIKIN ROUTINGAN
@neural_network_route.route("/predict-nn", methods=["POST"])
def predict_rf():
    try:
        data = request.json

        fitur = pd.DataFrame([{
            "Pregnancies": data["pregnancies"],
            "Glucose": data["glucose"],
            "BloodPressure": data["bloodpressure"],
            "SkinThickness": data["skinThickness"],
            "Insulin": data["insulin"],
            "BMI": data["bmi"],
            "DiabetesPedigreeFunction": data["diabetespedigreefunction"],
            "Age": data["age"],
        }])
        data_scaler = scaler.transform(fitur)

        prediksi = model.predict(data_scaler)
        kategorical = "Positif" if prediksi[0] == 1 else "Negatif"

        #  Akurasi
        try:
            acc_value = float(acc)
        except:
            acc_value = acc.get("accuracy") if isinstance(acc, dict) else None


        return jsonify({
            "kode": 200,
            "status"  : "success",
            "prediksi" : kategorical,
            "performa_model" : {
                "acc" : acc_value,
                "precision" : f"{round(evaluation['precision'],4)}",
                "recall" : f"{round(evaluation['precision'],4)}",
                "f1_score": f"{round(evaluation['precision'],4)}"
            }
        })
    except Exception as e:
        return jsonify({"kode" : 500, "error" : str(e)})
