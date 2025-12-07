from flask import Blueprint, request, jsonify
import joblib
import os
import pandas as pd

random_forest_route = Blueprint("random_forest", __name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "machineLearning", "Model", "rf" , "Model.pkl"))
ACC_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "machineLearning", "Model", "rf" , "Acc.pkl"))
EVALUATION_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "machineLearning", "Model", "rf" , "Evaluation.pkl"))

# LOAD MODEL
try:
    model = joblib.load(MODEL_PATH)
    acc = joblib.load(ACC_PATH)
    evaluation = joblib.load(EVALUATION_PATH)
except Exception as e:
    raise Exception(f"Gagal load model RF: {e}")


# KITA BIKIN ROUTINGAN
@random_forest_route.route("/predict-rf", methods=["POST"])
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

        prediksi = model.predict(fitur)
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
                "recall" : f"{round(evaluation['recall'],4)}",
                "f1_score": f"{round(evaluation['f1_score'],4)}"
            }
        })
    except Exception as e:
        return jsonify({"kode" : 500, "error" : str(e)})
