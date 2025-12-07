from flask import Blueprint, request, jsonify
import joblib
import os
import pandas as pd

xgboost_route = Blueprint("xgboost", __name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "machineLearning", "Model","xgboost", "xgboost_model.pkl"))
ACC_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "machineLearning", "Model","xgboost", "ACCXGBOOST.pkl"))
EVALUATION_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "machineLearning", "Model","xgboost", "Evaluation_xgboost.pkl"))

# LOAD MODEL
try:
    model = joblib.load(MODEL_PATH)
    acc = joblib.load(ACC_PATH)
    metrics = joblib.load(EVALUATION_PATH)
except Exception as e:
    raise Exception(f"Gagal load model XGBoost: {e}")


@xgboost_route.route("/predict-xgb-test", methods=["GET"])
def test_xgb():
    return jsonify({
        "kode": 200,
        "status": "hadir",
        "message": "Koneksi API XGBoost Berhasil"
    })


@xgboost_route.route("/predict-xgb", methods=["POST"])
def predict_xgb():  # ✅ Nama fungsi disesuaikan
    try:
        data = request.json

        # ✅ Validasi: pastikan semua field ada
        required_fields = [
            "pregnancies", "glucose", "bloodpressure", "skinThickness",
            "insulin", "bmi", "diabetespedigreefunction", "age"
        ]
        for field in required_fields:
            if field not in data:
                return jsonify({
                    "kode": 400,
                    "error": f"Field '{field}' wajib diisi"
                }), 400
        fitur = pd.DataFrame([{
            "Pregnancies": int(float(data["pregnancies"])),  # handle "4.0" → 4
            "Glucose": float(data["glucose"]),
            "BloodPressure": float(data["bloodpressure"]),
            "SkinThickness": float(data["skinThickness"]),
            "Insulin": float(data["insulin"]),
            "BMI": float(data["bmi"]),
            "DiabetesPedigreeFunction": float(data["diabetespedigreefunction"]),
            "Age": int(float(data["age"])),
        }])

        # Lakukan prediksi
        prediksi = model.predict(fitur)
        kategorical = "Positif" if prediksi[0] == 1 else "Negatif"

        # Ambil akurasi
        try:
            acc_value = float(acc)
        except:
            acc_value = acc.get("accuracy") if isinstance(acc, dict) else None

        return jsonify({
            "kode": 200,
            "status": "success",
            "prediksi": kategorical,
            "performa_model": {
                "acc": acc_value,
                "precision" : f"{round(metrics['precision'],4)}",
                "recall" : f"{round(metrics['recall'],4)}",
                "f1_score": f"{round(metrics['f1_score'],4)}"
            }
        })

    except (ValueError, TypeError) as e:
        # Error saat konversi angka (misal: "abc" → float)
        return jsonify({
            "kode": 400,
            "error": f"Data input tidak valid: {str(e)}"
        }), 400

    except KeyError as e:
        # Field tidak ditemukan di JSON
        return jsonify({
            "kode": 400,
            "error": f"Field tidak ditemukan: {str(e)}"
        }), 400

    except Exception as e:
        # Error lain (model, dll)
        return jsonify({
            "kode": 500,
            "error": f"Kesalahan server: {str(e)}"
        }), 500
