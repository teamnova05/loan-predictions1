from flask import Blueprint, request, jsonify
import joblib
import pandas as pd
import os
import traceback
import csv
import time

ml_bp = Blueprint("ml_predict", __name__)

# ---------- Resolve model path ----------
ROUTES_DIR = os.path.dirname(__file__)
BACKEND_DIR = os.path.dirname(ROUTES_DIR)
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)

MODEL_PATH = os.path.join(PROJECT_ROOT, "ml", "models", "rf_pipeline.joblib")
LOG_PATH = os.path.join(PROJECT_ROOT, "ml", "models", "prediction_log.csv")

# ---------- Load Model ----------
model = None
try:
    model = joblib.load(MODEL_PATH)
    print(f"[ml_predict] ML model loaded successfully from: {MODEL_PATH}")
except Exception as e:
    print(f"[ml_predict] Error loading ML model from {MODEL_PATH}: {e}")
    print(traceback.format_exc())


# ---------- Prediction Endpoint ----------
@ml_bp.route("/predict-loan-default", methods=["POST"])
def predict_loan_default():
    try:
        if model is None:
            return jsonify({"error": "Model not loaded on server"}), 500

        # -------- Get payload --------
        payload = request.get_json()
        if not payload or "records" not in payload:
            return jsonify({"error": "Invalid input. Expected JSON with key 'records'."}), 400

        records = payload["records"]
        df = pd.DataFrame(records)

        # -------- Validation: Required columns --------
        required_cols = [
            "person_age", "person_gender", "person_education", "person_income",
            "person_emp_exp", "person_home_ownership", "loan_amnt",
            "loan_intent", "loan_int_rate", "loan_percent_income",
            "cb_person_cred_hist_length", "credit_score",
            "previous_loan_defaults_on_file"
        ]

        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            return jsonify({
                "error": "Missing required fields.",
                "missing_columns": missing
            }), 400

        # -------- Prediction --------
        preds = model.predict(df).tolist()
        probs = model.predict_proba(df)[:, 1].tolist()

        # -------- Threshold & Labeling --------
        threshold = float(request.args.get("threshold", 0.5))
        labels = ["default" if p >= threshold else "no_default" for p in probs]

        # -------- Logging --------
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        header = ["timestamp", "probability", "prediction", "label"]

        file_exists = os.path.exists(LOG_PATH)

        with open(LOG_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(header)

            for p, pred, lbl in zip(probs, preds, labels):
                writer.writerow([timestamp, p, pred, lbl])

        # -------- Response --------
        return jsonify({
            "threshold_used": threshold,
            "predictions": preds,
            "probabilities": probs,
            "labels": labels
        }), 200

    except Exception as e:
        return jsonify({
            "error": str(e),
            "trace": traceback.format_exc()
        }), 500
