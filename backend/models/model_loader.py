import os
import joblib
import numpy as np

class ModelLoader:
    def __init__(self, fraud=False):
        self.fraud = fraud
        path = os.getenv("FRAUD_MODEL_PATH" if fraud else "MODEL_PATH", "./models/loan_model.joblib")
        self.model = None
        if os.path.exists(path):
            try:
                self.model = joblib.load(path)
                print(f"Loaded model from {path}")
            except Exception as e:
                print(f"Failed to load model at {path}: {e}")
        else:
            print(f"Model not found at {path}. Using dummy responses.")

    def predict(self, input_dict: dict):
        if self.model is None:
            return {"eligibility": "Pending - model not loaded", "confidence": 0.0}
        X = self._dict_to_features(input_dict)
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba([X])[0]
            pred = int(self.model.predict([X])[0])
            conf = float(proba[1]) if len(proba)>1 else float(max(proba))
        else:
            pred = int(self.model.predict([X])[0])
            conf = 1.0
        return {"eligibility": "Approved" if pred==1 else "Rejected", "confidence": conf}

    def predict_fraud(self, input_dict: dict):
        if self.model is None:
            return {"is_fraud": False, "score": 0.0}
        X = self._dict_to_features(input_dict)
        if hasattr(self.model, "decision_function"):
            score = float(self.model.decision_function([X])[0])
            is_fraud = bool(self.model.predict([X])[0] == -1)
            return {"is_fraud": is_fraud, "score": score}
        else:
            pred = int(self.model.predict([X])[0])
            return {"is_fraud": bool(pred==1), "score": float(pred)}

    def _dict_to_features(self, d: dict):
        # very small placeholder feature mapping - replace with real preprocessing
        keys = ["ApplicantIncome","CoapplicantIncome","LoanAmount","Loan_Amount_Term","Credit_History"]
        vals = []
        for k in keys:
            try:
                vals.append(float(d.get(k, 0)))
            except:
                vals.append(0.0)
        return np.array(vals)
