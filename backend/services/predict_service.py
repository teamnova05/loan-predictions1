# backend/services/predict_service.py
from models.model_loader import ModelLoader
from config.db import get_db
from datetime import datetime
import os

class PredictService:
    def __init__(self):
        self.loader = ModelLoader(fraud=False)
        self.db = get_db()

    def predict(self, data: dict, user_id=None):
        # Run model (or dummy response if model not loaded)
        result = self.loader.predict(data)

        # Build record to persist
        doc = {
            "user_id": user_id,
            "input": data,
            "prediction": result,
            "status": result.get("eligibility"),
            "model_version": os.getenv("MODEL_VERSION", "v0"),
            "created_at": datetime.utcnow()
        }

        # Try to persist, but do not break prediction on DB errors
        try:
            self.db["loan_requests"].insert_one(doc)
        except Exception as e:
            # log but return result
            print("Warning: failed to save loan_request to DB:", e)

        return result
