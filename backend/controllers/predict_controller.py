# backend/controllers/predict_controller.py
from flask import jsonify, request, g
from services.predict_service import PredictService

_service = PredictService()

def predict_handler(data):
    try:
        # Try to get user id from headers or auth middleware later
        user_id = request.headers.get("X-User-Id") or getattr(g, "user", {}).get("uid", None)
        result = _service.predict(data, user_id=user_id)
        return jsonify(result), 200
    except ValueError as ve:
        return jsonify({"error":"bad_request", "message": str(ve)}), 400
    except Exception as e:
        return jsonify({"error":"server_error", "message": str(e)}), 500
