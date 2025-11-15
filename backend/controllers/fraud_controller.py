from flask import jsonify
from services.fraud_service import FraudService

_service = FraudService()

def detect_fraud_handler(data):
    try:
        result = _service.detect(data)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error":"server_error", "message": str(e)}), 500
