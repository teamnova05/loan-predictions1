from flask import Blueprint, request, jsonify
from controllers.fraud_controller import detect_fraud_handler

fraud_bp = Blueprint("fraud", __name__)

@fraud_bp.route("/detect-fraud", methods=["POST"])
def detect_fraud():
    data = request.get_json(force=True)
    return detect_fraud_handler(data)
