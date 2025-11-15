from flask import Blueprint, request, jsonify
from controllers.predict_controller import predict_handler

predict_bp = Blueprint("predict", __name__)

@predict_bp.route("/predict-loan", methods=["POST"])
def predict_loan():
    data = request.get_json(force=True)
    return predict_handler(data)
