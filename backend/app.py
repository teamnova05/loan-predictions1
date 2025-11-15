import os
from flask import Flask, jsonify
from dotenv import load_dotenv

load_dotenv()

def create_app():
    app = Flask(__name__)

    # ============================
    # Register API blueprints
    # ============================
    from routes.predict import predict_bp
    from routes.fraud import fraud_bp
    from routes.requests import req_bp  # <-- NEW ROUTE

    app.register_blueprint(predict_bp, url_prefix="/api")
    app.register_blueprint(fraud_bp, url_prefix="/api")
    app.register_blueprint(req_bp, url_prefix="/api")  # <-- REGISTER HERE

    # ============================
    # Health check route
    # ============================
    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok"}), 200

    return app


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app = create_app()
    app.run(
        host="0.0.0.0",
        port=port,
        debug=(os.getenv("FLASK_ENV") == "development")
    )
