"""backend/routes/prediction_routes.py"""

from flask import Blueprint, request, jsonify
from ml.predict import predict

predict_bp = Blueprint("predict", __name__, url_prefix="/api")


@predict_bp.route("/predict", methods=["POST"])
def predict_trending():
    data = request.get_json(silent=True) or {}
    try:
        result = predict(data)
        return jsonify(result)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
