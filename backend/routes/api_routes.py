"""backend/routes/api_routes.py"""

from flask import Blueprint, jsonify
from backend.services import analysis, recommendation
from ml.feature_selection import get_feature_importance

api_bp = Blueprint("api", __name__, url_prefix="/api")


@api_bp.route("/overview", methods=["GET"])
def overview():
    return jsonify(analysis.overview_stats())


@api_bp.route("/categories", methods=["GET"])
def categories():
    return jsonify(analysis.views_by_category())


@api_bp.route("/upload-hours", methods=["GET"])
def upload_hours():
    return jsonify(analysis.upload_hour_distribution())


@api_bp.route("/upload-days", methods=["GET"])
def upload_days():
    return jsonify(analysis.upload_day_distribution())


@api_bp.route("/tags", methods=["GET"])
def tags():
    return jsonify(analysis.tag_count_analysis())


@api_bp.route("/title-length", methods=["GET"])
def title_length():
    return jsonify(analysis.title_length_analysis())


@api_bp.route("/thumbnail", methods=["GET"])
def thumbnail():
    return jsonify(analysis.thumbnail_analysis())


@api_bp.route("/top-videos", methods=["GET"])
def top_videos():
    return jsonify(analysis.top_trending_videos())


@api_bp.route("/correlation", methods=["GET"])
def correlation():
    return jsonify(analysis.correlation_matrix())


@api_bp.route("/recommendations", methods=["GET"])
def recommendations():
    return jsonify(recommendation.full_recommendations())


@api_bp.route("/feature-importance", methods=["GET"])
def feature_importance():
    try:
        return jsonify(get_feature_importance())
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 400
