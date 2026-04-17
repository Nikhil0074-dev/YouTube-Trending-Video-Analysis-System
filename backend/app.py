"""backend/app.py — Flask application entry point."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from flask import Flask, send_from_directory
from flask_cors import CORS

from backend.routes.api_routes import api_bp
from backend.routes.prediction_routes import predict_bp
from config import FLASK_HOST, FLASK_PORT, DEBUG

app = Flask(__name__, static_folder="../frontend", static_url_path="")
CORS(app)

app.register_blueprint(api_bp)
app.register_blueprint(predict_bp)


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/health")
def health():
    return {"status": "ok", "service": "YouTube Trending Analyser"}


if __name__ == "__main__":
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=DEBUG)
