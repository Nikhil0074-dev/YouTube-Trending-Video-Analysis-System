import os

# ── Paths ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_PATH = os.path.join(DATA_DIR, "raw", "trending_videos.csv")
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "processed", "clean_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "backend", "models", "ml_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "backend", "models", "scaler.pkl")
FEATURE_NAMES_PATH = os.path.join(BASE_DIR, "backend", "models", "feature_names.pkl")

# ── Flask ──────────────────────────────────────────────
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
DEBUG = True

# ── YouTube API (optional) ─────────────────────────────
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "")

# ── ML ────────────────────────────────────────────────
TEST_SIZE = 0.2
RANDOM_STATE = 42
TRENDING_THRESHOLD_VIEWS = 500_000   # views needed to be considered "trending"

# ── Analysis constants ────────────────────────────────
CATEGORIES = {
    "1": "Film & Animation",
    "2": "Autos & Vehicles",
    "10": "Music",
    "15": "Pets & Animals",
    "17": "Sports",
    "20": "Gaming",
    "22": "People & Blogs",
    "23": "Comedy",
    "24": "Entertainment",
    "25": "News & Politics",
    "26": "Howto & Style",
    "27": "Education",
    "28": "Science & Technology",
    "29": "Nonprofits & Activism",
}
