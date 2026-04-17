"""backend/services/feature_engineering.py"""

import pandas as pd
import numpy as np

# Features used by the ML model
FEATURE_COLS = [
    "upload_hour",
    "upload_day",
    "title_length",
    "description_length",
    "tag_count",
    "duration_seconds",
    "has_face_thumbnail",
    "thumbnail_brightness",
    "has_text_thumbnail",
    "is_weekend",
    "is_prime_time",
    "optimal_tags",
    "optimal_title",
    # category one-hot columns are added dynamically
]

CATEGORY_IDS = [10, 17, 20, 22, 23, 24, 25, 26, 27, 28]


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived/engineered columns to a dataframe."""
    df = df.copy()
    df["duration_minutes"] = df["duration_seconds"] / 60
    df["is_weekend"]       = df["upload_day"].isin([5, 6]).astype(int)
    df["is_prime_time"]    = df["upload_hour"].between(18, 21).astype(int)
    df["optimal_tags"]     = df["tag_count"].between(8, 15).astype(int)
    df["optimal_title"]    = df["title_length"].between(40, 60).astype(int)
    return df


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return X (features only) ready for model training or inference."""
    df = add_derived_features(df)

    # One-hot encode category_id
    cat_dummies = pd.get_dummies(df["category_id"], prefix="cat")
    # Ensure all known categories are present (needed during inference)
    for cid in CATEGORY_IDS:
        col = f"cat_{cid}"
        if col not in cat_dummies.columns:
            cat_dummies[col] = 0

    X = pd.concat([df[FEATURE_COLS], cat_dummies], axis=1)
    return X.fillna(0)


def input_dict_to_df(data: dict) -> pd.DataFrame:
    """Convert a single prediction request (dict) into a 1-row DataFrame."""
    defaults = {
        "upload_hour": 18,
        "upload_day": 2,
        "title_length": 50,
        "description_length": 500,
        "tag_count": 10,
        "duration_seconds": 600,
        "has_face_thumbnail": 1,
        "thumbnail_brightness": 180,
        "has_text_thumbnail": 1,
        "category_id": 24,
        "is_weekend": 0,
        "is_prime_time": 1,
        "optimal_tags": 1,
        "optimal_title": 1,
    }
    defaults.update(data)
    return pd.DataFrame([defaults])
