"""ml/predict.py
Load the saved model and return predictions with explanation.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import joblib
import numpy as np
import pandas as pd

from config import MODEL_PATH, SCALER_PATH, FEATURE_NAMES_PATH

_model        = None
_scaler       = None
_feature_names = None


def _load():
    global _model, _scaler, _feature_names
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError(
                "Model not found. Run: python ml/train_model.py first."
            )
        _model         = joblib.load(MODEL_PATH)
        _scaler        = joblib.load(SCALER_PATH)
        _feature_names = joblib.load(FEATURE_NAMES_PATH)


def predict(input_dict: dict) -> dict:
    """
    Parameters
    ----------
    input_dict : dict
        Keys matching feature_engineering.FEATURE_COLS + category_id

    Returns
    -------
    dict with keys: trending (bool), probability (float), top_factors (list)
    """
    _load()

    from backend.services.feature_engineering import (
        build_feature_matrix, input_dict_to_df
    )

    df_in = input_dict_to_df(input_dict)
    X     = build_feature_matrix(df_in)

    # Align columns to model's expected features
    for col in _feature_names:
        if col not in X.columns:
            X[col] = 0
    X = X[_feature_names]

    prob = _model.predict_proba(X)[0, 1]
    pred = int(prob >= 0.5)

    # Feature importance explanation
    importances = _model.feature_importances_
    feat_values  = X.values[0]
    contributions = importances * np.abs(feat_values)
    idx_sorted    = np.argsort(contributions)[::-1]

    top_factors = [
        {
            "feature":     _feature_names[i],
            "value":       float(round(feat_values[i], 3)),
            "importance":  float(round(float(importances[i]), 4)),
        }
        for i in idx_sorted[:6]
    ]

    return {
        "trending":    bool(pred),
        "probability": round(float(prob) * 100, 2),
        "top_factors": top_factors,
        "advice":      _advice(input_dict, prob),
    }


def _advice(inp: dict, prob: float) -> list:
    tips = []
    hour = inp.get("upload_hour", -1)
    if not (18 <= hour <= 21):
        tips.append("Upload between 6 PM – 9 PM for peak trending rate.")
    tags = inp.get("tag_count", -1)
    if not (8 <= tags <= 15):
        tips.append(f"Your tag count ({tags}) is outside the optimal 8–15 range.")
    tlen = inp.get("title_length", -1)
    if not (40 <= tlen <= 60):
        tips.append(f"Title length ({tlen} chars) — aim for 40–60 characters.")
    if not inp.get("has_face_thumbnail", 0):
        tips.append("Add a human face to your thumbnail for better CTR.")
    bright = inp.get("thumbnail_brightness", 0)
    if bright < 150:
        tips.append("Increase thumbnail brightness — darker thumbnails get fewer clicks.")
    if prob >= 0.7:
        tips.insert(0, " Strong trending potential! Maintain quality and promote early.")
    elif prob >= 0.5:
        tips.insert(0, " Good potential — a few tweaks could push you over the line.")
    else:
        tips.insert(0, " Low trending probability — focus on the tips below.")
    return tips
