"""ml/feature_selection.py
Analyse feature importance and correlation — callable as standalone script.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import joblib
import pandas as pd
import numpy as np
from config import MODEL_PATH, FEATURE_NAMES_PATH


def get_feature_importance() -> list:
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError("Model not trained yet. Run ml/train_model.py first.")
    model    = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURE_NAMES_PATH)
    pairs    = sorted(zip(features, model.feature_importances_),
                       key=lambda x: -x[1])
    return [
        {"feature": f, "importance": round(float(imp), 4)}
        for f, imp in pairs
    ]


if __name__ == "__main__":
    items = get_feature_importance()
    print("\n  Feature Importances:")
    for item in items[:15]:
        bar = "█" * int(item["importance"] * 200)
        print(f"  {item['feature']:<30} {item['importance']:.4f}  {bar}")
