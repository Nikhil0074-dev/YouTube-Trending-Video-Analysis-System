"""backend/services/data_loader.py"""

import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import RAW_DATA_PATH, PROCESSED_DATA_PATH


def load_raw() -> pd.DataFrame:
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(
            f"Raw data not found at {RAW_DATA_PATH}. "
            "Run: python generate_data.py"
        )
    return pd.read_csv(RAW_DATA_PATH, parse_dates=["upload_date"])


def clean_and_save() -> pd.DataFrame:
    df = load_raw()

    # ── Drop exact duplicates ──────────────────────────
    df.drop_duplicates(subset=["video_id"], inplace=True)

    # ── Fill missing numerics with median ─────────────
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # ── Clip extreme outliers (> 99th percentile) ──────
    for col in ["views", "likes", "comments"]:
        cap = df[col].quantile(0.99)
        df[col] = df[col].clip(upper=cap)

    # ── Derived features ───────────────────────────────
    df["duration_minutes"] = (df["duration_seconds"] / 60).round(2)
    df["is_weekend"]       = df["upload_day"].isin([5, 6]).astype(int)
    df["is_prime_time"]    = df["upload_hour"].between(18, 21).astype(int)
    df["optimal_tags"]     = df["tag_count"].between(8, 15).astype(int)
    df["optimal_title"]    = df["title_length"].between(40, 60).astype(int)

    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"  Cleaned data saved → {PROCESSED_DATA_PATH}  ({len(df)} rows)")
    return df


def load_processed() -> pd.DataFrame:
    if not os.path.exists(PROCESSED_DATA_PATH):
        return clean_and_save()
    return pd.read_csv(PROCESSED_DATA_PATH, parse_dates=["upload_date"])
