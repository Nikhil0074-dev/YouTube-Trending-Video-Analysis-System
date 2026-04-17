"""
generate_data.py
Generates a realistic synthetic YouTube trending-video dataset.
Run: python generate_data.py
"""

import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RAW_DATA_PATH

np.random.seed(42)
N = 2000  # number of records

CATEGORIES = {
    10: "Music", 20: "Gaming", 24: "Entertainment",
    22: "People & Blogs", 25: "News & Politics",
    27: "Education", 28: "Science & Technology",
    17: "Sports", 23: "Comedy", 26: "Howto & Style",
}
category_ids = list(CATEGORIES.keys())

def random_title(length_target):
    words = ["How", "Why", "Best", "Top", "Amazing", "Viral", "Official",
             "New", "Breaking", "Ultimate", "Real", "Secret", "Inside",
             "Review", "Challenge", "Reaction", "Tutorial", "Guide",
             "Explained", "Exposed", "First", "Last", "vs", "2024"]
    title = []
    while len(" ".join(title)) < length_target:
        title.append(np.random.choice(words))
    return " ".join(title)[:length_target]

# ── Core numeric features ─────────────────────────────
upload_hour = np.random.randint(0, 24, N)
upload_day  = np.random.randint(0, 7, N)   # 0=Mon … 6=Sun
category_id = np.random.choice(category_ids, N)
tag_count   = np.random.randint(0, 30, N)
title_len   = np.random.randint(15, 100, N)
desc_len    = np.random.randint(50, 5000, N)
duration_s  = np.random.randint(30, 3600, N)   # seconds
has_face    = np.random.randint(0, 2, N)
thumb_bright = np.random.randint(50, 255, N)
thumb_text   = np.random.randint(0, 2, N)

# ── Simulate views with realistic biases ─────────────
base_views = np.random.lognormal(mean=13, sigma=1.8, size=N).astype(int)

# Boost for prime upload times
prime_time = ((upload_hour >= 18) & (upload_hour <= 21)).astype(int)
base_views = (base_views * (1 + 0.4 * prime_time)).astype(int)

# Boost for optimal tag count
good_tags = ((tag_count >= 8) & (tag_count <= 15)).astype(int)
base_views = (base_views * (1 + 0.2 * good_tags)).astype(int)

# Boost for faces in thumbnail
base_views = (base_views * (1 + 0.15 * has_face)).astype(int)

# Boost for short titles
short_title = ((title_len >= 40) & (title_len <= 60)).astype(int)
base_views  = (base_views * (1 + 0.1 * short_title)).astype(int)

# ── Engagement derived from views ────────────────────
like_ratio    = np.random.uniform(0.03, 0.12, N)
comment_ratio = np.random.uniform(0.001, 0.03, N)
likes         = (base_views * like_ratio).astype(int)
comments      = (base_views * comment_ratio).astype(int)
dislikes      = (likes * np.random.uniform(0.01, 0.1, N)).astype(int)

# ── Trending label (binary) ───────────────────────────
trending = (base_views >= 500_000).astype(int)

# ── Timestamps ───────────────────────────────────────
dates = pd.date_range("2023-01-01", periods=N, freq="4h").to_numpy()
np.random.shuffle(dates)

# ── Build DataFrame ───────────────────────────────────
df = pd.DataFrame({
    "video_id":        [f"vid_{i:05d}" for i in range(N)],
    "title":           [random_title(tl) for tl in title_len],
    "category_id":     category_id,
    "category_name":   [CATEGORIES[c] for c in category_id],
    "upload_date":     dates,
    "upload_hour":     upload_hour,
    "upload_day":      upload_day,
    "title_length":    title_len,
    "description_length": desc_len,
    "tag_count":       tag_count,
    "duration_seconds": duration_s,
    "views":           base_views,
    "likes":           likes,
    "dislikes":        dislikes,
    "comments":        comments,
    "like_view_ratio": (likes / (base_views + 1)).round(4),
    "comment_view_ratio": (comments / (base_views + 1)).round(6),
    "has_face_thumbnail": has_face,
    "thumbnail_brightness": thumb_bright,
    "has_text_thumbnail": thumb_text,
    "trending":        trending,
})

os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)
df.to_csv(RAW_DATA_PATH, index=False)
print(f"  Dataset saved → {RAW_DATA_PATH}")
print(f"   Rows: {len(df)}  |  Trending: {trending.sum()}  |  Non-trending: {(1-trending).sum()}")
