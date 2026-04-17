"""backend/services/analysis.py
All analytical computations — returns plain Python dicts (JSON-serialisable).
"""

import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from backend.services.data_loader import load_processed


# ── helpers ───────────────────────────────────────────

def _safe(val):
    """Convert numpy scalars to Python native types."""
    if isinstance(val, (np.integer,)):   return int(val)
    if isinstance(val, (np.floating,)):  return float(round(float(val), 4))
    return val


def _df():
    return load_processed()


# ── public functions ──────────────────────────────────

def overview_stats() -> dict:
    df = _df()
    trending    = df[df["trending"] == 1]
    non_trending = df[df["trending"] == 0]

    return {
        "total_videos":        int(len(df)),
        "trending_videos":     int(len(trending)),
        "non_trending_videos": int(len(non_trending)),
        "trending_pct":        round(len(trending) / len(df) * 100, 2),
        "avg_views_trending":  int(trending["views"].mean()),
        "avg_views_non":       int(non_trending["views"].mean()),
        "avg_likes_trending":  int(trending["likes"].mean()),
        "avg_comments_trending": int(trending["comments"].mean()),
        "top_category":        df.groupby("category_name")["views"].sum().idxmax(),
    }


def views_by_category() -> list:
    df = _df()
    g = df.groupby("category_name").agg(
        avg_views=("views", "mean"),
        total_videos=("video_id", "count"),
        trending_count=("trending", "sum"),
    ).reset_index()
    g["trending_rate"] = (g["trending_count"] / g["total_videos"] * 100).round(2)
    g = g.sort_values("avg_views", ascending=False)
    return [
        {k: _safe(v) for k, v in row.items()}
        for row in g.to_dict(orient="records")
    ]


def upload_hour_distribution() -> list:
    df = _df()
    g = df.groupby("upload_hour").agg(
        total=("video_id", "count"),
        trending=("trending", "sum"),
    ).reset_index()
    g["trending_rate"] = (g["trending"] / g["total"] * 100).round(2)
    return [
        {"hour": int(r["upload_hour"]),
         "total": int(r["total"]),
         "trending": int(r["trending"]),
         "trending_rate": float(r["trending_rate"])}
        for _, r in g.iterrows()
    ]


def upload_day_distribution() -> list:
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    df = _df()
    g = df.groupby("upload_day").agg(
        total=("video_id", "count"),
        trending=("trending", "sum"),
    ).reset_index()
    g["trending_rate"] = (g["trending"] / g["total"] * 100).round(2)
    return [
        {"day": days[int(r["upload_day"])],
         "day_num": int(r["upload_day"]),
         "total": int(r["total"]),
         "trending": int(r["trending"]),
         "trending_rate": float(r["trending_rate"])}
        for _, r in g.iterrows()
    ]


def tag_count_analysis() -> list:
    df = _df()
    bins   = list(range(0, 32, 3))
    labels = [f"{b}-{b+2}" for b in bins[:-1]]
    df["tag_bucket"] = pd.cut(df["tag_count"], bins=bins, labels=labels, right=False)
    g = df.groupby("tag_bucket", observed=True).agg(
        total=("video_id", "count"),
        trending=("trending", "sum"),
        avg_views=("views", "mean"),
    ).reset_index()
    g["trending_rate"] = (g["trending"] / g["total"] * 100).round(2)
    return [
        {"bucket": str(r["tag_bucket"]),
         "total": int(r["total"]),
         "trending": int(r["trending"]),
         "trending_rate": float(r["trending_rate"]),
         "avg_views": int(r["avg_views"])}
        for _, r in g.iterrows()
    ]


def title_length_analysis() -> list:
    df = _df()
    bins   = [0, 20, 40, 60, 80, 100]
    labels = ["<20", "20-40", "40-60", "60-80", "80+"]
    df["title_bucket"] = pd.cut(df["title_length"], bins=bins, labels=labels)
    g = df.groupby("title_bucket", observed=True).agg(
        total=("video_id", "count"),
        trending=("trending", "sum"),
        avg_views=("views", "mean"),
    ).reset_index()
    g["trending_rate"] = (g["trending"] / g["total"] * 100).round(2)
    return [
        {"bucket": str(r["title_bucket"]),
         "total": int(r["total"]),
         "trending": int(r["trending"]),
         "trending_rate": float(r["trending_rate"]),
         "avg_views": int(r["avg_views"])}
        for _, r in g.iterrows()
    ]


def thumbnail_analysis() -> dict:
    df = _df()
    face   = df.groupby("has_face_thumbnail")["trending"].mean().to_dict()
    text   = df.groupby("has_text_thumbnail")["trending"].mean().to_dict()
    bright = df.groupby(pd.cut(df["thumbnail_brightness"],
                               bins=[0, 100, 150, 200, 255],
                               labels=["dark", "medium", "bright", "very bright"]),
                        observed=True)["trending"].mean()
    return {
        "face_effect": {
            "no_face":   round(float(face.get(0, 0)) * 100, 2),
            "with_face": round(float(face.get(1, 0)) * 100, 2),
        },
        "text_effect": {
            "no_text":   round(float(text.get(0, 0)) * 100, 2),
            "with_text": round(float(text.get(1, 0)) * 100, 2),
        },
        "brightness_effect": {
            str(k): round(float(v) * 100, 2)
            for k, v in bright.items()
        },
    }


def top_trending_videos(n: int = 10) -> list:
    df = _df()
    top = df[df["trending"] == 1].nlargest(n, "views")
    return [
        {
            "video_id":      r["video_id"],
            "title":         r["title"],
            "category":      r["category_name"],
            "views":         int(r["views"]),
            "likes":         int(r["likes"]),
            "upload_hour":   int(r["upload_hour"]),
            "tag_count":     int(r["tag_count"]),
        }
        for _, r in top.iterrows()
    ]


def correlation_matrix() -> dict:
    df = _df()
    cols = ["views", "likes", "comments", "tag_count",
            "title_length", "duration_seconds", "thumbnail_brightness"]
    corr = df[cols].corr().round(3)
    return {
        "columns": cols,
        "matrix":  corr.values.tolist(),
    }
