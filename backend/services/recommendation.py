"""backend/services/recommendation.py
Rule-based + data-driven recommendations for content creators.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from backend.services.analysis import upload_hour_distribution, upload_day_distribution, tag_count_analysis


def best_upload_time() -> dict:
    hours = upload_hour_distribution()
    best  = max(hours, key=lambda x: x["trending_rate"])

    days  = upload_day_distribution()
    best_day = max(days, key=lambda x: x["trending_rate"])

    hour_labels = {
        range(0, 6):   "Late night",
        range(6, 12):  "Morning",
        range(12, 18): "Afternoon",
        range(18, 22): "Evening (prime time)",
        range(22, 24): "Night",
    }
    label = "Unknown"
    for r, lbl in hour_labels.items():
        if best["hour"] in r:
            label = lbl
            break

    return {
        "best_hour":         best["hour"],
        "best_hour_label":   label,
        "best_hour_trending_rate": best["trending_rate"],
        "best_day":          best_day["day"],
        "best_day_trending_rate":  best_day["trending_rate"],
        "top_5_hours":       sorted(hours, key=lambda x: -x["trending_rate"])[:5],
    }


def tag_recommendations() -> dict:
    tags = tag_count_analysis()
    best = max(tags, key=lambda x: x["trending_rate"])
    return {
        "optimal_range":      best["bucket"],
        "trending_rate":      best["trending_rate"],
        "avg_views_at_range": best["avg_views"],
        "all_buckets":        tags,
        "tip": (
            f"Aim for {best['bucket']} tags. "
            "More tags don't always help — irrelevant tags can hurt discoverability."
        ),
    }


def thumbnail_tips() -> list:
    return [
        {
            "tip":    "Include a human face",
            "impact": "high",
            "reason": "Faces attract attention and signal relatability. CTR improves ~15%.",
        },
        {
            "tip":    "Use bright, high-contrast colours",
            "impact": "high",
            "reason": "Thumbnails with brightness 150–255 correlate with more clicks.",
        },
        {
            "tip":    "Add 2–4 words of overlay text",
            "impact": "medium",
            "reason": "Text gives context fast, especially on mobile.",
        },
        {
            "tip":    "Avoid cluttered designs",
            "impact": "medium",
            "reason": "Busy thumbnails are hard to read at small sizes.",
        },
    ]


def title_tips() -> list:
    return [
        {
            "tip":    "Keep title between 40 and 60 characters",
            "impact": "high",
            "reason": "Titles in this range see the highest trending rate in the dataset.",
        },
        {
            "tip":    "Lead with the most important keyword",
            "impact": "high",
            "reason": "YouTube search weighs the start of the title more.",
        },
        {
            "tip":    "Use numbers (e.g. Top 5, 3 Mistakes)",
            "impact": "medium",
            "reason": "Numbered titles signal concrete, scannable content.",
        },
        {
            "tip":    "Avoid clickbait that mismatches content",
            "impact": "high",
            "reason": "High drop-off rate hurts watch time, which hurts ranking.",
        },
    ]


def full_recommendations() -> dict:
    return {
        "upload_time":  best_upload_time(),
        "tags":         tag_recommendations(),
        "thumbnail":    thumbnail_tips(),
        "title":        title_tips(),
    }
