# YouTube Trending Video Analysis System

A full-stack data mining + ML project that uncovers what makes YouTube videos trend.

---

## What It Does

| Module | What it builds |
|--------|---------------|
| `generate_data.py` | Synthetic 2,000-video dataset with realistic biases |
| `ml/train_model.py` | Trains & compares 4 ML models, saves the best |
| `backend/` | Flask REST API (12 endpoints) |
| `frontend/index.html` | Single-page dashboard (no build step needed) |

---

## Manual Steps 

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate dataset
python generate_data.py

# 3. Train model
python ml/train_model.py

# 4. Start server
python backend/app.py
```

Then open **http://127.0.0.1:5000** in your browser.

---
## Quick Start (one command)

```bash
python setup_and_run.py
```

Then open **http://127.0.0.1:5000** in your browser.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/overview` | Dataset summary stats |
| GET | `/api/categories` | Views & trending rate by category |
| GET | `/api/upload-hours` | Trending rate per upload hour |
| GET | `/api/upload-days` | Trending rate per day of week |
| GET | `/api/tags` | Tag count analysis |
| GET | `/api/title-length` | Title length analysis |
| GET | `/api/thumbnail` | Thumbnail feature analysis |
| GET | `/api/top-videos` | Top 10 trending videos |
| GET | `/api/recommendations` | Creator recommendations |
| GET | `/api/feature-importance` | ML feature importances |
| POST | `/api/predict` | Predict trending probability |

### Predict example

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "upload_hour": 19,
    "upload_day": 6,
    "title_length": 52,
    "tag_count": 11,
    "duration_seconds": 480,
    "has_face_thumbnail": 1,
    "thumbnail_brightness": 190,
    "category_id": 24
  }'
```

---

## Project Structure

```
youtube-trending-analysis/
├── generate_data.py        # Synthetic dataset generator
├── setup_and_run.py        # One-click bootstrap
├── config.py               # Central config (paths, constants)
├── requirements.txt
│
├── data/
│   ├── raw/                # trending_videos.csv (2000 rows)
│   └── processed/          # clean_data.csv (after EDA)
│
├── backend/
│   ├── app.py              # Flask entry point
│   ├── routes/
│   │   ├── api_routes.py       # Analysis endpoints
│   │   └── prediction_routes.py
│   └── services/
│       ├── data_loader.py      # Load & clean data
│       ├── feature_engineering.py
│       ├── analysis.py         # All analysis functions
│       └── recommendation.py   # Rule-based tips
│
├── ml/
│   ├── train_model.py      # Train & compare 4 models
│   ├── predict.py          # Load model, return prediction
│   └── feature_selection.py
│
├── frontend/
│   └── index.html          # Full SPA dashboard
│
└── tests/
    └── test_model.py       # Pytest unit tests
```

---

## Dashboard Pages

- **Overview** — KPI cards, top videos table, views vs likes scatter
- **Categories** — Which categories dominate trending
- **Upload Timing** — Best hours and days to publish
- **Factors** — Tag count, title length, thumbnail analysis
- **Predictor** — Input your video details → get trending probability
- **Tips** — Data-driven creator recommendations
- **ML Model** — Feature importance bar chart

---

## Using Real YouTube Data

1. Download from [Kaggle – YouTube Trending Video Dataset](https://www.kaggle.com/datasets/datasnaek/youtube-new)
2. Map columns to match `config.py` field names
3. Place in `data/raw/trending_videos.csv`
4. Re-run `python ml/train_model.py`
