"""tests/test_model.py
Run: python -m pytest tests/ -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import pandas as pd
import numpy as np


# ── Data loader ───────────────────────────────────────
class TestDataLoader:
    def test_raw_data_exists_after_generate(self, tmp_path):
        """After generating data, the CSV must exist and have expected columns."""
        import importlib, config
        original = config.RAW_DATA_PATH
        config.RAW_DATA_PATH = str(tmp_path / "test_raw.csv")

        import generate_data  # runs when imported because of __name__ guard
        # Manually trigger generation
        exec(open("generate_data.py").read())

        df = pd.read_csv(config.RAW_DATA_PATH)
        expected_cols = {"video_id", "views", "likes", "trending", "tag_count",
                          "upload_hour", "category_name"}
        assert expected_cols.issubset(set(df.columns))
        assert len(df) > 0
        config.RAW_DATA_PATH = original


# ── Feature engineering ───────────────────────────────
class TestFeatureEngineering:
    def setup_method(self):
        from backend.services.feature_engineering import input_dict_to_df, build_feature_matrix
        self.input_dict_to_df = input_dict_to_df
        self.build_feature_matrix = build_feature_matrix

    def test_derived_features_added(self):
        df = self.input_dict_to_df({"upload_hour": 19, "upload_day": 6, "tag_count": 10})
        X  = self.build_feature_matrix(df)
        assert "is_prime_time" in X.columns
        assert "is_weekend" in X.columns
        assert "optimal_tags" in X.columns

    def test_prime_time_flag(self):
        df = self.input_dict_to_df({"upload_hour": 20})
        X  = self.build_feature_matrix(df)
        assert X["is_prime_time"].iloc[0] == 1

    def test_non_prime_time_flag(self):
        df = self.input_dict_to_df({"upload_hour": 3})
        X  = self.build_feature_matrix(df)
        assert X["is_prime_time"].iloc[0] == 0

    def test_optimal_tags(self):
        df_good = self.input_dict_to_df({"tag_count": 12})
        df_bad  = self.input_dict_to_df({"tag_count": 1})
        Xg = self.build_feature_matrix(df_good)
        Xb = self.build_feature_matrix(df_bad)
        assert Xg["optimal_tags"].iloc[0] == 1
        assert Xb["optimal_tags"].iloc[0] == 0


# ── Analysis service ──────────────────────────────────
class TestAnalysis:
    """These tests require the processed dataset to exist."""

    @pytest.fixture(autouse=True)
    def require_data(self):
        from config import PROCESSED_DATA_PATH
        if not os.path.exists(PROCESSED_DATA_PATH):
            pytest.skip("Processed data not found — run generate_data.py first")

    def test_overview_keys(self):
        from backend.services.analysis import overview_stats
        ov = overview_stats()
        for key in ["total_videos","trending_videos","trending_pct","avg_views_trending"]:
            assert key in ov, f"Missing key: {key}"

    def test_category_list(self):
        from backend.services.analysis import views_by_category
        cats = views_by_category()
        assert isinstance(cats, list)
        assert len(cats) > 0
        assert "category_name" in cats[0]

    def test_upload_hours_24(self):
        from backend.services.analysis import upload_hour_distribution
        hours = upload_hour_distribution()
        assert len(hours) == 24

    def test_thumbnail_keys(self):
        from backend.services.analysis import thumbnail_analysis
        t = thumbnail_analysis()
        assert "face_effect" in t
        assert "text_effect" in t


# ── Recommendation service ────────────────────────────
class TestRecommendations:
    @pytest.fixture(autouse=True)
    def require_data(self):
        from config import PROCESSED_DATA_PATH
        if not os.path.exists(PROCESSED_DATA_PATH):
            pytest.skip("Processed data not found")

    def test_full_recommendations_shape(self):
        from backend.services.recommendation import full_recommendations
        r = full_recommendations()
        assert "upload_time" in r
        assert "tags" in r
        assert "thumbnail" in r
        assert "title" in r

    def test_thumbnail_tips_have_impact(self):
        from backend.services.recommendation import thumbnail_tips
        tips = thumbnail_tips()
        for tip in tips:
            assert tip["impact"] in ("high", "medium", "low")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
