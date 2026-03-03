"""Tests for src/feature_engineering.py module."""

import pandas as pd
import numpy as np
import pytest

from src.feature_engineering import (
    create_temporal_features,
    create_composite_indicators,
    create_interaction_features,
    feature_engineering_pipeline,
)


class TestCreateTemporalFeatures:
    """Tests for temporal feature creation."""

    def test_creates_diff_features(self):
        df = pd.DataFrame({
            "INDE_2020": [5.0, 6.0],
            "INDE_2021": [6.0, 7.0],
            "INDE_2022": [7.0, 8.0],
        })
        result = create_temporal_features(df)
        assert "INDE_diff_2020_2021" in result.columns
        assert "INDE_diff_2021_2022" in result.columns

    def test_creates_trend_features(self):
        df = pd.DataFrame({
            "INDE_2020": [5.0, 6.0],
            "INDE_2021": [6.0, 7.0],
            "INDE_2022": [7.0, 8.0],
        })
        result = create_temporal_features(df)
        assert "INDE_trend" in result.columns
        assert result["INDE_trend"].iloc[0] == 2.0  # 7 - 5

    def test_creates_mean_and_std(self):
        df = pd.DataFrame({
            "INDE_2020": [5.0],
            "INDE_2021": [6.0],
            "INDE_2022": [7.0],
        })
        result = create_temporal_features(df)
        assert "INDE_mean" in result.columns
        assert "INDE_std" in result.columns
        assert abs(result["INDE_mean"].iloc[0] - 6.0) < 0.01

    def test_single_year_skips_temporal(self):
        df = pd.DataFrame({"INDE_2022": [7.0], "value": [1]})
        result = create_temporal_features(df)
        # Should not create diff/trend columns
        assert not any("_diff_" in c for c in result.columns)

    def test_preserves_original_columns(self):
        df = pd.DataFrame({
            "INDE_2020": [5.0],
            "INDE_2022": [7.0],
        })
        result = create_temporal_features(df)
        assert "INDE_2020" in result.columns
        assert "INDE_2022" in result.columns


class TestCreateCompositeIndicators:
    """Tests for composite indicator creation."""

    def test_creates_academic_composite(self):
        df = pd.DataFrame({
            "INDE_2022": [7.0],
            "IAA_2022": [6.0],
            "IDA_2022": [8.0],
        })
        result = create_composite_indicators(df)
        assert "ACADEMIC_COMPOSITE" in result.columns
        assert abs(result["ACADEMIC_COMPOSITE"].iloc[0] - 7.0) < 0.01

    def test_creates_engagement_composite(self):
        df = pd.DataFrame({
            "IEG_2022": [7.0],
            "IPS_2022": [5.0],
            "INDE_2022": [6.0],
        })
        result = create_composite_indicators(df)
        assert "ENGAGEMENT_COMPOSITE" in result.columns
        assert abs(result["ENGAGEMENT_COMPOSITE"].iloc[0] - 6.0) < 0.01

    def test_creates_risk_score(self):
        df = pd.DataFrame({
            "INDE_2022": [8.0],
            "IEG_2022": [7.0],
            "IPS_2022": [6.0],
        })
        result = create_composite_indicators(df)
        assert "RISK_SCORE" in result.columns
        expected = 10 - (8 + 7 + 6) / 3
        assert abs(result["RISK_SCORE"].iloc[0] - expected) < 0.01

    def test_creates_age_phase_gap(self):
        df = pd.DataFrame({
            "IDADE_ALUNO_2022": [14],
            "FASE_2022": [4],
            "INDE_2022": [7.0],
        })
        result = create_composite_indicators(df)
        assert "AGE_PHASE_GAP" in result.columns
        assert result["AGE_PHASE_GAP"].iloc[0] == 4  # 14 - 4 - 6

    def test_handles_missing_columns_gracefully(self):
        df = pd.DataFrame({"OTHER_2022": [1.0]})
        # Should not raise
        result = create_composite_indicators(df)
        assert "OTHER_2022" in result.columns


class TestCreateInteractionFeatures:
    """Tests for interaction feature creation."""

    def test_creates_inde_ieg_interaction(self):
        df = pd.DataFrame({
            "INDE_2022": [7.0],
            "IEG_2022": [6.0],
        })
        result = create_interaction_features(df)
        assert "INDE_IEG_INTERACTION" in result.columns
        assert result["INDE_IEG_INTERACTION"].iloc[0] == 42.0

    def test_creates_ips_iaa_interaction(self):
        df = pd.DataFrame({
            "IPS_2022": [5.0],
            "IAA_2022": [8.0],
            "INDE_2022": [7.0],
        })
        result = create_interaction_features(df)
        assert "IPS_IAA_INTERACTION" in result.columns
        assert result["IPS_IAA_INTERACTION"].iloc[0] == 40.0

    def test_handles_missing_columns(self):
        df = pd.DataFrame({"INDE_2022": [7.0]})
        result = create_interaction_features(df)
        assert "IPS_IAA_INTERACTION" not in result.columns


class TestFeatureEngineeringPipeline:
    """Tests for the full feature engineering pipeline."""

    def test_pipeline_adds_features(self, synthetic_data):
        initial_cols = len(synthetic_data.columns)
        result = feature_engineering_pipeline(synthetic_data)
        assert len(result.columns) > initial_cols

    def test_pipeline_preserves_original_data(self, synthetic_data):
        original_cols = set(synthetic_data.columns)
        result = feature_engineering_pipeline(synthetic_data)
        assert original_cols.issubset(set(result.columns))

    def test_pipeline_no_nulls_in_new_features(self, synthetic_data):
        # Fill NaN first to test feature engineering specifically
        from src.preprocessing import handle_missing_values
        clean_data = handle_missing_values(synthetic_data)
        result = feature_engineering_pipeline(clean_data)
        new_cols = [c for c in result.columns if c not in synthetic_data.columns]
        for col in new_cols:
            assert result[col].isnull().sum() == 0, f"NaN found in new feature {col}"
