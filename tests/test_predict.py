"""Tests for src/predict.py module."""

import numpy as np
import pandas as pd
import pytest

from src.predict import Predictor


class TestPredictor:
    """Tests for the Predictor class."""

    def test_initialization(self, saved_model_and_artifacts):
        model_path, pipeline_path, _ = saved_model_and_artifacts
        predictor = Predictor(model_path=model_path, pipeline_path=pipeline_path)
        assert predictor.model is not None
        assert predictor.artifacts is not None
        assert len(predictor.feature_names) > 0

    def test_predict_returns_expected_keys(self, saved_model_and_artifacts, sample_features):
        model_path, pipeline_path, _ = saved_model_and_artifacts
        predictor = Predictor(model_path=model_path, pipeline_path=pipeline_path)
        result = predictor.predict(sample_features)
        assert "prediction" in result
        assert "risk_level" in result
        assert "probability" in result
        assert "model_type" in result
        assert "timestamp" in result

    def test_prediction_is_binary(self, saved_model_and_artifacts, sample_features):
        model_path, pipeline_path, _ = saved_model_and_artifacts
        predictor = Predictor(model_path=model_path, pipeline_path=pipeline_path)
        result = predictor.predict(sample_features)
        assert result["prediction"] in [0, 1]

    def test_risk_level_matches_prediction(self, saved_model_and_artifacts, sample_features):
        model_path, pipeline_path, _ = saved_model_and_artifacts
        predictor = Predictor(model_path=model_path, pipeline_path=pipeline_path)
        result = predictor.predict(sample_features)
        if result["prediction"] == 1:
            assert result["risk_level"] == "HIGH"
        else:
            assert result["risk_level"] == "LOW"

    def test_probability_sums_to_one(self, saved_model_and_artifacts, sample_features):
        model_path, pipeline_path, _ = saved_model_and_artifacts
        predictor = Predictor(model_path=model_path, pipeline_path=pipeline_path)
        result = predictor.predict(sample_features)
        if result["probability"] is not None:
            total = result["probability"]["no_risk"] + result["probability"]["at_risk"]
            assert abs(total - 1.0) < 0.01

    def test_predict_batch(self, saved_model_and_artifacts, sample_features):
        model_path, pipeline_path, _ = saved_model_and_artifacts
        predictor = Predictor(model_path=model_path, pipeline_path=pipeline_path)
        results = predictor.predict_batch([sample_features, sample_features])
        assert len(results) == 2
        for r in results:
            assert "prediction" in r
            assert "risk_level" in r

    def test_handles_partial_features(self, saved_model_and_artifacts):
        model_path, pipeline_path, _ = saved_model_and_artifacts
        predictor = Predictor(model_path=model_path, pipeline_path=pipeline_path)
        partial = {"INDE_2022": 7.5, "IAA_2022": 6.0}
        result = predictor.predict(partial)
        assert result["prediction"] in [0, 1]

    def test_prepare_dataframe_returns_dataframe(self, saved_model_and_artifacts, sample_features):
        model_path, pipeline_path, _ = saved_model_and_artifacts
        predictor = Predictor(model_path=model_path, pipeline_path=pipeline_path)
        df = predictor._prepare_dataframe(sample_features)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert list(df.columns) == predictor.feature_names

    def test_missing_model_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            Predictor(
                model_path=str(tmp_path / "nonexistent.joblib"),
                pipeline_path=str(tmp_path / "nonexistent2.joblib"),
            )
