"""Tests for src/train.py module."""

import pandas as pd
import numpy as np
import pytest
import joblib
from pathlib import Path

from src.train import (
    get_models,
    cross_validate_models,
    train_best_model,
    save_model,
    load_model,
    save_pipeline_artifacts,
    load_pipeline_artifacts,
)


class TestGetModels:
    """Tests for model catalog."""

    def test_returns_dict(self):
        models = get_models()
        assert isinstance(models, dict)

    def test_contains_expected_models(self):
        models = get_models()
        assert "LogisticRegression" in models
        assert "RandomForest" in models
        assert "GradientBoosting" in models
        assert "SVM" in models

    def test_each_model_has_grid(self):
        models = get_models()
        for name, (model, grid) in models.items():
            assert model is not None
            assert isinstance(grid, dict)
            assert len(grid) > 0


class TestCrossValidateModels:
    """Tests for cross-validation."""

    def test_returns_dataframe(self, preprocessed_data):
        X_train, _, y_train, _, _ = preprocessed_data
        results = cross_validate_models(X_train, y_train, cv_folds=3)
        assert isinstance(results, pd.DataFrame)

    def test_contains_expected_columns(self, preprocessed_data):
        X_train, _, y_train, _, _ = preprocessed_data
        results = cross_validate_models(X_train, y_train, cv_folds=3)
        assert "model" in results.columns
        assert "mean_score" in results.columns
        assert "std_score" in results.columns

    def test_scores_between_0_and_1(self, preprocessed_data):
        X_train, _, y_train, _, _ = preprocessed_data
        results = cross_validate_models(X_train, y_train, cv_folds=3)
        assert (results["mean_score"] >= 0).all()
        assert (results["mean_score"] <= 1).all()

    def test_sorted_by_score_descending(self, preprocessed_data):
        X_train, _, y_train, _, _ = preprocessed_data
        results = cross_validate_models(X_train, y_train, cv_folds=3)
        scores = results["mean_score"].values
        assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))


class TestTrainBestModel:
    """Tests for model training."""

    def test_trains_specific_model(self, preprocessed_data):
        X_train, _, y_train, _, _ = preprocessed_data
        model, _, params = train_best_model(
            X_train, y_train, model_name="LogisticRegression", cv_folds=3
        )
        assert model is not None
        assert isinstance(params, dict)

    def test_auto_selects_best_model(self, preprocessed_data):
        X_train, _, y_train, _, _ = preprocessed_data
        model, cv_results, params = train_best_model(
            X_train, y_train, cv_folds=3
        )
        assert model is not None
        assert cv_results is not None

    def test_invalid_model_name_raises(self, preprocessed_data):
        X_train, _, y_train, _, _ = preprocessed_data
        with pytest.raises(ValueError, match="Modelo desconhecido"):
            train_best_model(X_train, y_train, model_name="NonExistent")

    def test_trained_model_can_predict(self, preprocessed_data):
        X_train, X_test, y_train, _, _ = preprocessed_data
        model, _, _ = train_best_model(
            X_train, y_train, model_name="RandomForest", cv_folds=3
        )
        predictions = model.predict(X_test)
        assert len(predictions) == len(X_test)


class TestSaveLoadModel:
    """Tests for model serialization."""

    def test_save_and_load(self, trained_model, tmp_path):
        filepath = tmp_path / "test_model.joblib"
        save_model(trained_model, filepath)
        loaded = load_model(filepath)
        assert type(loaded).__name__ == type(trained_model).__name__

    def test_save_creates_file(self, trained_model, tmp_path):
        filepath = tmp_path / "test_model.joblib"
        save_model(trained_model, filepath)
        assert filepath.exists()

    def test_load_nonexistent_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_model(tmp_path / "nonexistent.joblib")

    def test_loaded_model_predicts_same(self, trained_model, preprocessed_data, tmp_path):
        X_train, X_test, _, _, _ = preprocessed_data
        filepath = tmp_path / "test_model.joblib"

        pred_before = trained_model.predict(X_test)
        save_model(trained_model, filepath)
        loaded = load_model(filepath)
        pred_after = loaded.predict(X_test)

        np.testing.assert_array_equal(pred_before, pred_after)


class TestSaveLoadArtifacts:
    """Tests for pipeline artifact serialization."""

    def test_save_and_load(self, tmp_path):
        artifacts = {"scaler": "test_scaler", "feature_names": ["A", "B"]}
        filepath = tmp_path / "artifacts.joblib"
        save_pipeline_artifacts(artifacts, filepath)
        loaded = load_pipeline_artifacts(filepath)
        assert loaded["scaler"] == "test_scaler"
        assert loaded["feature_names"] == ["A", "B"]

    def test_load_nonexistent_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_pipeline_artifacts(tmp_path / "nonexistent.joblib")
