"""Tests for src/evaluate.py module."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.evaluate import (
    compute_metrics,
    compute_confusion_matrix,
    get_classification_report,
    get_feature_importance,
    evaluate_model,
    save_evaluation_report,
)


class TestComputeMetrics:
    """Tests for metric computation."""

    def test_returns_expected_keys(self, y_true, y_pred):
        metrics = compute_metrics(y_true, y_pred)
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics

    def test_with_probabilities(self, y_true, y_pred, y_proba):
        metrics = compute_metrics(y_true, y_pred, y_proba)
        assert "auc_roc" in metrics
        assert metrics["auc_roc"] is not None

    def test_perfect_predictions(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        metrics = compute_metrics(y_true, y_pred)
        assert metrics["accuracy"] == 1.0
        assert metrics["f1_score"] == 1.0

    def test_all_wrong_predictions(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])
        metrics = compute_metrics(y_true, y_pred)
        assert metrics["accuracy"] == 0.0

    def test_metrics_between_0_and_1(self, y_true, y_pred):
        metrics = compute_metrics(y_true, y_pred)
        for key, value in metrics.items():
            if value is not None:
                assert 0 <= value <= 1, f"{key} = {value} is out of range"


class TestComputeConfusionMatrix:
    """Tests for confusion matrix computation."""

    def test_returns_matrix(self, y_true, y_pred):
        cm = compute_confusion_matrix(y_true, y_pred)
        assert "matrix" in cm

    def test_binary_components(self, y_true, y_pred):
        cm = compute_confusion_matrix(y_true, y_pred)
        assert "true_negatives" in cm
        assert "false_positives" in cm
        assert "false_negatives" in cm
        assert "true_positives" in cm

    def test_components_sum_to_total(self, y_true, y_pred):
        cm = compute_confusion_matrix(y_true, y_pred)
        total = cm["true_negatives"] + cm["false_positives"] + cm["false_negatives"] + cm["true_positives"]
        assert total == len(y_true)


class TestGetClassificationReport:
    """Tests for classification report."""

    def test_returns_string(self, y_true, y_pred):
        report = get_classification_report(y_true, y_pred)
        assert isinstance(report, str)

    def test_contains_class_labels(self, y_true, y_pred):
        report = get_classification_report(y_true, y_pred)
        assert "Sem Risco" in report
        assert "Em Risco" in report


class TestGetFeatureImportance:
    """Tests for feature importance extraction."""

    def test_with_rf_model(self, preprocessed_data):
        X_train, _, y_train, _, _ = preprocessed_data
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        importance = get_feature_importance(model, X_train.columns.tolist())
        assert isinstance(importance, pd.DataFrame)
        assert "feature" in importance.columns
        assert "importance" in importance.columns
        assert len(importance) == len(X_train.columns)

    def test_sorted_descending(self, preprocessed_data):
        X_train, _, y_train, _, _ = preprocessed_data
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        importance = get_feature_importance(model, X_train.columns.tolist())
        values = importance["importance"].values
        assert all(values[i] >= values[i + 1] for i in range(len(values) - 1))

    def test_model_without_importances(self):
        class DummyModel:
            pass
        importance = get_feature_importance(DummyModel(), ["A", "B"])
        assert importance.empty


class TestEvaluateModel:
    """Tests for full model evaluation."""

    def test_returns_expected_keys(self, trained_model, preprocessed_data):
        _, X_test, _, y_test, _ = preprocessed_data
        results = evaluate_model(trained_model, X_test, y_test)
        assert "metrics" in results
        assert "confusion_matrix" in results
        assert "classification_report" in results
        assert "feature_importance" in results
        assert "model_type" in results
        assert "n_test_samples" in results

    def test_metrics_are_valid(self, trained_model, preprocessed_data):
        _, X_test, _, y_test, _ = preprocessed_data
        results = evaluate_model(trained_model, X_test, y_test)
        metrics = results["metrics"]
        for key in ["accuracy", "precision", "recall", "f1_score"]:
            assert 0 <= metrics[key] <= 1


class TestSaveEvaluationReport:
    """Tests for report saving."""

    def test_saves_json(self, tmp_path):
        results = {
            "metrics": {"accuracy": 0.85, "f1_score": 0.82},
            "model_type": "RandomForest",
        }
        path = save_evaluation_report(results, tmp_path / "report.json")
        assert path.exists()

    def test_file_is_valid_json(self, tmp_path):
        import json
        results = {"metrics": {"accuracy": 0.85}}
        path = save_evaluation_report(results, tmp_path / "report.json")
        with open(path) as f:
            data = json.load(f)
        assert data["metrics"]["accuracy"] == 0.85
