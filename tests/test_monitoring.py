"""Tests for src/monitoring.py module."""

import json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.monitoring import (
    log_prediction,
    log_model_metrics,
    detect_prediction_drift,
    get_prediction_stats,
)


class TestLogPrediction:
    """Tests for prediction logging."""

    def test_logs_to_local_file(self, tmp_path):
        with patch("src.monitoring.MODELS_DIR", tmp_path):
            input_data = {"INDE_2022": 7.5}
            prediction = {
                "prediction": 1,
                "risk_level": "HIGH",
                "probability": {"no_risk": 0.3, "at_risk": 0.7},
                "model_type": "RandomForest",
            }
            log_prediction(input_data, prediction)

            log_file = tmp_path / "predictions_log.jsonl"
            assert log_file.exists()

            with open(log_file) as f:
                entry = json.loads(f.readline())
            assert entry["prediction"] == 1
            assert entry["risk_level"] == "HIGH"

    def test_logs_to_supabase(self, mock_supabase):
        input_data = {"INDE_2022": 7.5}
        prediction = {
            "prediction": 0,
            "risk_level": "LOW",
            "probability": {"no_risk": 0.8, "at_risk": 0.2},
            "model_type": "RandomForest",
        }
        log_prediction(input_data, prediction, supabase_client=mock_supabase)
        mock_supabase.table.assert_called_with("predictions_log")

    def test_handles_supabase_error(self, tmp_path):
        client = MagicMock()
        client.table.return_value.insert.return_value.execute.side_effect = Exception("DB Error")

        with patch("src.monitoring.MODELS_DIR", tmp_path):
            input_data = {"INDE_2022": 7.5}
            prediction = {"prediction": 0, "risk_level": "LOW", "probability": None, "model_type": "RF"}
            # Should not raise, just log error
            log_prediction(input_data, prediction, supabase_client=client)


class TestLogModelMetrics:
    """Tests for model metrics logging."""

    def test_logs_metrics_locally(self, tmp_path):
        with patch("src.monitoring.MODELS_DIR", tmp_path):
            metrics = {"accuracy": 0.85, "f1_score": 0.82}
            log_model_metrics(metrics, "RandomForest")

            log_file = tmp_path / "model_metrics.jsonl"
            assert log_file.exists()

            with open(log_file) as f:
                entry = json.loads(f.readline())
            assert entry["accuracy"] == 0.85
            assert entry["model_type"] == "RandomForest"


class TestDetectPredictionDrift:
    """Tests for prediction drift detection."""

    def test_no_drift_similar_distributions(self):
        ref = np.array([0.3, 0.4, 0.5, 0.6, 0.7])
        curr = np.array([0.35, 0.45, 0.55, 0.65, 0.7])
        result = detect_prediction_drift(ref, curr, threshold=0.1)
        assert result["drift_detected"] is False

    def test_drift_detected_different_distributions(self):
        ref = np.array([0.2, 0.3, 0.3, 0.4, 0.3])
        curr = np.array([0.7, 0.8, 0.9, 0.85, 0.75])
        result = detect_prediction_drift(ref, curr, threshold=0.1)
        assert result["drift_detected"] is True

    def test_returns_expected_keys(self):
        ref = np.array([0.5, 0.5])
        curr = np.array([0.5, 0.5])
        result = detect_prediction_drift(ref, curr)
        assert "drift_detected" in result
        assert "reference_mean" in result
        assert "current_mean" in result
        assert "mean_difference" in result
        assert "threshold" in result
        assert "timestamp" in result


class TestGetPredictionStats:
    """Tests for prediction statistics."""

    def test_empty_when_no_logs(self, tmp_path):
        with patch("src.monitoring.MODELS_DIR", tmp_path):
            stats = get_prediction_stats()
            assert stats["total_predictions"] == 0

    def test_reads_local_log(self, tmp_path):
        log_file = tmp_path / "predictions_log.jsonl"
        entries = [
            {"risk_level": "HIGH", "probability_at_risk": 0.8, "timestamp": "2024-01-01"},
            {"risk_level": "LOW", "probability_at_risk": 0.2, "timestamp": "2024-01-02"},
        ]
        with open(log_file, "w") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")

        with patch("src.monitoring.MODELS_DIR", tmp_path):
            stats = get_prediction_stats()
            assert stats["total_predictions"] == 2
