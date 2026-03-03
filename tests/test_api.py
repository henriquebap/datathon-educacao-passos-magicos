"""Testes da API FastAPI."""

import json
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


@pytest.fixture
def mock_predictor():
    """Mock do Predictor para testes da API."""
    predictor = MagicMock()
    predictor.model = MagicMock()
    type(predictor.model).__name__ = "RandomForestClassifier"

    predictor.predict.return_value = {
        "prediction": 1,
        "risk_level": "HIGH",
        "probability": {"no_risk": 0.3, "at_risk": 0.7},
        "model_type": "RandomForestClassifier",
        "timestamp": "2024-01-01T00:00:00+00:00",
        "input_features": {},
    }

    predictor.predict_batch.return_value = [
        {
            "prediction": 1,
            "risk_level": "HIGH",
            "probability": {"no_risk": 0.3, "at_risk": 0.7},
            "model_type": "RandomForestClassifier",
            "timestamp": "2024-01-01T00:00:00+00:00",
            "input_features": {},
        },
        {
            "prediction": 0,
            "risk_level": "LOW",
            "probability": {"no_risk": 0.8, "at_risk": 0.2},
            "model_type": "RandomForestClassifier",
            "timestamp": "2024-01-01T00:00:00+00:00",
            "input_features": {},
        },
    ]

    return predictor


@pytest.fixture
def client(mock_predictor):
    """TestClient com mocks injetados via app.state."""
    from api.main import app

    # Injeta mocks no state da app (mesmo padrão do lifespan)
    app.state.predictor = mock_predictor
    app.state.supabase_client = None
    app.state.evaluation_results = {"metrics": {"accuracy": 0.85, "f1_score": 0.82}}

    with TestClient(app, raise_server_exceptions=False) as c:
        yield c

    # Cleanup
    app.state.predictor = None


class TestRootEndpoint:
    def test_root_returns_200(self, client):
        response = client.get("/")
        assert response.status_code == 200

    def test_root_contains_api_name(self, client):
        response = client.get("/")
        data = response.json()
        assert "name" in data
        assert "Passos" in data["name"]

    def test_root_lists_endpoints(self, client):
        response = client.get("/")
        data = response.json()
        assert "endpoints" in data
        assert "predict" in data["endpoints"]
        assert "health" in data["endpoints"]


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_shows_model_loaded(self, client):
        response = client.get("/health")
        data = response.json()
        assert data["model_loaded"] is True
        assert data["status"] == "healthy"

    def test_health_contains_timestamp(self, client):
        response = client.get("/health")
        data = response.json()
        assert "timestamp" in data


class TestPredictEndpoint:
    def test_predict_returns_200(self, client):
        payload = {"INDE_2022": 7.5, "IAA_2022": 6.8}
        response = client.post("/predict", json=payload)
        assert response.status_code == 200

    def test_predict_returns_expected_fields(self, client):
        payload = {"INDE_2022": 7.5}
        response = client.post("/predict", json=payload)
        data = response.json()
        assert "prediction" in data
        assert "risk_level" in data
        assert "probability" in data
        assert "model_type" in data

    def test_predict_with_full_payload(self, client, sample_features):
        response = client.post("/predict", json=sample_features)
        assert response.status_code == 200

    def test_predict_with_empty_payload(self, client):
        response = client.post("/predict", json={})
        assert response.status_code == 200

    def test_predict_risk_level_valid(self, client):
        response = client.post("/predict", json={"INDE_2022": 3.0})
        data = response.json()
        assert data["risk_level"] in ["LOW", "HIGH"]


class TestBatchPredictEndpoint:
    def test_batch_returns_200(self, client):
        payload = {
            "students": [
                {"INDE_2022": 7.5},
                {"INDE_2022": 3.0},
            ]
        }
        response = client.post("/predict/batch", json=payload)
        assert response.status_code == 200

    def test_batch_returns_correct_count(self, client):
        payload = {
            "students": [
                {"INDE_2022": 7.5},
                {"INDE_2022": 3.0},
            ]
        }
        response = client.post("/predict/batch", json=payload)
        data = response.json()
        assert data["total"] == 2
        assert len(data["predictions"]) == 2


class TestMetricsEndpoint:
    def test_metrics_returns_200(self, client):
        response = client.get("/metrics")
        assert response.status_code == 200

    def test_metrics_contains_model_info(self, client):
        response = client.get("/metrics")
        data = response.json()
        assert "model_type" in data
        assert "metrics" in data
        assert "prediction_stats" in data


class TestDriftEndpoint:
    def test_drift_returns_200(self, client):
        response = client.get("/monitoring/drift")
        assert response.status_code == 200

    def test_drift_response_structure(self, client):
        response = client.get("/monitoring/drift")
        data = response.json()
        assert "drift_detected" in data
        assert "details" in data
        assert "timestamp" in data


class TestPredictNoModel:
    def test_predict_without_model_returns_503(self, client):
        """Simula cenário sem modelo setando predictor=None no state."""
        from api.main import app
        original = app.state.predictor
        app.state.predictor = None

        try:
            response = client.post("/predict", json={"INDE_2022": 7.5})
            assert response.status_code == 503
        finally:
            app.state.predictor = original
