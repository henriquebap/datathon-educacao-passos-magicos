"""Fixtures compartilhadas dos testes."""

import numpy as np
import pandas as pd
import pytest
import joblib
from pathlib import Path
from unittest.mock import MagicMock

from src.utils import generate_synthetic_data, MODELS_DIR, PEDRA_MAP


@pytest.fixture
def synthetic_data():
    """Dataset sintético com 100 amostras."""
    return generate_synthetic_data(n_samples=100, seed=42)


@pytest.fixture
def small_synthetic_data():
    """Dataset menor pra testes rápidos."""
    return generate_synthetic_data(n_samples=50, seed=42)


@pytest.fixture
def sample_features():
    """Dicionário de features de um aluno para teste de predição.

    Nota: IAN foi removido propositalmente (é coluna de leakage).
    """
    return {
        "INDE_2020": 6.5,
        "INDE_2021": 7.0,
        "INDE_2022": 7.5,
        "IAA_2020": 5.8,
        "IAA_2021": 6.2,
        "IAA_2022": 6.8,
        "IEG_2020": 6.0,
        "IEG_2021": 6.5,
        "IEG_2022": 7.2,
        "IPS_2020": 5.5,
        "IPS_2021": 6.0,
        "IPS_2022": 6.5,
        "IDA_2020": 6.0,
        "IDA_2021": 6.5,
        "IDA_2022": 7.0,
        "IPP_2020": 4.5,
        "IPP_2021": 5.0,
        "IPP_2022": 5.5,
        "IPV_2020": 5.0,
        "IPV_2021": 5.5,
        "IPV_2022": 5.8,
        "IDADE_ALUNO_2020": 12,
        "IDADE_ALUNO_2021": 13,
        "IDADE_ALUNO_2022": 14,
        "PEDRA_2020": "Ametista",
        "PEDRA_2021": "Ametista",
        "PEDRA_2022": "Topázio",
        "FASE_2020": 3,
        "FASE_2021": 4,
        "FASE_2022": 5,
        "TURMA_2020": "A",
        "TURMA_2021": "B",
        "TURMA_2022": "B",
        "PONTO_VIRADA_2020": 0,
        "PONTO_VIRADA_2021": 0,
        "PONTO_VIRADA_2022": 1,
        "BOLSISTA_2020": 0,
        "BOLSISTA_2021": 1,
        "BOLSISTA_2022": 1,
        "ANOS_PM_2020": 2,
        "ANOS_PM_2021": 3,
        "ANOS_PM_2022": 4,
    }


@pytest.fixture
def preprocessed_data(synthetic_data):
    """Dados pré-processados (train/test split) prontos pra teste."""
    from src.preprocessing import preprocess_pipeline

    X_train, X_test, y_train, y_test, artifacts = preprocess_pipeline(
        synthetic_data, missing_strategy="median"
    )
    return X_train, X_test, y_train, y_test, artifacts


@pytest.fixture
def trained_model(preprocessed_data):
    """RandomForest treinado rápido (10 árvores) pra testes."""
    from sklearn.ensemble import RandomForestClassifier

    X_train, X_test, y_train, y_test, artifacts = preprocessed_data
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    return model


@pytest.fixture
def saved_model_and_artifacts(trained_model, preprocessed_data, tmp_path):
    """Modelo e artefatos salvos em diretório temporário."""
    X_train, X_test, y_train, y_test, artifacts = preprocessed_data

    model_path = tmp_path / "model.joblib"
    pipeline_path = tmp_path / "pipeline.joblib"

    joblib.dump(trained_model, model_path)
    joblib.dump(artifacts, pipeline_path)

    return str(model_path), str(pipeline_path), artifacts


@pytest.fixture
def mock_supabase():
    """Mock do cliente Supabase."""
    client = MagicMock()
    client.table.return_value.insert.return_value.execute.return_value = MagicMock(data=[])
    client.table.return_value.select.return_value.order.return_value.limit.return_value.execute.return_value = MagicMock(data=[])
    return client


@pytest.fixture
def y_true():
    return np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 0])


@pytest.fixture
def y_pred():
    return np.array([0, 1, 1, 1, 0, 0, 0, 1, 1, 0])


@pytest.fixture
def y_proba():
    return np.array([0.2, 0.6, 0.8, 0.9, 0.3, 0.4, 0.1, 0.7, 0.85, 0.15])
