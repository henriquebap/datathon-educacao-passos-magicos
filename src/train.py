"""Treinamento e seleção de modelos para o pipeline de classificação."""

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.svm import SVC

from src.utils import CV_FOLDS, MODELS_DIR, RANDOM_STATE


def get_models() -> dict:
    """Retorna os modelos candidatos com seus grids de hiperparâmetros."""
    models = {
        "LogisticRegression": (
            LogisticRegression(
                random_state=RANDOM_STATE,
                max_iter=1000,
                class_weight="balanced",
            ),
            {
                "C": [0.01, 0.1, 1.0, 10.0],
                "penalty": ["l2"],
            },
        ),
        "RandomForest": (
            RandomForestClassifier(
                random_state=RANDOM_STATE,
                class_weight="balanced",
                n_jobs=-1,
            ),
            {
                "n_estimators": [100, 200, 300],
                "max_depth": [5, 10, 15, None],
                "min_samples_split": [2, 5, 10],
            },
        ),
        "GradientBoosting": (
            GradientBoostingClassifier(
                random_state=RANDOM_STATE,
            ),
            {
                "n_estimators": [100, 200],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.1, 0.2],
                "subsample": [0.8, 1.0],
            },
        ),
        "SVM": (
            SVC(
                random_state=RANDOM_STATE,
                class_weight="balanced",
                probability=True,
            ),
            {
                "C": [0.1, 1.0, 10.0],
                "kernel": ["rbf", "linear"],
            },
        ),
    }

    try:
        from xgboost import XGBClassifier

        models["XGBoost"] = (
            XGBClassifier(
                random_state=RANDOM_STATE,
                eval_metric="logloss",
            ),
            {
                "n_estimators": [100, 200, 300],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.1, 0.2],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0],
            },
        )
    except ImportError:
        logger.warning("XGBoost nao instalado, pulando")

    return models


def cross_validate_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    scoring: str = "f1",
    cv_folds: int = CV_FOLDS,
) -> pd.DataFrame:
    """Roda cross-validation em todos os modelos candidatos.

    Retorna DataFrame com resultados ordenados por score.
    """
    models = get_models()
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)

    results = []
    for name, (model, _) in models.items():
        logger.info(f"Cross-validando {name}...")
        start = time.time()

        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        elapsed = time.time() - start

        result = {
            "model": name,
            "mean_score": scores.mean(),
            "std_score": scores.std(),
            "min_score": scores.min(),
            "max_score": scores.max(),
            "time_seconds": round(elapsed, 2),
        }
        results.append(result)

        logger.info(
            f"  {name}: {scoring}={scores.mean():.4f} (+/- {scores.std():.4f}) "
            f"[{elapsed:.1f}s]"
        )

    results_df = pd.DataFrame(results).sort_values("mean_score", ascending=False)
    logger.info(f"\nMelhor modelo por {scoring}: {results_df.iloc[0]['model']}")
    return results_df


def train_best_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str | None = None,
    scoring: str = "f1",
    cv_folds: int = CV_FOLDS,
) -> tuple:
    """Treina o melhor modelo com tuning de hiperparâmetros via GridSearchCV.

    Se model_name for None, faz cross-validation pra selecionar automaticamente.
    """
    models = get_models()

    if model_name is None:
        cv_results = cross_validate_models(X_train, y_train, scoring=scoring, cv_folds=cv_folds)
        model_name = cv_results.iloc[0]["model"]
        logger.info(f"Selecionado automaticamente: {model_name}")
    else:
        cv_results = None

    if model_name not in models:
        raise ValueError(f"Modelo desconhecido: {model_name}. Disponíveis: {list(models.keys())}")

    model, param_grid = models[model_name]

    logger.info(f"Tuning {model_name} com GridSearchCV...")
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        verbose=0,
        refit=True,
    )

    start = time.time()
    grid_search.fit(X_train, y_train)
    elapsed = time.time() - start

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    logger.info(f"Melhores params de {model_name}: {best_params}")
    logger.info(f"Melhor CV {scoring}: {best_score:.4f} [{elapsed:.1f}s]")

    return best_model, cv_results, best_params


def save_model(model, filepath: str | Path | None = None) -> Path:
    """Salva modelo treinado via joblib."""
    if filepath is None:
        filepath = MODELS_DIR / "model.joblib"
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, filepath)
    logger.info(f"Modelo salvo em {filepath}")
    return filepath


def load_model(filepath: str | Path | None = None):
    """Carrega modelo treinado do disco."""
    if filepath is None:
        filepath = MODELS_DIR / "model.joblib"
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Arquivo do modelo não encontrado: {filepath}")

    model = joblib.load(filepath)
    logger.info(f"Modelo carregado de {filepath}")
    return model


def save_pipeline_artifacts(artifacts: dict, filepath: str | Path | None = None) -> Path:
    """Salva artefatos de preprocessing (scaler, encoders, feature_names)."""
    if filepath is None:
        filepath = MODELS_DIR / "pipeline.joblib"
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(artifacts, filepath)
    logger.info(f"Artefatos do pipeline salvos em {filepath}")
    return filepath


def load_pipeline_artifacts(filepath: str | Path | None = None) -> dict:
    """Carrega artefatos do pipeline de preprocessing."""
    if filepath is None:
        filepath = MODELS_DIR / "pipeline.joblib"
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Artefatos do pipeline não encontrados: {filepath}")

    artifacts = joblib.load(filepath)
    logger.info(f"Artefatos do pipeline carregados de {filepath}")
    return artifacts


def training_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    artifacts: dict,
    model_name: str | None = None,
    scoring: str = "f1",
) -> tuple:
    """Executa o pipeline completo de treinamento: seleção, tuning, persistência."""
    logger.info("=" * 50)
    logger.info("Iniciando pipeline de treinamento")
    logger.info("=" * 50)

    model, cv_results, best_params = train_best_model(
        X_train, y_train, model_name=model_name, scoring=scoring
    )

    save_model(model)
    save_pipeline_artifacts(artifacts)

    logger.info("Pipeline de treinamento concluído")
    return model, cv_results, best_params
