"""Avaliação de modelos — métricas, confusion matrix, importância de features."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.utils import MODELS_DIR


def compute_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    y_proba: np.ndarray | None = None,
) -> dict:
    """Calcula métricas de classificação binária."""
    metrics = {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_true, y_pred, zero_division=0), 4),
    }

    if y_proba is not None:
        try:
            metrics["auc_roc"] = round(roc_auc_score(y_true, y_proba), 4)
        except ValueError:
            metrics["auc_roc"] = None

    return metrics


def compute_confusion_matrix(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
) -> dict:
    """Calcula confusion matrix e extrai TP, TN, FP, FN."""
    cm = confusion_matrix(y_true, y_pred)

    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        return {
            "matrix": cm.tolist(),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        }

    return {"matrix": cm.tolist()}


def get_classification_report(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
) -> str:
    """Gera classification report formatado."""
    target_names = ["Sem Risco (0)", "Em Risco (1)"]
    return classification_report(y_true, y_pred, target_names=target_names, zero_division=0)


def get_feature_importance(model, feature_names: list[str]) -> pd.DataFrame:
    """Extrai importância das features do modelo treinado."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        logger.warning("Modelo não possui feature_importances_ nem coef_")
        return pd.DataFrame(columns=["feature", "importance"])

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    return importance_df


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_names: list[str] | None = None,
) -> dict:
    """Avaliação completa do modelo no conjunto de teste."""
    logger.info("=" * 50)
    logger.info("Avaliando modelo")
    logger.info("=" * 50)

    y_pred = model.predict(X_test)

    y_proba = None
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]

    metrics = compute_metrics(y_test, y_pred, y_proba)
    logger.info(f"Métricas: {metrics}")

    cm = compute_confusion_matrix(y_test, y_pred)
    logger.info(f"Confusion Matrix:\n{np.array(cm['matrix'])}")

    report = get_classification_report(y_test, y_pred)
    logger.info(f"\nClassification Report:\n{report}")

    if feature_names is None:
        feature_names = X_test.columns.tolist()
    importance_df = get_feature_importance(model, feature_names)

    if not importance_df.empty:
        logger.info(f"\nTop 10 features:\n{importance_df.head(10).to_string()}")

    results = {
        "metrics": metrics,
        "confusion_matrix": cm,
        "classification_report": report,
        "feature_importance": importance_df.to_dict("records") if not importance_df.empty else [],
        "model_type": type(model).__name__,
        "n_test_samples": len(y_test),
    }

    # Justificativa da métrica primária (F1):
    # Recall alto = não deixar passar alunos em risco (falsos negativos caros)
    # Precision ok = não gerar alarmes falsos excessivos
    # F1 equilibra os dois
    logger.info(
        f"\n--- Justificativa da métrica ---\n"
        f"Métrica primária: F1-Score = {metrics['f1_score']}\n"
        f"Recall = {metrics['recall']} (capacidade de detectar alunos em risco)\n"
        f"Precision = {metrics['precision']} (acurácia das predições de risco)\n"
        f"Priorizamos Recall para minimizar falsos negativos — não podemos\n"
        f"deixar de identificar alunos que realmente estão em defasagem.\n"
        f"O F1-Score equilibra isso com a precisão."
    )

    logger.info("Avaliação concluída")
    return results


def save_evaluation_report(results: dict, filepath: str | Path | None = None) -> Path:
    """Salva resultado da avaliação em JSON."""
    if filepath is None:
        filepath = MODELS_DIR / "evaluation_report.json"
    filepath = Path(filepath)

    serializable = {}
    for key, value in results.items():
        if isinstance(value, pd.DataFrame):
            serializable[key] = value.to_dict("records")
        elif isinstance(value, np.ndarray):
            serializable[key] = value.tolist()
        else:
            serializable[key] = value

    with open(filepath, "w") as f:
        json.dump(serializable, f, indent=2, default=str)

    logger.info(f"Relatório de avaliação salvo em {filepath}")
    return filepath
