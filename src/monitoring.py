"""Monitoramento de predições e detecção de drift (Evidently AI + Supabase)."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from src.utils import DATA_PROCESSED_DIR, MODELS_DIR, SUPABASE_KEY, SUPABASE_URL


def get_supabase_client():
    """Conecta ao Supabase se as credenciais estiverem configuradas."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        logger.warning("Credenciais Supabase não configuradas — log será apenas local")
        return None

    try:
        from supabase import create_client
        client = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info("Supabase conectado")
        return client
    except Exception as e:
        logger.error(f"Falha ao conectar no Supabase: {e}")
        return None


def log_prediction(input_data: dict, prediction: dict, supabase_client=None) -> None:
    """Registra predição no Supabase e/ou arquivo JSONL local."""
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_data": json.dumps(input_data, default=str),
        "prediction": prediction.get("prediction"),
        "risk_level": prediction.get("risk_level"),
        "probability_at_risk": (
            prediction.get("probability", {}).get("at_risk")
            if prediction.get("probability")
            else None
        ),
        "model_type": prediction.get("model_type"),
    }

    if supabase_client:
        try:
            supabase_client.table("predictions_log").insert(log_entry).execute()
            logger.debug("Predição logada no Supabase")
        except Exception as e:
            logger.error(f"Falha ao logar no Supabase: {e}")

    log_file = MODELS_DIR / "predictions_log.jsonl"
    with open(log_file, "a") as f:
        f.write(json.dumps(log_entry, default=str) + "\n")


def log_model_metrics(metrics: dict, model_type: str, supabase_client=None) -> None:
    """Registra métricas do modelo (Supabase + local)."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_type": model_type,
        **metrics,
    }

    if supabase_client:
        try:
            supabase_client.table("model_metrics").insert(entry).execute()
            logger.info("Métricas logadas no Supabase")
        except Exception as e:
            logger.error(f"Falha ao logar métricas: {e}")

    log_file = MODELS_DIR / "model_metrics.jsonl"
    with open(log_file, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def detect_data_drift(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    report_path: str | Path | None = None,
) -> dict:
    """Detecta data drift entre dados de referência e dados atuais via Evidently."""
    try:
        from evidently.metric_preset import DataDriftPreset
        from evidently.report import Report
    except ImportError:
        logger.warning("Evidently não instalado. pip install evidently")
        return {"error": "Evidently não instalado", "drift_detected": None}

    if report_path is None:
        report_dir = Path("monitoring/reports")
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f"drift_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"

    common_cols = list(set(reference_data.columns) & set(current_data.columns))
    numeric_cols = reference_data[common_cols].select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        logger.warning("Sem colunas numéricas em comum para detecção de drift")
        return {"error": "Sem colunas numéricas comuns", "drift_detected": None}

    ref = reference_data[numeric_cols].copy()
    curr = current_data[numeric_cols].copy()

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref, current_data=curr)
    report.save_html(str(report_path))
    logger.info(f"Relatório de drift salvo em {report_path}")

    report_dict = report.as_dict()
    results_list = report_dict.get("metrics", [])

    drift_detected = False
    drifted_features = []

    for metric_result in results_list:
        metric_res = metric_result.get("result", {})
        if "drift_by_columns" in metric_res:
            for col, col_result in metric_res["drift_by_columns"].items():
                if col_result.get("drift_detected", False):
                    drift_detected = True
                    drifted_features.append(col)

    result = {
        "drift_detected": drift_detected,
        "drifted_features": drifted_features,
        "n_features_analyzed": len(numeric_cols),
        "n_features_drifted": len(drifted_features),
        "report_path": str(report_path),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if drift_detected:
        logger.warning(f"DRIFT DETECTADO em {len(drifted_features)} features: {drifted_features}")
    else:
        logger.info("Nenhum drift detectado")

    return result


def detect_prediction_drift(
    reference_predictions: np.ndarray,
    current_predictions: np.ndarray,
    threshold: float = 0.1,
) -> dict:
    """Detecta drift nas predições comparando distribuições ref vs atual."""
    ref_mean = float(np.mean(reference_predictions))
    curr_mean = float(np.mean(current_predictions))
    ref_std = float(np.std(reference_predictions))
    curr_std = float(np.std(current_predictions))

    mean_diff = abs(curr_mean - ref_mean)
    drift_detected = mean_diff > threshold

    result = {
        "drift_detected": drift_detected,
        "reference_mean": round(ref_mean, 4),
        "current_mean": round(curr_mean, 4),
        "mean_difference": round(mean_diff, 4),
        "reference_std": round(ref_std, 4),
        "current_std": round(curr_std, 4),
        "threshold": threshold,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if drift_detected:
        logger.warning(f"DRIFT DE PREDIÇÃO: diff = {mean_diff:.4f} > threshold {threshold}")
    else:
        logger.info(f"Sem drift de predição: diff = {mean_diff:.4f}")

    return result


def get_prediction_stats(supabase_client=None) -> dict:
    """Retorna estatísticas das predições recentes."""
    if supabase_client:
        try:
            response = (
                supabase_client.table("predictions_log")
                .select("*")
                .order("timestamp", desc=True)
                .limit(1000)
                .execute()
            )
            predictions = response.data
            if predictions:
                df = pd.DataFrame(predictions)
                return {
                    "total_predictions": len(df),
                    "risk_distribution": df["risk_level"].value_counts().to_dict(),
                    "avg_probability_at_risk": df["probability_at_risk"].mean(),
                    "latest_prediction": df.iloc[0]["timestamp"],
                }
        except Exception as e:
            logger.error(f"Falha ao buscar stats do Supabase: {e}")

    # Fallback pro log local
    log_file = MODELS_DIR / "predictions_log.jsonl"
    if log_file.exists():
        entries = []
        with open(log_file) as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line))

        if entries:
            df = pd.DataFrame(entries)
            return {
                "total_predictions": len(df),
                "risk_distribution": df["risk_level"].value_counts().to_dict() if "risk_level" in df.columns else {},
                "avg_probability_at_risk": float(df["probability_at_risk"].mean()) if "probability_at_risk" in df.columns else None,
                "latest_prediction": entries[-1].get("timestamp"),
            }

    return {"total_predictions": 0, "risk_distribution": {}, "avg_probability_at_risk": None}
