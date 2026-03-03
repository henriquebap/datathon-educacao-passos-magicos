"""Drift monitoring dashboard using Evidently AI."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from src.utils import DATA_PROCESSED_DIR, MODELS_DIR, setup_logging


def generate_drift_report(
    reference_path: str | Path | None = None,
    current_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> dict:
    """Generate a comprehensive drift report.

    Args:
        reference_path: Path to reference (training) data CSV.
        current_path: Path to current (production) data CSV.
        output_path: Path to save the HTML report.

    Returns:
        Dictionary with drift detection results.
    """
    from src.monitoring import detect_data_drift

    if reference_path is None:
        reference_path = DATA_PROCESSED_DIR / "train_data.csv"
    if current_path is None:
        current_path = DATA_PROCESSED_DIR / "production_data.csv"
    if output_path is None:
        output_dir = Path("monitoring/reports")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "drift_report.html"

    reference_path = Path(reference_path)
    current_path = Path(current_path)

    if not reference_path.exists():
        logger.error(f"Reference data not found: {reference_path}")
        return {"error": "Reference data not found"}

    if not current_path.exists():
        logger.error(f"Current data not found: {current_path}")
        return {"error": "Current data not found"}

    reference = pd.read_csv(reference_path)
    current = pd.read_csv(current_path)

    logger.info(f"Reference data: {reference.shape}, Current data: {current.shape}")

    result = detect_data_drift(reference, current, report_path=output_path)
    return result


def get_prediction_history() -> pd.DataFrame:
    """Load prediction history from local log.

    Returns:
        DataFrame with prediction history.
    """
    log_file = MODELS_DIR / "predictions_log.jsonl"

    if not log_file.exists():
        logger.warning("No prediction log found")
        return pd.DataFrame()

    entries = []
    with open(log_file) as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))

    if not entries:
        return pd.DataFrame()

    return pd.DataFrame(entries)


def generate_monitoring_summary() -> dict:
    """Generate a summary of the monitoring status.

    Returns:
        Dictionary with monitoring summary.
    """
    summary = {"status": "active"}

    # Prediction history
    history = get_prediction_history()
    if not history.empty:
        summary["total_predictions"] = len(history)
        summary["risk_distribution"] = (
            history["risk_level"].value_counts().to_dict()
            if "risk_level" in history.columns else {}
        )
        if "probability_at_risk" in history.columns:
            summary["avg_risk_probability"] = round(
                history["probability_at_risk"].astype(float).mean(), 4
            )
        summary["latest_prediction"] = history.iloc[-1].get("timestamp", "N/A")
    else:
        summary["total_predictions"] = 0
        summary["message"] = "No predictions logged yet"

    # Model metrics
    metrics_file = MODELS_DIR / "model_metrics.jsonl"
    if metrics_file.exists():
        with open(metrics_file) as f:
            lines = f.readlines()
        if lines:
            latest = json.loads(lines[-1])
            summary["latest_model_metrics"] = latest

    # Evaluation report
    eval_file = MODELS_DIR / "evaluation_report.json"
    if eval_file.exists():
        with open(eval_file) as f:
            eval_data = json.load(f)
        summary["evaluation_metrics"] = eval_data.get("metrics", {})

    return summary


if __name__ == "__main__":
    setup_logging()
    logger.info("Generating monitoring summary...")
    summary = generate_monitoring_summary()
    logger.info(f"Summary: {json.dumps(summary, indent=2, default=str)}")

    # Try to generate drift report
    try:
        drift_result = generate_drift_report()
        logger.info(f"Drift result: {json.dumps(drift_result, indent=2, default=str)}")
    except Exception as e:
        logger.warning(f"Could not generate drift report: {e}")
