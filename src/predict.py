"""Módulo de predição — carrega modelo treinado e faz inferência."""

from datetime import datetime, timezone

import numpy as np
import pandas as pd
from loguru import logger

from src.train import load_model, load_pipeline_artifacts


class Predictor:
    """Encapsula carregamento do modelo e inferência com preprocessamento."""

    def __init__(self, model_path: str | None = None, pipeline_path: str | None = None):
        self.model = load_model(model_path)
        self.artifacts = load_pipeline_artifacts(pipeline_path)
        self.scaler = self.artifacts.get("scaler")
        self.feature_names = self.artifacts.get("feature_names", [])
        self.encoders = self.artifacts.get("encoders", {})
        logger.info(
            f"Predictor pronto: {type(self.model).__name__}, "
            f"{len(self.feature_names)} features"
        )

    def _prepare_dataframe(self, data: dict | list[dict]) -> pd.DataFrame:
        """Converte entrada (dict ou lista) em DataFrame alinhado ao schema de treino."""
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = pd.DataFrame(data)

        for col, encoder in self.encoders.items():
            if col not in df.columns:
                continue
            if isinstance(encoder, dict):
                df[col] = df[col].map(encoder).fillna(0).astype(int)
            else:
                try:
                    df[col] = encoder.transform(df[col].astype(str))
                except ValueError:
                    df[col] = 0

        for feat in self.feature_names:
            if feat not in df.columns:
                df[feat] = 0

        df = df[self.feature_names]

        if self.scaler is not None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            common_cols = [c for c in numeric_cols if c in self.feature_names]
            if common_cols:
                df[common_cols] = self.scaler.transform(df[common_cols])

        return df

    def predict(self, data: dict) -> dict:
        """Faz predição para um único aluno."""
        ts = datetime.now(timezone.utc)
        X = self._prepare_dataframe(data)

        prediction = int(self.model.predict(X)[0])

        probability = None
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)[0]
            probability = {
                "no_risk": round(float(proba[0]), 4),
                "at_risk": round(float(proba[1]), 4),
            }

        risk_level = "HIGH" if prediction == 1 else "LOW"

        result = {
            "prediction": prediction,
            "risk_level": risk_level,
            "probability": probability,
            "model_type": type(self.model).__name__,
            "timestamp": ts.isoformat(),
            "input_features": data,
        }

        logger.info(
            f"Predição: {risk_level} (classe={prediction}) | "
            f"P(risco)={probability['at_risk'] if probability else 'N/A'}"
        )
        return result

    def predict_batch(self, data_list: list[dict]) -> list[dict]:
        """Faz predição vetorizada para múltiplos alunos de uma vez.

        Ao invés de iterar um por um, monta o DataFrame completo e roda
        predict/predict_proba em batch — bem mais eficiente.
        """
        if not data_list:
            return []

        ts = datetime.now(timezone.utc)
        X = self._prepare_dataframe(data_list)

        predictions = self.model.predict(X)

        probas = None
        if hasattr(self.model, "predict_proba"):
            probas = self.model.predict_proba(X)

        results = []
        model_name = type(self.model).__name__
        for i, pred in enumerate(predictions):
            pred_int = int(pred)
            probability = None
            if probas is not None:
                probability = {
                    "no_risk": round(float(probas[i][0]), 4),
                    "at_risk": round(float(probas[i][1]), 4),
                }

            results.append({
                "prediction": pred_int,
                "risk_level": "HIGH" if pred_int == 1 else "LOW",
                "probability": probability,
                "model_type": model_name,
                "timestamp": ts.isoformat(),
                "input_features": data_list[i],
            })

        logger.info(f"Batch: {len(results)} alunos processados de uma vez")
        return results
