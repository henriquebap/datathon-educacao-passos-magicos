"""Rotas da API — usa app.state ao invés de variáveis globais."""

from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Request
from loguru import logger

from api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    DriftResponse,
    ErrorResponse,
    HealthResponse,
    MetricsResponse,
    PredictionResponse,
    StudentFeatures,
)

router = APIRouter()


def _get_predictor(request: Request):
    """Extrai o predictor do state da app, ou levanta 503 se não estiver disponível."""
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo não carregado. Verifique os logs do servidor.",
        )
    return predictor


def _get_supabase(request: Request):
    return getattr(request.app.state, "supabase_client", None)


def _get_eval_results(request: Request):
    return getattr(request.app.state, "evaluation_results", None)


@router.get("/health", response_model=HealthResponse, tags=["Sistema"])
async def health_check(request: Request):
    """Verifica status da API e se o modelo está carregado."""
    predictor = getattr(request.app.state, "predictor", None)
    return HealthResponse(
        status="healthy" if predictor is not None else "degraded",
        model_loaded=predictor is not None,
        model_type=type(predictor.model).__name__ if predictor else None,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@router.post(
    "/predict",
    response_model=PredictionResponse,
    responses={500: {"model": ErrorResponse}},
    tags=["Predições"],
)
async def predict(student: StudentFeatures, request: Request):
    """Predição de risco de defasagem para um aluno."""
    predictor = _get_predictor(request)

    try:
        input_data = {k: v for k, v in student.model_dump().items() if v is not None}
        result = predictor.predict(input_data)

        from src.monitoring import log_prediction
        log_prediction(input_data, result, _get_supabase(request))

        return PredictionResponse(
            prediction=result["prediction"],
            risk_level=result["risk_level"],
            probability=result["probability"],
            model_type=result["model_type"],
            timestamp=result["timestamp"],
        )

    except Exception as e:
        logger.error(f"Erro na predição: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["Predições"],
)
async def predict_batch(req: BatchPredictionRequest, request: Request):
    """Predição em lote para múltiplos alunos (inferência vetorizada)."""
    predictor = _get_predictor(request)

    try:
        data_list = [
            {k: v for k, v in s.model_dump().items() if v is not None}
            for s in req.students
        ]

        results = predictor.predict_batch(data_list)

        predictions = [
            PredictionResponse(
                prediction=r["prediction"],
                risk_level=r["risk_level"],
                probability=r["probability"],
                model_type=r["model_type"],
                timestamp=r["timestamp"],
            )
            for r in results
        ]

        return BatchPredictionResponse(
            predictions=predictions,
            total=len(predictions),
        )

    except Exception as e:
        logger.error(f"Erro no batch: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics", response_model=MetricsResponse, tags=["Monitoramento"])
async def get_metrics(request: Request):
    """Retorna métricas do modelo e estatísticas das predições."""
    from src.monitoring import get_prediction_stats

    eval_results = _get_eval_results(request)
    metrics = {}
    if eval_results and "metrics" in eval_results:
        metrics = eval_results["metrics"]

    stats = get_prediction_stats(_get_supabase(request))

    predictor = getattr(request.app.state, "predictor", None)
    model_type = type(predictor.model).__name__ if predictor else "desconhecido"

    n_test_samples = eval_results.get("n_test_samples") if eval_results else None
    confusion_matrix = eval_results.get("confusion_matrix") if eval_results else None
    feature_importance = eval_results.get("feature_importance") if eval_results else None

    return MetricsResponse(
        model_type=model_type,
        metrics=metrics,
        prediction_stats=stats,
        n_test_samples=n_test_samples,
        confusion_matrix=confusion_matrix,
        feature_importance=feature_importance,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@router.get("/monitoring/drift", response_model=DriftResponse, tags=["Monitoramento"])
async def check_drift(request: Request):
    """Verifica drift nos dados e predições."""
    from src.monitoring import get_prediction_stats

    stats = get_prediction_stats(_get_supabase(request))
    total = stats.get("total_predictions", 0)

    if total < 10:
        return DriftResponse(
            drift_detected=None,
            details={
                "message": f"Predições insuficientes para análise (precisa de 10, tem {total})",
                "total_predictions": total,
            },
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    risk_dist = stats.get("risk_distribution", {})
    avg_risk = stats.get("avg_probability_at_risk")

    return DriftResponse(
        drift_detected=False,
        details={
            "total_predictions": total,
            "risk_distribution": risk_dist,
            "avg_probability_at_risk": avg_risk,
            "message": "Monitoramento de drift ativo",
        },
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
