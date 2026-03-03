"""Aplicação FastAPI — API de predição de risco de defasagem escolar."""

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from src.utils import API_ENV, MODELS_DIR, setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: carrega modelo, artefatos, Supabase. Shutdown: cleanup."""
    setup_logging()
    logger.info("Iniciando API Passos Mágicos...")

    try:
        from src.predict import Predictor
        predictor = Predictor()
        # Disponibiliza no state da app (sem globals)
        app.state.predictor = predictor
        logger.info(f"Modelo carregado: {type(predictor.model).__name__}")

        eval_path = MODELS_DIR / "evaluation_report.json"
        if eval_path.exists():
            with open(eval_path) as f:
                app.state.evaluation_results = json.load(f)
            logger.info("Resultados de avaliação carregados")
        else:
            app.state.evaluation_results = None

        from src.monitoring import get_supabase_client
        app.state.supabase_client = get_supabase_client()

    except FileNotFoundError as e:
        logger.error(f"Arquivos do modelo não encontrados: {e}")
        logger.warning("API rodando em modo degradado — treine o modelo primeiro!")
        app.state.predictor = None
        app.state.evaluation_results = None
        app.state.supabase_client = None
    except Exception as e:
        logger.error(f"Erro no startup: {e}")
        logger.warning("API em modo degradado")
        app.state.predictor = None
        app.state.evaluation_results = None
        app.state.supabase_client = None

    yield

    logger.info("Desligando API...")


app = FastAPI(
    title="Datathon Educação - Passos Mágicos API",
    description=(
        "API para predição de risco de defasagem escolar dos estudantes "
        "da Associação Passos Mágicos. Datathon PosTech - Machine Learning Engineering."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS aberto pra facilitar desenvolvimento e testes do Streamlit.
# Em produção real, restringir allow_origins aos domínios do frontend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from api.routes import router  # noqa: E402
app.include_router(router)


@app.get("/", tags=["Sistema"])
async def root():
    """Informações gerais da API."""
    return {
        "name": "Datathon Educação - Passos Mágicos API",
        "version": "1.0.0",
        "environment": API_ENV,
        "docs": "/docs",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "predict_batch": "/predict/batch",
            "metrics": "/metrics",
            "drift": "/monitoring/drift",
        },
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("API_PORT", "8000"))
    host = os.getenv("API_HOST", "0.0.0.0")
    uvicorn.run("api.main:app", host=host, port=port, reload=(API_ENV == "development"))
