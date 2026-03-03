"""Schemas Pydantic para validação de request/response da API."""

from typing import Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "healthy"
    model_loaded: bool
    model_type: str | None = None
    api_version: str = "1.0.0"
    timestamp: str


class StudentFeatures(BaseModel):
    """Dados de entrada do estudante para predição.

    Contém os indicadores educacionais utilizados pelo modelo.
    Campos opcionais para lidar com dados parciais.

    Nota: IAN e NIVEL_IDEAL foram removidos intencionalmente pois são
    proxies diretos da variável alvo (DEFASAGEM), o que causaria data leakage.
    """

    # Indicadores numéricos (escala 0-10)
    INDE_2020: Optional[float] = Field(None, ge=0, le=10, description="Índice de Desenvolvimento Educacional 2020")
    INDE_2021: Optional[float] = Field(None, ge=0, le=10, description="Índice de Desenvolvimento Educacional 2021")
    INDE_2022: Optional[float] = Field(None, ge=0, le=10, description="Índice de Desenvolvimento Educacional 2022")

    IAA_2020: Optional[float] = Field(None, ge=0, le=10, description="Indicador de Autoavaliação 2020")
    IAA_2021: Optional[float] = Field(None, ge=0, le=10, description="Indicador de Autoavaliação 2021")
    IAA_2022: Optional[float] = Field(None, ge=0, le=10, description="Indicador de Autoavaliação 2022")

    IEG_2020: Optional[float] = Field(None, ge=0, le=10, description="Indicador de Engajamento 2020")
    IEG_2021: Optional[float] = Field(None, ge=0, le=10, description="Indicador de Engajamento 2021")
    IEG_2022: Optional[float] = Field(None, ge=0, le=10, description="Indicador de Engajamento 2022")

    IPS_2020: Optional[float] = Field(None, ge=0, le=10, description="Indicador Psicossocial 2020")
    IPS_2021: Optional[float] = Field(None, ge=0, le=10, description="Indicador Psicossocial 2021")
    IPS_2022: Optional[float] = Field(None, ge=0, le=10, description="Indicador Psicossocial 2022")

    IDA_2020: Optional[float] = Field(None, ge=0, le=10, description="Indicador de Adequação de Nível 2020")
    IDA_2021: Optional[float] = Field(None, ge=0, le=10, description="Indicador de Adequação de Nível 2021")
    IDA_2022: Optional[float] = Field(None, ge=0, le=10, description="Indicador de Adequação de Nível 2022")

    IPP_2020: Optional[float] = Field(None, ge=0, le=10, description="Indicador Psicopedagógico 2020")
    IPP_2021: Optional[float] = Field(None, ge=0, le=10, description="Indicador Psicopedagógico 2021")
    IPP_2022: Optional[float] = Field(None, ge=0, le=10, description="Indicador Psicopedagógico 2022")

    IPV_2020: Optional[float] = Field(None, ge=0, le=10, description="Indicador do Ponto de Virada 2020")
    IPV_2021: Optional[float] = Field(None, ge=0, le=10, description="Indicador do Ponto de Virada 2021")
    IPV_2022: Optional[float] = Field(None, ge=0, le=10, description="Indicador do Ponto de Virada 2022")

    # Idade
    IDADE_ALUNO_2020: Optional[int] = Field(None, ge=5, le=25, description="Idade do aluno em 2020")
    IDADE_ALUNO_2021: Optional[int] = Field(None, ge=5, le=25, description="Idade do aluno em 2021")
    IDADE_ALUNO_2022: Optional[int] = Field(None, ge=5, le=25, description="Idade do aluno em 2022")

    # Categóricas
    PEDRA_2020: Optional[str] = Field(None, description="Classificação PEDRA 2020 (Quartzo/Ágata/Ametista/Topázio)")
    PEDRA_2021: Optional[str] = Field(None, description="Classificação PEDRA 2021")
    PEDRA_2022: Optional[str] = Field(None, description="Classificação PEDRA 2022")

    FASE_2020: Optional[int] = Field(None, ge=0, le=8, description="Fase no programa 2020")
    FASE_2021: Optional[int] = Field(None, ge=0, le=8, description="Fase no programa 2021")
    FASE_2022: Optional[int] = Field(None, ge=0, le=8, description="Fase no programa 2022")

    TURMA_2020: Optional[str] = Field(None, description="Turma 2020")
    TURMA_2021: Optional[str] = Field(None, description="Turma 2021")
    TURMA_2022: Optional[str] = Field(None, description="Turma 2022")

    # Binárias
    PONTO_VIRADA_2020: Optional[int] = Field(None, ge=0, le=1, description="Atingiu ponto de virada 2020")
    PONTO_VIRADA_2021: Optional[int] = Field(None, ge=0, le=1, description="Atingiu ponto de virada 2021")
    PONTO_VIRADA_2022: Optional[int] = Field(None, ge=0, le=1, description="Atingiu ponto de virada 2022")

    BOLSISTA_2020: Optional[int] = Field(None, ge=0, le=1, description="Bolsista 2020")
    BOLSISTA_2021: Optional[int] = Field(None, ge=0, le=1, description="Bolsista 2021")
    BOLSISTA_2022: Optional[int] = Field(None, ge=0, le=1, description="Bolsista 2022")

    # Anos no programa
    ANOS_PM_2020: Optional[int] = Field(None, ge=0, le=15, description="Anos no programa Passos Mágicos 2020")
    ANOS_PM_2021: Optional[int] = Field(None, ge=0, le=15, description="Anos no programa Passos Mágicos 2021")
    ANOS_PM_2022: Optional[int] = Field(None, ge=0, le=15, description="Anos no programa Passos Mágicos 2022")

    model_config = {"json_schema_extra": {
        "examples": [
            {
                "INDE_2022": 7.5,
                "IAA_2022": 6.8,
                "IEG_2022": 7.2,
                "IPS_2022": 6.5,
                "IDA_2022": 7.0,
                "IPP_2022": 5.5,
                "IPV_2022": 5.8,
                "IDADE_ALUNO_2022": 14,
                "PEDRA_2022": "Ametista",
                "FASE_2022": 4,
                "TURMA_2022": "B",
                "PONTO_VIRADA_2022": 0,
                "BOLSISTA_2022": 1,
                "ANOS_PM_2022": 3,
            }
        ]
    }}


class PredictionResponse(BaseModel):
    """Resposta de predição individual."""
    prediction: int = Field(description="0 = Sem risco, 1 = Em risco")
    risk_level: str = Field(description="LOW ou HIGH para risco de defasagem")
    probability: Optional[dict] = Field(None, description="Probabilidade por classe")
    model_type: str = Field(description="Tipo do modelo utilizado")
    timestamp: str = Field(description="Timestamp da predição")


class BatchPredictionRequest(BaseModel):
    students: list[StudentFeatures] = Field(description="Lista de dados de estudantes")


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    total: int


class MetricsResponse(BaseModel):
    model_type: str
    metrics: dict
    prediction_stats: dict
    timestamp: str


class DriftResponse(BaseModel):
    drift_detected: Optional[bool]
    details: dict
    timestamp: str


class ErrorResponse(BaseModel):
    error: str
    detail: str
    timestamp: str
