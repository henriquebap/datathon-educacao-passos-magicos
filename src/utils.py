"""Utilitários gerais e configuração do pipeline de ML — Datathon Passos Mágicos."""

import os
import json
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

# ── Caminhos do projeto ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

for _dir in (DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ── Hiperparâmetros e constantes ─────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

YEARS = [2020, 2021, 2022]

# Indicadores numéricos presentes em cada ano
NUMERIC_INDICATORS = [
    "INDE", "IAA", "IEG", "IPS", "IDA", "IPP", "IPV",
]

CATEGORICAL_INDICATORS = ["PEDRA", "FASE", "TURMA"]
BINARY_INDICATORS = ["PONTO_VIRADA", "BOLSISTA"]

TARGET_COL = "DEFASAGEM"

# Colunas que devem ser removidas antes do treinamento (identificadores)
DROP_COLUMNS = ["NOME", "NOME_COMPLETO", "ID", "MATRICULA"]

# Colunas que causam data leakage — são proxies diretos da variável alvo.
# IAN (Indicador de Adequação de Nível) e NIVEL_IDEAL refletem diretamente
# se o aluno está na série correta, ou seja, a própria definição de defasagem.
LEAKAGE_COLUMNS = ["IAN", "NIVEL_IDEAL"]

# Mapeamento PEDRA: classificação ordinal de desempenho dos alunos.
# Inclui variantes com e sem acento para robustez contra diferenças no CSV.
PEDRA_MAP = {
    "Quartzo": 1,
    "Ágata": 2,
    "Agata": 2,
    "Ametista": 3,
    "Topázio": 4,
    "Topazio": 4,
}

# ── Variáveis de ambiente ────────────────────────────────────────────
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_REPO_ID = os.getenv("HF_REPO_ID", "")
MODEL_PATH = os.getenv("MODEL_PATH", str(MODELS_DIR / "model.joblib"))
PIPELINE_PATH = os.getenv("PIPELINE_PATH", str(MODELS_DIR / "pipeline.joblib"))
API_ENV = os.getenv("API_ENV", "development")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


def setup_logging(level: str = LOG_LEVEL) -> None:
    """Configura o loguru com saída em arquivo e console."""
    logger.remove()
    logger.add(
        "logs/pipeline_{time:YYYY-MM-DD}.log",
        rotation="1 day",
        retention="30 days",
        level=level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {module}:{function}:{line} | {message}",
    )
    logger.add(
        lambda msg: print(msg, end=""),
        level=level,
        format="{time:HH:mm:ss} | {level} | {message}",
    )


def load_raw_data(filepath: str | Path | None = None) -> pd.DataFrame:
    """Carrega o dataset bruto (CSV ou Excel) do diretório data/raw/.

    Se nenhum path for passado, tenta detectar automaticamente o arquivo.
    """
    if filepath is None:
        raw_files = list(DATA_RAW_DIR.glob("*.csv")) + list(DATA_RAW_DIR.glob("*.xlsx"))
        if not raw_files:
            raise FileNotFoundError(
                f"Nenhum CSV ou Excel encontrado em {DATA_RAW_DIR}. "
                "Coloque o dataset PEDE lá."
            )
        filepath = raw_files[0]
        logger.info(f"Arquivo detectado: {filepath.name}")

    filepath = Path(filepath)

    if filepath.suffix == ".csv":
        df = pd.read_csv(filepath, sep=None, engine="python")
    elif filepath.suffix in (".xlsx", ".xls"):
        df = pd.read_excel(filepath)
    else:
        raise ValueError(f"Formato não suportado: {filepath.suffix}")

    logger.info(f"Carregados {len(df)} registros, {len(df.columns)} colunas de {filepath.name}")
    return df


def save_processed_data(df: pd.DataFrame, filename: str = "processed.csv") -> Path:
    """Salva dados processados em CSV."""
    path = DATA_PROCESSED_DIR / filename
    df.to_csv(path, index=False)
    logger.info(f"Dados processados salvos em {path}")
    return path


def generate_synthetic_data(n_samples: int = 500, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Gera dados sintéticos no schema do dataset PEDE para testes.

    Útil para rodar o pipeline sem o CSV real.
    """
    rng = np.random.RandomState(seed)
    data = {}

    for year in YEARS:
        for indicator in NUMERIC_INDICATORS:
            col = f"{indicator}_{year}"
            base = rng.uniform(2.0, 9.0, n_samples)
            noise = rng.normal(0, 0.5, n_samples)
            data[col] = np.clip(base + noise, 0, 10).round(2)

        base_age = rng.randint(7, 18, n_samples)
        year_offset = year - YEARS[0]
        data[f"IDADE_ALUNO_{year}"] = base_age + year_offset

        pedra_options = ["Quartzo", "Ágata", "Ametista", "Topázio"]
        data[f"PEDRA_{year}"] = rng.choice(pedra_options, n_samples)

        data[f"FASE_{year}"] = rng.randint(0, 9, n_samples)

        turma_options = ["A", "B", "C", "D", "E"]
        data[f"TURMA_{year}"] = rng.choice(turma_options, n_samples)

        data[f"PONTO_VIRADA_{year}"] = rng.choice([0, 1], n_samples, p=[0.7, 0.3])
        data[f"BOLSISTA_{year}"] = rng.choice([0, 1], n_samples, p=[0.8, 0.2])
        data[f"ANOS_PM_{year}"] = rng.randint(1, 8, n_samples)

        # Target correlacionado com os indicadores
        inde_col = f"INDE_{year}"
        iaa_col = f"IAA_{year}"
        risk_score = (10 - data[inde_col]) * 0.4 + (10 - data[iaa_col]) * 0.3 + rng.uniform(0, 3, n_samples)
        data[f"DEFASAGEM_{year}"] = (risk_score > 5).astype(int)

    df = pd.DataFrame(data)

    # Insere ~5% de NaN nas colunas numéricas (simula dados faltantes reais)
    mask = rng.random(df.shape) < 0.05
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        col_mask = mask[:, df.columns.get_loc(col)]
        df.loc[col_mask, col] = np.nan

    logger.info(f"Dados sintéticos gerados: {n_samples} amostras, {len(df.columns)} colunas")
    return df


def get_latest_year(df: pd.DataFrame) -> int:
    """Identifica o ano mais recente presente nos sufixos das colunas."""
    years_found = set()
    for col in df.columns:
        parts = col.rsplit("_", 1)
        if len(parts) == 2:
            try:
                year = int(parts[1])
                if 2000 <= year <= 2030:
                    years_found.add(year)
            except ValueError:
                continue

    if not years_found:
        raise ValueError("Nenhuma coluna com sufixo de ano encontrada no dataset")

    latest = max(years_found)
    logger.info(f"Anos encontrados: {sorted(years_found)}, mais recente: {latest}")
    return latest


def get_year_columns(df: pd.DataFrame, year: int) -> list[str]:
    """Retorna todas as colunas de um ano específico."""
    return [col for col in df.columns if col.endswith(f"_{year}")]
