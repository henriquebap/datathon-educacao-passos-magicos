"""Pré-processamento dos dados do dataset PEDE/Passos Mágicos.

Inclui tratamento de missings, encoding categórico, normalização
e extração do target (DEFASAGEM) com remoção de colunas de leakage.
"""

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from src.utils import (
    RANDOM_STATE,
    TEST_SIZE,
    DROP_COLUMNS,
    LEAKAGE_COLUMNS,
    PEDRA_MAP,
    TARGET_COL,
    get_latest_year,
)


def drop_identifier_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove colunas de identificação (NOME, ID, etc.) que não devem entrar como features."""
    cols_to_drop = [c for c in df.columns if any(
        c.lower() == drop_col.lower()
        or c.lower().startswith(drop_col.lower() + "_")
        or c.lower().endswith("_" + drop_col.lower())
        for drop_col in DROP_COLUMNS
    )]
    if cols_to_drop:
        logger.info(f"Removendo colunas de identificação: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop, errors="ignore")
    return df


def handle_missing_values(df: pd.DataFrame, strategy: str = "median") -> pd.DataFrame:
    """Trata valores faltantes. Numéricas: mediana/média. Categóricas: moda."""
    initial_nulls = df.isnull().sum().sum()
    logger.info(f"Valores faltantes antes do tratamento: {initial_nulls}")

    if strategy == "drop":
        df = df.dropna()
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                fill_value = df[col].median() if strategy == "median" else df[col].mean()
                df[col] = df[col].fillna(fill_value)

        categorical_cols = df.select_dtypes(include=["object", "category", "string"]).columns
        for col in categorical_cols:
            if df[col].isnull().any():
                mode_value = df[col].mode()
                if len(mode_value) > 0:
                    df[col] = df[col].fillna(mode_value[0])

    final_nulls = df.isnull().sum().sum()
    logger.info(f"Valores faltantes após tratamento: {final_nulls}")
    return df


def encode_categorical_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Aplica encoding nas colunas categóricas.

    PEDRA recebe mapeamento ordinal (Quartzo < Ágata < Ametista < Topázio).
    Demais colunas texto viram LabelEncoder.
    """
    encoders = {}

    def _is_text(series: pd.Series) -> bool:
        return pd.api.types.is_string_dtype(series) or series.dtype == "object"

    # PEDRA tem ordem semântica, então mapeamento ordinal faz mais sentido
    pedra_cols = [c for c in df.columns if "PEDRA" in c.upper()]
    for col in pedra_cols:
        if col in df.columns and _is_text(df[col]):
            df[col] = df[col].map(PEDRA_MAP).fillna(0).astype(int)
            encoders[col] = PEDRA_MAP
            logger.info(f"Encoded {col} com mapeamento PEDRA (ordinal)")

    # Restante: LabelEncoder genérico
    remaining_cat = [col for col in df.columns if _is_text(df[col])]
    for col in remaining_cat:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        logger.info(f"Label-encoded {col} ({len(le.classes_)} classes)")

    return df, encoders


def normalize_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Normaliza features numéricas com StandardScaler (fit no train, transform no test)."""
    scaler = StandardScaler()
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

    logger.info(f"Normalizadas {len(numeric_cols)} features numéricas")
    return X_train_scaled, X_test_scaled, scaler


def extract_target(df: pd.DataFrame, target_year: int | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Extrai a variável alvo (DEFASAGEM) e remove colunas de leakage.

    A DEFASAGEM indica se o aluno está atrasado em relação à série ideal.
    Valores negativos = defasado = em risco (classe 1).

    Colunas IAN e NIVEL_IDEAL são removidas porque representam
    a mesma informação que o target (data leakage).
    """
    if target_year is None:
        target_year = get_latest_year(df)

    target_col = f"{TARGET_COL}_{target_year}"

    if target_col not in df.columns:
        # Fallback: tenta encontrar qualquer coluna DEFASAGEM
        defas_cols = sorted(
            [c for c in df.columns if c.upper().startswith(TARGET_COL)],
            reverse=True,
        )
        if defas_cols:
            target_col = defas_cols[0]
            logger.info(f"Usando coluna target disponível: {target_col}")
        elif TARGET_COL in df.columns:
            target_col = TARGET_COL
        else:
            raise ValueError(
                f"Coluna target '{TARGET_COL}' não encontrada. "
                f"Colunas disponíveis: {list(df.columns)}"
            )

    y = df[target_col].copy()

    # Linhas sem target não servem pra treino
    nan_mask = y.isna()
    if nan_mask.any():
        logger.warning(f"Removendo {nan_mask.sum()} linhas com target NaN")
        y = y[~nan_mask]
        df = df.loc[y.index]

    # Binarização: defasagem negativa = em risco
    if y.nunique() > 2:
        logger.info(f"Target com {y.nunique()} valores únicos — binarizando: < 0 = em risco")
        y = (y < 0).astype(int)
    else:
        y = y.astype(int)

    # Remove target + colunas de leakage das features
    defasagem_cols = [c for c in df.columns if TARGET_COL in c.upper()]
    leakage_cols = [
        c for c in df.columns
        if any(lk.upper() in c.upper() for lk in LEAKAGE_COLUMNS)
    ]
    cols_to_remove = list(set(defasagem_cols + leakage_cols))
    if leakage_cols:
        logger.info(f"Removendo colunas de leakage: {sorted(leakage_cols)}")
    X = df.loc[y.index].drop(columns=cols_to_remove, errors="ignore")

    logger.info(
        f"Target: {target_col} | Distribuição: "
        f"0={int((y == 0).sum())} ({(y == 0).mean():.1%}), "
        f"1={int((y == 1).sum())} ({(y == 1).mean():.1%})"
    )
    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split estratificado em treino/teste."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    logger.info(
        f"Treino: {len(X_train)} | Teste: {len(X_test)} | "
        f"Dist treino: 0={int((y_train == 0).sum())}, 1={int((y_train == 1).sum())}"
    )
    return X_train, X_test, y_train, y_test


def preprocess_pipeline(
    df: pd.DataFrame,
    target_year: int | None = None,
    missing_strategy: str = "median",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, dict]:
    """Pipeline completo de pré-processamento.

    Etapas: drop IDs → tratar missings → extrair target → encoding → split → normalizar.
    """
    logger.info("=" * 50)
    logger.info("Iniciando pré-processamento")
    logger.info("=" * 50)

    df = drop_identifier_columns(df)
    df = handle_missing_values(df, strategy=missing_strategy)
    X, y = extract_target(df, target_year=target_year)
    X, encoders = encode_categorical_columns(X)
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train, X_test, scaler = normalize_features(X_train, X_test)

    artifacts = {
        "encoders": encoders,
        "scaler": scaler,
        "feature_names": X_train.columns.tolist(),
        "target_year": target_year or get_latest_year(df) if TARGET_COL not in df.columns else None,
    }

    logger.info("Pré-processamento concluído")
    return X_train, X_test, y_train, y_test, artifacts
