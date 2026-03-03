"""Feature engineering — cria features temporais, compostas e de interação.

A ideia é enriquecer o dataset com sinais derivados que o modelo
não conseguiria extrair sozinho a partir das colunas brutas.
"""

import numpy as np
import pandas as pd
from loguru import logger

from src.utils import (
    NUMERIC_INDICATORS,
    YEARS,
    get_latest_year,
)


def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cria features de evolução temporal entre anos (diffs, trends, média, desvio)."""
    available_years = sorted(set(
        int(col.rsplit("_", 1)[1])
        for col in df.columns
        if col.rsplit("_", 1)[-1].isdigit()
        and 2000 <= int(col.rsplit("_", 1)[1]) <= 2030
    ))

    if len(available_years) < 2:
        logger.warning("Menos de 2 anos disponíveis, pulando features temporais")
        return df

    numeric_cols = set(df.select_dtypes(include=[np.number]).columns)

    for indicator in NUMERIC_INDICATORS:
        indicator_years = [
            y for y in available_years
            if f"{indicator}_{y}" in df.columns and f"{indicator}_{y}" in numeric_cols
        ]

        if len(indicator_years) < 2:
            continue

        # Diff ano-a-ano
        for i in range(1, len(indicator_years)):
            prev_year = indicator_years[i - 1]
            curr_year = indicator_years[i]
            col_prev = f"{indicator}_{prev_year}"
            col_curr = f"{indicator}_{curr_year}"
            diff_col = f"{indicator}_diff_{prev_year}_{curr_year}"

            if col_prev in numeric_cols and col_curr in numeric_cols:
                df[diff_col] = pd.to_numeric(df[col_curr], errors="coerce") - pd.to_numeric(df[col_prev], errors="coerce")

        # Tendência geral (primeiro → último ano)
        first_year = indicator_years[0]
        last_year = indicator_years[-1]
        col_first = f"{indicator}_{first_year}"
        col_last = f"{indicator}_{last_year}"

        if col_first in numeric_cols and col_last in numeric_cols:
            df[f"{indicator}_trend"] = pd.to_numeric(df[col_last], errors="coerce") - pd.to_numeric(df[col_first], errors="coerce")

        # Média e desvio entre os anos
        year_cols = [f"{indicator}_{y}" for y in indicator_years if f"{indicator}_{y}" in numeric_cols]
        if year_cols:
            numeric_subset = df[year_cols].apply(pd.to_numeric, errors="coerce")
            df[f"{indicator}_mean"] = numeric_subset.mean(axis=1)
            df[f"{indicator}_std"] = numeric_subset.std(axis=1).fillna(0)

    logger.info(f"Features temporais criadas a partir de {len(available_years)} anos")
    return df


def create_composite_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Cria indicadores compostos combinando múltiplas variáveis."""
    latest_year = get_latest_year(df)

    def _safe_numeric(series):
        return pd.to_numeric(series, errors="coerce")

    # Composto acadêmico: INDE + IAA + IDA
    academic_cols = [f"INDE_{latest_year}", f"IAA_{latest_year}", f"IDA_{latest_year}"]
    available = [c for c in academic_cols if c in df.columns]
    if available:
        df["ACADEMIC_COMPOSITE"] = df[available].apply(_safe_numeric).mean(axis=1)

    # Composto de engajamento: IEG + IPS
    engagement_cols = [f"IEG_{latest_year}", f"IPS_{latest_year}"]
    available = [c for c in engagement_cols if c in df.columns]
    if available:
        df["ENGAGEMENT_COMPOSITE"] = df[available].apply(_safe_numeric).mean(axis=1)

    # Score de risco: inverso da média dos indicadores principais
    risk_cols = [f"INDE_{latest_year}", f"IEG_{latest_year}", f"IPS_{latest_year}"]
    available = [c for c in risk_cols if c in df.columns]
    if available:
        df["RISK_SCORE"] = 10 - df[available].apply(_safe_numeric).mean(axis=1)

    # Índice de progresso: ponto de virada * anos no programa
    pv_col = f"PONTO_VIRADA_{latest_year}"
    anos_col = f"ANOS_PM_{latest_year}"
    if pv_col in df.columns and anos_col in df.columns:
        df["PROGRESS_INDEX"] = _safe_numeric(df[pv_col]) * _safe_numeric(df[anos_col])

    # Gap idade-fase: idade - fase - 6 (idade esperada pra fase 0)
    age_col = f"IDADE_ALUNO_{latest_year}"
    fase_col = f"FASE_{latest_year}"
    if age_col in df.columns and fase_col in df.columns:
        df["AGE_PHASE_GAP"] = _safe_numeric(df[age_col]) - _safe_numeric(df[fase_col]) - 6

    logger.info("Indicadores compostos criados")
    return df


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cria features de interação entre indicadores-chave."""
    latest_year = get_latest_year(df)

    def _to_num(series):
        return pd.to_numeric(series, errors="coerce").fillna(0)

    inde_col = f"INDE_{latest_year}"
    ieg_col = f"IEG_{latest_year}"
    if inde_col in df.columns and ieg_col in df.columns:
        df["INDE_IEG_INTERACTION"] = _to_num(df[inde_col]) * _to_num(df[ieg_col])

    ips_col = f"IPS_{latest_year}"
    iaa_col = f"IAA_{latest_year}"
    if ips_col in df.columns and iaa_col in df.columns:
        df["IPS_IAA_INTERACTION"] = _to_num(df[ips_col]) * _to_num(df[iaa_col])

    bolsista_col = f"BOLSISTA_{latest_year}"
    if bolsista_col in df.columns and inde_col in df.columns:
        df["BOLSISTA_INDE_INTERACTION"] = _to_num(df[bolsista_col]) * _to_num(df[inde_col])

    logger.info("Features de interação criadas")
    return df


def select_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    method: str = "importance",
    top_k: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Seleção de features por importância (RandomForest) ou correlação.

    Se top_k for None, mantém todas as features.
    """
    if method == "all" or top_k is None:
        logger.info(f"Mantendo todas as {len(X_train.columns)} features")
        return X_train, X_test, X_train.columns.tolist()

    if method == "importance":
        from sklearn.ensemble import RandomForestClassifier

        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        importances = pd.Series(rf.feature_importances_, index=X_train.columns)
        selected = importances.nlargest(top_k).index.tolist()

    elif method == "correlation":
        correlations = X_train.corrwith(y_train).abs()
        selected = correlations.nlargest(top_k).index.tolist()

    else:
        raise ValueError(f"Método desconhecido: {method}")

    logger.info(f"Selecionadas top {len(selected)} features via {method}: {selected[:5]}...")
    return X_train[selected], X_test[selected], selected


def feature_engineering_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline completo de feature engineering.

    Roda na ordem: temporais → compostas → interações.
    """
    logger.info("=" * 50)
    logger.info("Iniciando feature engineering")
    logger.info("=" * 50)

    initial_cols = len(df.columns)

    df = create_temporal_features(df)
    df = create_composite_indicators(df)
    df = create_interaction_features(df)

    final_cols = len(df.columns)
    logger.info(f"Feature engineering concluído: {initial_cols} → {final_cols} features (+{final_cols - initial_cols})")

    return df
