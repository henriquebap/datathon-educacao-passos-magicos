"""Pagina de Analise Exploratoria de Dados."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import (
    NUMERIC_INDICATORS,
    YEARS,
    PEDRA_MAP,
    DATA_RAW_DIR,
    generate_synthetic_data,
)

st.set_page_config(page_title="EDA - Passos Mágicos", page_icon="📊", layout="wide")
st.title("📊 Análise Exploratória dos Dados")
st.divider()


@st.cache_data
def load_data():
    """Carrega dados reais ou sinteticos para o EDA."""
    raw_files = list(DATA_RAW_DIR.glob("*.csv")) + list(DATA_RAW_DIR.glob("*.xlsx"))
    csv_files = [f for f in raw_files if "PEDE" in f.name.upper() or "PASSOS" in f.name.upper()]

    for candidates, label in [(csv_files, "Dados reais"), (raw_files, "Dados")]:
        if not candidates:
            continue
        filepath = candidates[0]
        try:
            if filepath.suffix == ".csv":
                df = pd.read_csv(filepath, sep=None, engine="python")
            else:
                df = pd.read_excel(filepath)
            st.sidebar.success(f"{label}: {filepath.name}")
            return df, False
        except Exception as e:
            st.sidebar.error(f"Erro ao ler {filepath.name}: {e}")

    df = generate_synthetic_data(n_samples=500)
    st.sidebar.warning("Usando dados sintéticos (coloque o CSV real em data/raw/)")
    return df, True


df, is_synthetic = load_data()

if is_synthetic:
    st.info(
        "Os gráficos abaixo utilizam **dados sintéticos** para demonstração. "
        "Coloque o dataset PEDE em `data/raw/` para análise com dados reais."
    )

# --- Visao Geral ---
st.header("1. Visão Geral do Dataset")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Registros", f"{len(df):,}")
col2.metric("Colunas", f"{len(df.columns)}")
col3.metric("Valores Nulos", f"{df.isnull().sum().sum():,}")
col4.metric("% Nulos", f"{df.isnull().mean().mean():.1%}")

with st.expander("Ver amostra dos dados"):
    st.dataframe(df.head(20), use_container_width=True)

with st.expander("Estatísticas descritivas"):
    st.dataframe(df.describe().round(2), use_container_width=True)

st.divider()

# --- Distribuicao dos Indicadores ---
st.header("2. Distribuição dos Indicadores por Ano")

latest_year = max(YEARS)
selected_year = st.selectbox("Selecione o ano:", YEARS, index=len(YEARS) - 1)

available_indicators = [
    ind for ind in NUMERIC_INDICATORS if f"{ind}_{selected_year}" in df.columns
]

if available_indicators:
    cols_to_plot = [f"{ind}_{selected_year}" for ind in available_indicators]
    fig = px.box(
        df[cols_to_plot].melt(var_name="Indicador", value_name="Valor"),
        x="Indicador",
        y="Valor",
        color="Indicador",
        title=f"Distribuição dos Indicadores — {selected_year}",
    )
    fig.update_layout(showlegend=False, xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        selected_ind = st.selectbox("Histograma do indicador:", available_indicators)
        col_name = f"{selected_ind}_{selected_year}"
        fig_hist = px.histogram(
            df, x=col_name, nbins=30, title=f"Distribuição de {col_name}",
            marginal="box", color_discrete_sequence=["#636EFA"],
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_b:
        inde_col = f"INDE_{selected_year}"
        if inde_col in df.columns:
            fig_inde = px.histogram(
                df, x=inde_col, nbins=30,
                title=f"INDE {selected_year} — Índice de Desenvolvimento",
                marginal="violin", color_discrete_sequence=["#EF553B"],
            )
            st.plotly_chart(fig_inde, use_container_width=True)
else:
    st.warning(f"Nenhum indicador encontrado para o ano {selected_year}")

st.divider()

# --- Correlacao ---
st.header("3. Matriz de Correlação")

year_for_corr = st.selectbox("Ano para correlação:", YEARS, index=len(YEARS) - 1, key="corr_year")
corr_cols = [
    f"{ind}_{year_for_corr}" for ind in NUMERIC_INDICATORS
    if f"{ind}_{year_for_corr}" in df.columns
]

if len(corr_cols) >= 2:
    corr_matrix = df[corr_cols].corr()
    labels = [c.replace(f"_{year_for_corr}", "") for c in corr_cols]

    fig_corr = px.imshow(
        corr_matrix.values,
        x=labels, y=labels,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title=f"Correlação entre Indicadores — {year_for_corr}",
        text_auto=".2f",
    )
    fig_corr.update_layout(width=700, height=600)
    st.plotly_chart(fig_corr, use_container_width=True)

st.divider()

# --- Evolucao Temporal ---
st.header("4. Evolução Temporal dos Indicadores")

selected_indicators = st.multiselect(
    "Indicadores para comparar:",
    NUMERIC_INDICATORS,
    default=["INDE", "IAA", "IEG"],
)

if selected_indicators:
    means = {}
    for year in YEARS:
        year_means = {}
        for ind in selected_indicators:
            col = f"{ind}_{year}"
            if col in df.columns:
                year_means[ind] = df[col].mean()
        means[year] = year_means

    means_df = pd.DataFrame(means).T
    means_df.index.name = "Ano"

    fig_evolution = px.line(
        means_df.reset_index(),
        x="Ano", y=selected_indicators,
        title="Evolução da Média dos Indicadores ao Longo dos Anos",
        markers=True,
    )
    fig_evolution.update_layout(yaxis_title="Média", xaxis_title="Ano")
    st.plotly_chart(fig_evolution, use_container_width=True)

st.divider()

# --- PEDRA Classification ---
st.header("5. Classificação PEDRA")

pedra_col = f"PEDRA_{latest_year}"
if pedra_col in df.columns:
    pedra_counts = df[pedra_col].value_counts()

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        fig_pedra = px.pie(
            values=pedra_counts.values,
            names=pedra_counts.index,
            title=f"Distribuição PEDRA — {latest_year}",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        st.plotly_chart(fig_pedra, use_container_width=True)

    with col_p2:
        fig_pedra_bar = px.bar(
            x=pedra_counts.index,
            y=pedra_counts.values,
            title=f"Contagem por Classificação PEDRA — {latest_year}",
            labels={"x": "PEDRA", "y": "Quantidade"},
            color=pedra_counts.index,
        )
        st.plotly_chart(fig_pedra_bar, use_container_width=True)

st.divider()

# --- Target (Defasagem) ---
st.header("6. Análise da Variável Alvo (Defasagem)")

target_col = f"DEFASAGEM_{latest_year}"
if target_col in df.columns:
    target_counts = df[target_col].dropna().astype(int).value_counts().sort_index()
    labels_map = {0: "Sem Risco", 1: "Em Risco"}

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        fig_target = px.pie(
            values=target_counts.values,
            names=[labels_map.get(i, str(i)) for i in target_counts.index],
            title=f"Distribuição de Defasagem — {latest_year}",
            color_discrete_sequence=["#2ecc71", "#e74c3c"],
        )
        st.plotly_chart(fig_target, use_container_width=True)

    with col_t2:
        inde_col = f"INDE_{latest_year}"
        if inde_col in df.columns:
            df_plot = df[[inde_col, target_col]].dropna().copy()
            df_plot[target_col] = df_plot[target_col].astype(int).map(labels_map)
            fig_violin = px.violin(
                df_plot, y=inde_col, color=target_col,
                title=f"INDE por Status de Defasagem — {latest_year}",
                box=True, points="outliers",
            )
            st.plotly_chart(fig_violin, use_container_width=True)

st.divider()

# --- Valores Ausentes ---
st.header("7. Valores Ausentes")

null_counts = df.isnull().sum()
null_counts = null_counts[null_counts > 0].sort_values(ascending=False).head(20)

if len(null_counts) > 0:
    fig_null = px.bar(
        x=null_counts.index,
        y=null_counts.values,
        title="Top 20 Colunas com Valores Ausentes",
        labels={"x": "Coluna", "y": "Quantidade de Nulos"},
        color_discrete_sequence=["#e74c3c"],
    )
    fig_null.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_null, use_container_width=True)
else:
    st.success("Nenhum valor ausente encontrado no dataset.")
