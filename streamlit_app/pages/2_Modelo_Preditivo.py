"""Pagina de Predicao - Modelo Preditivo."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import MODELS_DIR, NUMERIC_INDICATORS, PEDRA_MAP

st.set_page_config(page_title="Predição - Passos Mágicos", page_icon="🔮", layout="wide")
st.title("🔮 Modelo Preditivo")
st.markdown("Insira os dados do estudante para prever o risco de defasagem escolar.")
st.divider()

MODEL_PATH = MODELS_DIR / "model.joblib"
PIPELINE_PATH = MODELS_DIR / "pipeline.joblib"


@st.cache_resource
def load_predictor():
    """Load model and pipeline artifacts."""
    if not MODEL_PATH.exists() or not PIPELINE_PATH.exists():
        return None
    try:
        from src.predict import Predictor
        return Predictor(str(MODEL_PATH), str(PIPELINE_PATH))
    except Exception as e:
        st.error(f"Erro ao carregar modelo: {e}")
        return None


predictor = load_predictor()

if predictor is None:
    st.warning(
        "Modelo não encontrado. Execute o treinamento primeiro:\n\n"
        "```bash\npython run_training.py --synthetic\n```"
    )
    st.stop()

st.success(f"Modelo carregado: **{type(predictor.model).__name__}**")

st.header("Dados do Estudante")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Indicadores Acadêmicos")
    inde = st.slider("INDE (Desenvolvimento)", 0.0, 10.0, 6.0, 0.1)
    iaa = st.slider("IAA (Autoavaliação)", 0.0, 10.0, 6.0, 0.1)
    ieg = st.slider("IEG (Engajamento)", 0.0, 10.0, 6.0, 0.1)
    ida = st.slider("IDA (Adequação)", 0.0, 10.0, 6.0, 0.1)

with col2:
    st.subheader("Indicadores Psicossociais")
    ips = st.slider("IPS (Psicossocial)", 0.0, 10.0, 6.0, 0.1)
    ipp = st.slider("IPP (Psicopedagógico)", 0.0, 10.0, 5.0, 0.1)
    ipv = st.slider("IPV (Ponto de Virada)", 0.0, 10.0, 5.0, 0.1)

with col3:
    st.subheader("Dados Gerais")
    idade = st.number_input("Idade do Aluno", 6, 25, 13)
    pedra = st.selectbox("Classificação PEDRA", list(PEDRA_MAP.keys()))
    fase = st.number_input("Fase no Programa", 0, 8, 4)
    ponto_virada = st.selectbox("Ponto de Virada", [0, 1], format_func=lambda x: "Sim" if x else "Não")
    bolsista = st.selectbox("Bolsista", [0, 1], format_func=lambda x: "Sim" if x else "Não")
    anos_pm = st.number_input("Anos no Programa", 1, 10, 3)

st.divider()

if st.button("🔍 Realizar Predição", type="primary", use_container_width=True):
    input_data = {
        "INDE_2022": inde,
        "IAA_2022": iaa,
        "IEG_2022": ieg,
        "IPS_2022": ips,
        "IDA_2022": ida,
        "IPP_2022": ipp,
        "IPV_2022": ipv,
        "IDADE_ALUNO_2022": idade,
        "PEDRA_2022": pedra,
        "FASE_2022": fase,
        "PONTO_VIRADA_2022": ponto_virada,
        "BOLSISTA_2022": bolsista,
        "ANOS_PM_2022": anos_pm,
    }

    with st.spinner("Processando predição..."):
        try:
            result = predictor.predict(input_data)
        except Exception as e:
            st.error(f"Erro na predição: {e}")
            st.stop()

    st.divider()
    st.header("Resultado da Predição")

    risk_level = result.get("risk_level", "UNKNOWN")
    probability = result.get("probability", {})

    if risk_level == "HIGH":
        st.error("⚠️ **ALTO RISCO** de defasagem escolar")
    else:
        st.success("✅ **BAIXO RISCO** de defasagem escolar")

    col_r1, col_r2, col_r3 = st.columns(3)

    with col_r1:
        st.metric("Predição", "Em Risco" if result["prediction"] == 1 else "Sem Risco")

    with col_r2:
        prob_risk = probability.get("at_risk", 0)
        st.metric("Probabilidade de Risco", f"{prob_risk:.1%}")

    with col_r3:
        st.metric("Modelo", result.get("model_type", "N/A"))

    if probability:
        import plotly.graph_objects as go

        fig = go.Figure(go.Bar(
            x=[probability.get("no_risk", 0), probability.get("at_risk", 0)],
            y=["Sem Risco", "Em Risco"],
            orientation="h",
            marker_color=["#2ecc71", "#e74c3c"],
            text=[f"{probability.get('no_risk', 0):.1%}", f"{probability.get('at_risk', 0):.1%}"],
            textposition="auto",
        ))
        fig.update_layout(
            title="Probabilidades",
            xaxis_title="Probabilidade",
            yaxis_title="",
            xaxis=dict(range=[0, 1]),
            height=250,
        )
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Dados de entrada utilizados"):
        st.json(input_data)

st.divider()

# --- Batch ---
st.header("Predição em Lote")
st.markdown("Faça upload de um CSV para predição em lote.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    try:
        batch_df = pd.read_csv(uploaded_file)
        st.write(f"Registros carregados: **{len(batch_df)}**")
        st.dataframe(batch_df.head(), use_container_width=True)

        if st.button("🚀 Predição em Lote", use_container_width=True):
            with st.spinner("Processando..."):
                results = predictor.predict_batch(batch_df.to_dict("records"))

            results_df = pd.DataFrame([
                {
                    "Predição": r["prediction"],
                    "Risco": r["risk_level"],
                    "P(Risco)": r["probability"]["at_risk"] if r.get("probability") else None,
                }
                for r in results
            ])

            combined = pd.concat([batch_df.reset_index(drop=True), results_df], axis=1)
            st.dataframe(combined, use_container_width=True)

            risk_counts = results_df["Risco"].value_counts()
            st.metric("Total Em Risco", int(risk_counts.get("HIGH", 0)))
            st.metric("Total Sem Risco", int(risk_counts.get("LOW", 0)))

            csv_output = combined.to_csv(index=False)
            st.download_button(
                "📥 Baixar Resultados (CSV)",
                csv_output,
                "predicoes_passos_magicos.csv",
                "text/csv",
            )
    except Exception as e:
        st.error(f"Erro ao processar arquivo: {e}")
