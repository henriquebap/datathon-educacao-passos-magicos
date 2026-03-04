"""Pagina de Predicao - consome a API FastAPI via HTTP."""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from streamlit_app.api_client import API_URL, health_check, predict, predict_batch

st.set_page_config(page_title="Predição - Passos Mágicos", page_icon="🔮", layout="wide")
st.title("🔮 Modelo Preditivo")
st.markdown("Insira os dados do estudante para prever o risco de defasagem escolar.")
st.divider()

# --- Health check ---
try:
    status = health_check()
    if status.get("model_loaded"):
        st.success(f"API conectada — modelo **{status.get('model_type', 'N/A')}** carregado")
    else:
        st.warning("API online, mas modelo não carregado.")
        st.stop()
except Exception as e:
    st.error(f"API indisponível em `{API_URL}` — {e}")
    st.caption("Configure `API_URL` como variável de ambiente para apontar para outra instância.")
    st.stop()

# --- Formulário ---
st.header("Dados do Estudante")

PEDRA_OPTIONS = ["Quartzo", "Ágata", "Ametista", "Topázio"]

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
    pedra = st.selectbox("Classificação PEDRA", PEDRA_OPTIONS)
    fase = st.number_input("Fase no Programa", 0, 8, 4)
    ponto_virada = st.selectbox("Ponto de Virada", [0, 1], format_func=lambda x: "Sim" if x else "Não")
    bolsista = st.selectbox("Bolsista", [0, 1], format_func=lambda x: "Sim" if x else "Não")
    anos_pm = st.number_input("Anos no Programa", 1, 10, 3)

st.divider()

if st.button("🔍 Realizar Predição", type="primary", use_container_width=True):
    payload = {
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

    with st.spinner("Enviando para a API..."):
        try:
            result = predict(payload)
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

    with st.expander("Payload enviado para a API"):
        st.json(payload)

    with st.expander("Resposta completa da API"):
        st.json(result)

st.divider()

# --- Batch via API ---
st.header("Predição em Lote (via API)")
st.markdown("Faça upload de um CSV — os dados serão enviados ao endpoint `/predict/batch`.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    try:
        batch_df = pd.read_csv(uploaded_file)
        st.write(f"Registros carregados: **{len(batch_df)}**")
        st.dataframe(batch_df.head(), use_container_width=True)

        if st.button("🚀 Predição em Lote", use_container_width=True):
            with st.spinner(f"Enviando {len(batch_df)} registros para a API..."):
                response = predict_batch(batch_df.to_dict("records"))

            predictions = response.get("predictions", [])
            results_df = pd.DataFrame([
                {
                    "Predição": r["prediction"],
                    "Risco": r["risk_level"],
                    "P(Risco)": r["probability"]["at_risk"] if r.get("probability") else None,
                }
                for r in predictions
            ])

            combined = pd.concat([batch_df.reset_index(drop=True), results_df], axis=1)
            st.dataframe(combined, use_container_width=True)

            risk_counts = results_df["Risco"].value_counts()
            col_b1, col_b2, col_b3 = st.columns(3)
            col_b1.metric("Total Registros", response.get("total", len(predictions)))
            col_b2.metric("Em Risco", int(risk_counts.get("HIGH", 0)))
            col_b3.metric("Sem Risco", int(risk_counts.get("LOW", 0)))

            csv_output = combined.to_csv(index=False)
            st.download_button(
                "📥 Baixar Resultados (CSV)",
                csv_output,
                "predicoes_passos_magicos.csv",
                "text/csv",
            )
    except Exception as e:
        st.error(f"Erro ao processar arquivo: {e}")
