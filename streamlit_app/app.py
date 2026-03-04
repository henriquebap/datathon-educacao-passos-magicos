"""Datathon Educacao - Passos Magicos | Streamlit App (consome API FastAPI)."""

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from streamlit_app.api_client import API_URL, health_check

st.set_page_config(
    page_title="Datathon Educacao - Passos Magicos",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Sidebar: status da API ---
with st.sidebar:
    st.markdown("### Status da API")
    try:
        status = health_check()
        if status.get("model_loaded"):
            st.success(f"Online — {status.get('model_type', 'modelo carregado')}")
        else:
            st.warning("Online — modelo não carregado")
    except Exception:
        st.error("API offline")
    st.caption(f"`{API_URL}`")
    st.divider()

# --- Conteúdo principal ---
st.title("🎓 Datathon Educação — Passos Mágicos")
st.markdown("**Modelo Preditivo de Desenvolvimento Educacional**")
st.divider()

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown(
        """
        ### Sobre o Projeto

        Este projeto foi desenvolvido como parte do **Datathon PosTech FIAP — Fase 5**,
        com o objetivo de criar uma proposta preditiva que demonstre o impacto da ONG
        **Passos Mágicos** sobre a comunidade que atende.

        A Associação Passos Mágicos utiliza a educação como ferramenta para transformar
        as condições de vida de crianças e jovens em vulnerabilidade social no município
        de Embu-Guaçu/SP.

        ---

        ### Proposta Preditiva

        Utilizamos dados da **Pesquisa Extensiva de Desenvolvimento Educacional (PEDE)**
        dos períodos de 2020, 2021 e 2022 para construir um modelo capaz de prever o
        risco de **defasagem escolar** dos estudantes.

        O modelo permite à equipe pedagógica:
        - Identificar precocemente alunos em risco
        - Priorizar recursos de acompanhamento
        - Monitorar a evolução dos indicadores
        """
    )

with col2:
    st.markdown("### Indicadores Educacionais (PEDE)")
    indicators = {
        "INDE": "Índice de Desenvolvimento Educacional",
        "IAA": "Indicador de Autoavaliação",
        "IEG": "Indicador de Engajamento",
        "IPS": "Indicador Psicossocial",
        "IDA": "Indicador de Adequação de Nível",
        "IPP": "Indicador Psicopedagógico",
        "IPV": "Indicador do Ponto de Virada",
    }
    for code, desc in indicators.items():
        st.markdown(f"- **{code}**: {desc}")

st.divider()

col_a, col_b, col_c = st.columns(3)

with col_a:
    st.metric("Modelos Avaliados", "5")
    st.caption("LogReg, RF, GB, XGB, SVM")

with col_b:
    st.metric("Métrica Principal", "F1-Score")
    st.caption("Balanceamento Precision/Recall")

with col_c:
    st.metric("Períodos Analisados", "2020–2022")
    st.caption("Dados PEDE — Passos Mágicos")

st.divider()

st.markdown("### Arquitetura")

SPACE_URL = "https://henriquebap-datathon-educacao-passos-magicos.hf.space"

st.markdown(
    "Este dashboard **consome a API REST (FastAPI)** que roda no mesmo container. "
    "Um reverse proxy (nginx) roteia as requisições: os endpoints da API são "
    "acessíveis diretamente pela mesma URL do Space."
)

st.markdown(
    f"**Teste a API:** "
    f"[Swagger UI (interativo)]({SPACE_URL}/docs) · "
    f"[ReDoc (documentação)]({SPACE_URL}/redoc)"
)

st.divider()

st.markdown("### Endpoints da API REST")

st.markdown(
    f"""
| Método | Endpoint | Descrição |
|--------|----------|-----------|
| `GET` | [`/health`]({SPACE_URL}/health) | Status da API e do modelo carregado |
| `GET` | [`/api`]({SPACE_URL}/api) | Informações gerais da API (versão, ambiente, lista de endpoints) |
| `GET` | [`/metrics`]({SPACE_URL}/metrics) | Métricas de avaliação (accuracy, F1, AUC-ROC, confusion matrix, feature importance) |
| `GET` | [`/monitoring/drift`]({SPACE_URL}/monitoring/drift) | Monitoramento de drift nas predições em produção |
| `POST` | [`/predict`]({SPACE_URL}/docs#/Predi%C3%A7%C3%B5es/predict_predict_post) | Predição de risco para **um** estudante (JSON body) |
| `POST` | [`/predict/batch`]({SPACE_URL}/docs#/Predi%C3%A7%C3%B5es/predict_batch_predict_batch_post) | Predição em lote para **múltiplos** estudantes |
| `GET` | [`/docs`]({SPACE_URL}/docs) | **Swagger UI** — interface interativa para testar a API |
| `GET` | [`/redoc`]({SPACE_URL}/redoc) | **ReDoc** — documentação detalhada dos schemas |
"""
)

st.divider()
st.caption("*Projeto acadêmico — PosTech FIAP Datathon 2025/2026*")
