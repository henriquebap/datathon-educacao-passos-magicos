"""Datathon Educacao - Passos Magicos | Streamlit App."""

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(
    page_title="Datathon Educacao - Passos Magicos",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

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

st.markdown(
    """
    ### Navegação

    Use o menu lateral para acessar as páginas:

    | Página | Descrição |
    |--------|-----------|
    | **Análise Exploratória** | Visualizações e estatísticas dos dados |
    | **Modelo Preditivo** | Faça predições para novos estudantes |
    | **Performance do Modelo** | Métricas, confusion matrix, feature importance |

    ---

    *Projeto acadêmico — PosTech FIAP Datathon 2024/2025*
    """
)
