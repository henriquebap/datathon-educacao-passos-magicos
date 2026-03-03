"""Pagina de Performance do Modelo."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import MODELS_DIR

st.set_page_config(page_title="Performance - Passos Mágicos", page_icon="📈", layout="wide")
st.title("📈 Performance do Modelo")
st.divider()

REPORT_PATH = MODELS_DIR / "evaluation_report.json"

if not REPORT_PATH.exists():
    st.warning(
        "Relatório de avaliação não encontrado. Execute o treinamento primeiro:\n\n"
        "```bash\npython run_training.py --synthetic\n```"
    )
    st.stop()

with open(REPORT_PATH) as f:
    report = json.load(f)

metrics = report.get("metrics", {})
cm_data = report.get("confusion_matrix", {})
feat_imp = report.get("feature_importance", [])
model_type = report.get("model_type", "N/A")
n_samples = report.get("n_test_samples", 0)

# --- Resumo ---
st.header("1. Resumo do Modelo")

col1, col2 = st.columns([1, 2])

with col1:
    st.metric("Modelo Selecionado", model_type)
    st.metric("Amostras de Teste", n_samples)

with col2:
    metric_cols = st.columns(5)
    metric_labels = {
        "accuracy": "Acurácia",
        "precision": "Precisão",
        "recall": "Recall",
        "f1_score": "F1-Score",
        "auc_roc": "AUC-ROC",
    }
    for i, (key, label) in enumerate(metric_labels.items()):
        val = metrics.get(key)
        if val is not None:
            metric_cols[i].metric(label, f"{val:.4f}")

st.divider()

# --- Metricas em barras ---
st.header("2. Comparação de Métricas")

metric_names = [v for k, v in metric_labels.items() if metrics.get(k) is not None]
metric_values = [metrics[k] for k in metric_labels if metrics.get(k) is not None]

fig_metrics = go.Figure(go.Bar(
    x=metric_names,
    y=metric_values,
    marker_color=["#3498db", "#2ecc71", "#e74c3c", "#f39c12", "#9b59b6"],
    text=[f"{v:.4f}" for v in metric_values],
    textposition="auto",
))
fig_metrics.update_layout(
    title=f"Métricas de Avaliação — {model_type}",
    yaxis_title="Valor",
    yaxis=dict(range=[0, 1]),
    height=400,
)
st.plotly_chart(fig_metrics, use_container_width=True)

st.divider()

# --- Confusion Matrix ---
st.header("3. Matriz de Confusão")

cm_matrix = cm_data.get("matrix", [[0, 0], [0, 0]])

col_cm1, col_cm2 = st.columns([1, 1])

with col_cm1:
    labels = ["Sem Risco (0)", "Em Risco (1)"]
    fig_cm = px.imshow(
        cm_matrix,
        x=labels, y=labels,
        color_continuous_scale="Blues",
        title="Matriz de Confusão",
        text_auto=True,
    )
    fig_cm.update_layout(
        xaxis_title="Predição",
        yaxis_title="Real",
        width=500, height=450,
    )
    st.plotly_chart(fig_cm, use_container_width=True)

with col_cm2:
    st.markdown("### Detalhamento")
    tn = cm_data.get("true_negatives", 0)
    fp = cm_data.get("false_positives", 0)
    fn = cm_data.get("false_negatives", 0)
    tp = cm_data.get("true_positives", 0)

    st.markdown(f"""
    | Métrica | Valor | Significado |
    |---------|-------|-------------|
    | **Verdadeiros Negativos** | {tn} | Corretamente sem risco |
    | **Verdadeiros Positivos** | {tp} | Corretamente em risco |
    | **Falsos Positivos** | {fp} | Alarme falso |
    | **Falsos Negativos** | {fn} | Risco não detectado |
    """)

    st.markdown(f"""
    ---
    **Interpretação**: O modelo identificou corretamente **{tp}** alunos em risco
    e **{tn}** alunos sem risco. Houve **{fn}** alunos em risco que não foram
    detectados (falsos negativos) — minimizar este número é prioritário.
    """)

st.divider()

# --- Classification Report ---
st.header("4. Relatório de Classificação")

class_report = report.get("classification_report", "")
if class_report:
    st.code(class_report, language=None)

st.markdown("""
**Justificativa da Métrica F1-Score**:

No contexto de defasagem escolar, o F1-Score é a métrica principal porque equilibra:
- **Recall (Sensibilidade)**: Capacidade de detectar alunos em risco. Falsos negativos
  significam alunos em risco que não receberão suporte.
- **Precision**: Evitar alarmes falsos que sobrecarreguem a equipe pedagógica.
""")

st.divider()

# --- Feature Importance ---
st.header("5. Importância das Features")

if feat_imp:
    imp_df = pd.DataFrame(feat_imp)

    top_n = st.slider("Top N features:", 5, min(30, len(imp_df)), 15)
    top_features = imp_df.head(top_n)

    fig_imp = px.bar(
        top_features,
        x="importance",
        y="feature",
        orientation="h",
        title=f"Top {top_n} Features Mais Importantes",
        color="importance",
        color_continuous_scale="Viridis",
    )
    fig_imp.update_layout(
        yaxis=dict(autorange="reversed"),
        height=max(400, top_n * 30),
        xaxis_title="Importância",
        yaxis_title="",
    )
    st.plotly_chart(fig_imp, use_container_width=True)

    with st.expander("Tabela completa de importância"):
        st.dataframe(imp_df, use_container_width=True)
else:
    st.info("Feature importance não disponível para este modelo.")

st.divider()

# --- Historico de Metricas ---
st.header("6. Histórico de Métricas")

metrics_log = MODELS_DIR / "model_metrics.jsonl"
if metrics_log.exists():
    entries = []
    with open(metrics_log) as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))

    if entries:
        history_df = pd.DataFrame(entries)
        st.dataframe(history_df, use_container_width=True)
    else:
        st.info("Sem registros no histórico.")
else:
    st.info("Arquivo de histórico não encontrado.")
