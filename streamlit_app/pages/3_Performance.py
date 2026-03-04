"""Pagina de Performance - consome /metrics e /monitoring/drift da API."""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from streamlit_app.api_client import API_URL, health_check, get_metrics, get_drift

st.set_page_config(page_title="Performance - Passos Mágicos", page_icon="📈", layout="wide")
st.title("📈 Performance do Modelo")
st.divider()

# --- Health check ---
try:
    status = health_check()
    if not status.get("model_loaded"):
        st.warning("API online, mas modelo não carregado.")
        st.stop()
except Exception as e:
    st.error(f"API indisponível em `{API_URL}` — {e}")
    st.stop()

# --- Métricas do modelo ---
st.header("1. Métricas do Modelo")

try:
    metrics_data = get_metrics()
except Exception as e:
    st.error(f"Erro ao buscar métricas: {e}")
    st.stop()

model_type = metrics_data.get("model_type", "N/A")
metrics = metrics_data.get("metrics", {})
prediction_stats = metrics_data.get("prediction_stats", {})

n_test = metrics_data.get("n_test_samples") or metrics.pop("n_test_samples", None)

col1, col2 = st.columns([1, 2])

with col1:
    st.metric("Modelo em Produção", model_type)
    if n_test:
        st.metric("Amostras de Teste", n_test)

with col2:
    metric_labels = {
        "accuracy": "Acurácia",
        "precision": "Precisão",
        "recall": "Recall",
        "f1_score": "F1-Score",
        "auc_roc": "AUC-ROC",
    }
    available = {k: v for k, v in metric_labels.items() if metrics.get(k) is not None}

    if available:
        metric_cols = st.columns(len(available))
        for i, (key, label) in enumerate(available.items()):
            metric_cols[i].metric(label, f"{metrics[key]:.4f}")
    else:
        st.info("Métricas de avaliação não disponíveis na API.")

st.divider()

# --- Gráfico de métricas ---
if available:
    st.header("2. Comparação de Métricas")

    metric_names = list(available.values())
    metric_values = [metrics[k] for k in available]
    colors = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12", "#9b59b6"]

    fig_metrics = go.Figure(go.Bar(
        x=metric_names,
        y=metric_values,
        marker_color=colors[:len(metric_names)],
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

# --- Confusion Matrix ---
cm_data = metrics_data.get("confusion_matrix")
if cm_data and "matrix" in cm_data:
    st.subheader("Matriz de Confusão")
    col_cm1, col_cm2 = st.columns([1, 1])

    with col_cm1:
        matrix = cm_data["matrix"]
        fig_cm = go.Figure(data=go.Heatmap(
            z=matrix,
            x=["Pred: Sem Risco", "Pred: Em Risco"],
            y=["Real: Sem Risco", "Real: Em Risco"],
            text=[[str(v) for v in row] for row in matrix],
            texttemplate="%{text}",
            colorscale="Blues",
            showscale=False,
        ))
        fig_cm.update_layout(title=f"Confusion Matrix ({n_test or '?'} amostras)", height=350)
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_cm2:
        st.metric("Verdadeiros Positivos (TP)", cm_data.get("true_positives", "—"))
        st.metric("Verdadeiros Negativos (TN)", cm_data.get("true_negatives", "—"))
        st.metric("Falsos Positivos (FP)", cm_data.get("false_positives", "—"))
        st.metric("Falsos Negativos (FN)", cm_data.get("false_negatives", "—"))

# --- Feature Importance ---
feat_imp = metrics_data.get("feature_importance")
if feat_imp:
    st.subheader("Top 15 Features mais Importantes")
    top_feats = feat_imp[:15]
    fig_feat = go.Figure(go.Bar(
        x=[f["importance"] for f in reversed(top_feats)],
        y=[f["feature"] for f in reversed(top_feats)],
        orientation="h",
        marker_color="#3498db",
    ))
    fig_feat.update_layout(
        title=f"Feature Importance — {model_type}",
        xaxis_title="Importância",
        height=500,
        margin=dict(l=200),
    )
    st.plotly_chart(fig_feat, use_container_width=True)

st.divider()

# --- Estatísticas de predições ---
st.header("3. Estatísticas de Predições em Produção")

total = prediction_stats.get("total_predictions", 0)

if total > 0:
    col_s1, col_s2, col_s3 = st.columns(3)

    risk_dist = prediction_stats.get("risk_distribution", {})
    avg_prob = prediction_stats.get("avg_probability_at_risk")

    col_s1.metric("Total de Predições", total)
    col_s2.metric("Alto Risco", risk_dist.get("HIGH", 0))
    col_s3.metric("Baixo Risco", risk_dist.get("LOW", 0))

    if avg_prob is not None:
        st.metric("Probabilidade Média de Risco", f"{avg_prob:.1%}")

    if risk_dist:
        fig_dist = px.pie(
            values=list(risk_dist.values()),
            names=["Alto Risco" if k == "HIGH" else "Baixo Risco" for k in risk_dist],
            title="Distribuição das Predições em Produção",
            color_discrete_sequence=["#e74c3c", "#2ecc71"],
        )
        st.plotly_chart(fig_dist, use_container_width=True)
else:
    st.info("Nenhuma predição realizada ainda. Use a página **Modelo Preditivo** para gerar predições.")

st.divider()

# --- Drift ---
st.header("4. Monitoramento de Drift")

try:
    drift_data = get_drift()

    drift_detected = drift_data.get("drift_detected")
    details = drift_data.get("details", {})

    if drift_detected is None:
        st.info(details.get("message", "Dados insuficientes para análise de drift."))
    elif drift_detected:
        st.error("⚠️ **Drift detectado** nos dados de produção!")
    else:
        st.success("✅ Sem drift detectado — modelo estável.")

    with st.expander("Detalhes do monitoramento"):
        st.json(details)

except Exception as e:
    st.warning(f"Drift não disponível: {e}")

st.divider()

# --- Justificativa ---
st.header("5. Justificativa da Métrica")

st.markdown("""
**Por que F1-Score?**

No contexto de defasagem escolar, o F1-Score é a métrica principal porque equilibra:
- **Recall (Sensibilidade)**: Capacidade de detectar alunos em risco. Falsos negativos
  significam alunos em risco que não receberão suporte.
- **Precision**: Evitar alarmes falsos que sobrecarreguem a equipe pedagógica.

O modelo em produção (**XGBClassifier**) foi selecionado automaticamente por ter
obtido o melhor F1-Score na cross-validation estratificada (5 folds) dentre os
cinco modelos candidatos (Logistic Regression, Random Forest, Gradient Boosting,
XGBoost e SVM).
""")

st.divider()
st.caption(f"Dados obtidos via API: `{API_URL}`")
