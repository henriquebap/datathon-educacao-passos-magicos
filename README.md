# Datathon Educação - Passos Mágicos

Modelo preditivo de desenvolvimento educacional para estudantes da Associação Passos Mágicos, desenvolvido como parte do **Datathon PosTech FIAP - Fase 5**.

---

## 1. Visão Geral do Projeto

### Objetivo

Criar uma proposta preditiva que demonstre o impacto da ONG **Passos Mágicos** sobre a comunidade atendida, utilizando dados da Pesquisa Extensiva de Desenvolvimento Educacional (PEDE) nos períodos de 2020, 2021 e 2022.

### Problema de Negócio

A Associação Passos Mágicos atua na transformação da vida de crianças e jovens de baixa renda por meio da educação. Com um modelo preditivo, a associação pode:

- **Identificar precocemente** estudantes em risco de defasagem escolar
- **Priorizar recursos** de acompanhamento psicopedagógico
- **Monitorar a evolução** dos indicadores educacionais ao longo do tempo

### Solução Proposta

Pipeline completa de Machine Learning com deploy via Docker (FastAPI) no **HuggingFace Spaces**, incluindo análise exploratória, modelo preditivo e monitoramento de drift.

### Stack Tecnológica

| Componente | Tecnologia |
|---|---|
| Linguagem | Python 3.11 |
| Frameworks ML | scikit-learn, pandas, numpy, XGBoost |
| API / Deploy | FastAPI + Uvicorn (Docker no HuggingFace Spaces) |
| Dashboard | Streamlit |
| Serialização | joblib |
| Testes | pytest |
| Containerização | Docker (multi-stage build) |
| Monitoramento | Evidently AI + logging estruturado |
| Model Registry | HuggingFace Hub |

---

## 2. Estrutura do Projeto

```
datathon/
├── data/
│   ├── raw/                        # Dataset PEDE (CSV/Excel)
│   └── processed/                  # Dados processados
├── docs/
│   └── GUIA_API.md                 # Guia completo de uso da API
├── models/                         # Modelos serializados (.joblib)
├── notebooks/
│   ├── 01_EDA.ipynb                # Análise Exploratória dos Dados
│   ├── 02_Pipeline_ML.ipynb        # Pipeline de Machine Learning
│   └── 03_Avaliacao_Modelo.ipynb   # Avaliação e métricas do modelo
├── src/
│   ├── __init__.py
│   ├── utils.py                    # Configuração, utilitários
│   ├── preprocessing.py            # Limpeza, encoding, normalização, split
│   ├── feature_engineering.py      # Features temporais, compostas, interação
│   ├── train.py                    # Treinamento, tuning, serialização
│   ├── evaluate.py                 # Métricas, confusion matrix, feature importance
│   ├── predict.py                  # Classe Predictor para inferência
│   └── monitoring.py               # Drift detection, logging de predições
├── streamlit_app/
│   ├── app.py                      # App principal Streamlit
│   └── pages/
│       ├── 1_Analise_Exploratoria.py
│       ├── 2_Modelo_Preditivo.py
│       └── 3_Performance.py
├── api/                            # API REST (complementar)
│   ├── __init__.py
│   ├── main.py
│   ├── schemas.py
│   └── routes.py
├── tests/
│   ├── conftest.py
│   ├── test_utils.py
│   ├── test_preprocessing.py
│   ├── test_feature_engineering.py
│   ├── test_train.py
│   ├── test_evaluate.py
│   ├── test_predict.py
│   ├── test_monitoring.py
│   └── test_api.py
├── monitoring/
│   └── dashboard.py
├── scripts/
│   └── setup_supabase.sql
├── run_training.py
├── Dockerfile
├── Dockerfile.hf
├── docker-compose.yml
├── pyproject.toml
├── requirements.txt
├── .env.example
├── .dockerignore
└── .gitignore
```

---

## 3. API em Produção (Deploy)

> **A API está disponível publicamente e pode ser acessada diretamente pelo navegador — sem instalação.**

### URL Base

```
https://henriquebap-datathon-educacao-passos-magicos.hf.space
```

### Endpoints Disponíveis

| Endpoint | Método | Descrição |
|---|---|---|
| [`/docs`](https://henriquebap-datathon-educacao-passos-magicos.hf.space/docs) | GET | **Documentação interativa (Swagger UI)** — teste os endpoints direto no navegador |
| [`/health`](https://henriquebap-datathon-educacao-passos-magicos.hf.space/health) | GET | Verifica se a API e o modelo estão carregados |
| `/predict` | POST | Predição individual de risco de defasagem |
| `/predict/batch` | POST | Predição para múltiplos alunos de uma vez |
| [`/metrics`](https://henriquebap-datathon-educacao-passos-magicos.hf.space/metrics) | GET | Métricas de performance do modelo |
| [`/monitoring/drift`](https://henriquebap-datathon-educacao-passos-magicos.hf.space/monitoring/drift) | GET | Monitoramento de drift nos dados |

### Exemplo de Predição (curl)

```bash
curl -X POST https://henriquebap-datathon-educacao-passos-magicos.hf.space/predict \
  -H "Content-Type: application/json" \
  -d '{
    "INDE_2022": 7.5,
    "IAA_2022": 6.8,
    "IEG_2022": 7.0,
    "IPS_2022": 6.5,
    "IDA_2022": 7.2,
    "IPP_2022": 5.5,
    "IPV_2022": 6.0,
    "IDADE_ALUNO_2022": 13,
    "PEDRA_2022": "Ametista",
    "ANOS_PM_2022": 3
  }'
```

Resposta:

```json
{
  "prediction": 1,
  "risk_level": "HIGH",
  "probability": { "no_risk": 0.003, "at_risk": 0.997 },
  "model_type": "XGBClassifier",
  "timestamp": "2026-02-26T15:30:00+00:00"
}
```

Para mais exemplos (Python, batch, Swagger), consulte o **[Guia Completo da API](docs/GUIA_API.md)**.

### Links do Projeto

| Recurso | URL |
|---|---|
| API (produção) | https://henriquebap-datathon-educacao-passos-magicos.hf.space |
| Swagger UI | https://henriquebap-datathon-educacao-passos-magicos.hf.space/docs |
| Modelo (artefatos) | https://huggingface.co/henriquebap/datathon-educacao-passos-magicos-model |
| Código fonte | https://github.com/henriquebap/datathon-educacao-passos-magicos |

---

## 4. Como Executar Localmente

### Pré-requisitos

- Python 3.11+
- Dataset PEDE em `data/raw/` (fornecido pelo datathon)

### Instalação

```bash
git clone https://github.com/henriquebap/datathon-educacao-passos-magicos.git
cd datathon-educacao-passos-magicos

python -m venv venv
source venv/bin/activate  # Linux/Mac

pip install -r requirements.txt

cp .env.example .env
```

### Treinamento do Modelo

```bash
# Treinar com dados reais (coloque o CSV/XLSX em data/raw/)
python run_training.py

# Treinar com dados sintéticos (para teste)
python run_training.py --synthetic

# Treinar modelo específico
python run_training.py --model RandomForest --scoring f1
```

### Executar o Streamlit (Deploy Principal)

```bash
streamlit run streamlit_app/app.py
```

Acesse em `http://localhost:8501`

### Executar a API (Complementar)

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Docker

```bash
docker compose up --build
```

### Testes

```bash
pytest tests/ -v
```

---

## 5. Pipeline de Machine Learning

### 5.1 Pré-processamento (`src/preprocessing.py`)

1. **Remoção de identificadores**: colunas como NOME, ID, MATRÍCULA
2. **Tratamento de valores ausentes**: mediana para numéricos, moda para categóricos
3. **Encoding**: PEDRA com mapeamento ordinal, LabelEncoder para demais categóricos
4. **Normalização**: StandardScaler para features numéricas
5. **Split**: 80/20 estratificado

### 5.2 Engenharia de Features (`src/feature_engineering.py`)

1. **Features temporais**: diferenças ano-a-ano, tendência, média e desvio padrão
2. **Indicadores compostos**: Academic Composite, Engagement Composite, Risk Score
3. **Features de interação**: INDE×IEG, IPS×IAA, Bolsista×INDE
4. **Gap idade-fase**: diferença entre idade real e fase esperada

### 5.3 Treinamento (`src/train.py`)

**Modelos avaliados:** Logistic Regression, Random Forest, Gradient Boosting, XGBoost, SVM

**Processo:** Cross-validation estratificada (5 folds), seleção automática por F1-Score, GridSearchCV para tuning.

### 5.4 Justificativa da Métrica (F1-Score)

No contexto de defasagem escolar, precisamos balancear:
- **Recall**: não deixar de identificar alunos em risco (minimizar falsos negativos)
- **Precision**: evitar alarmes excessivos que sobrecarreguem a equipe pedagógica

O F1-Score equilibra ambos. Utilizamos `class_weight="balanced"` para lidar com desbalanceamento entre as classes.

### 5.5 Tratamento de Data Leakage

As colunas **IAN** (Indicador de Adequação de Nível) e **NIVEL_IDEAL** são removidas do treinamento e da inferência, pois são proxies diretos da variável alvo (DEFASAGEM) — incluí-las seria equivalente a dar a resposta ao modelo.

---

## 6. Notebooks

| Notebook | Conteúdo |
|---|---|
| `01_EDA.ipynb` | Análise exploratória: distribuições, correlações, perfil dos estudantes |
| `02_Pipeline_ML.ipynb` | Pipeline completo: preprocessing, feature engineering, treinamento |
| `03_Avaliacao_Modelo.ipynb` | Métricas, confusion matrix, feature importance, comparação de modelos |

---

## 7. Sobre a Associação Passos Mágicos

A Associação Passos Mágicos tem 32 anos de atuação, transformando a vida de crianças e jovens de baixa renda por meio da educação. Idealizada por Michelle Flues e Dimetri Ivanoff, atua em Embu-Guaçu oferecendo educação de qualidade, auxílio psicológico/psicopedagógico, ampliação de visão de mundo e protagonismo.

---

## Licença

Projeto acadêmico - PosTech FIAP Datathon 2025/2026.
