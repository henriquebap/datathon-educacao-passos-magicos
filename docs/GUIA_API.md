# Guia de Uso da API - Datathon Passos Mágicos

## URL de Acesso (Deploy em Produção)

A API está disponível publicamente no HuggingFace Spaces:

```
https://henriquebap-datathon-educacao-passos-magicos.hf.space
```

Não precisa instalar nada. Basta abrir no navegador ou usar qualquer ferramenta HTTP.

---

## Acesso Rápido

| O que você quer fazer | URL |
|---|---|
| Ver a documentação interativa (Swagger) | [/docs](https://henriquebap-datathon-educacao-passos-magicos.hf.space/docs) |
| Verificar se a API está no ar | [/health](https://henriquebap-datathon-educacao-passos-magicos.hf.space/health) |
| Fazer uma predição individual | POST [/predict](https://henriquebap-datathon-educacao-passos-magicos.hf.space/predict) |
| Fazer predições em lote | POST [/predict/batch](https://henriquebap-datathon-educacao-passos-magicos.hf.space/predict/batch) |
| Ver métricas do modelo | [/metrics](https://henriquebap-datathon-educacao-passos-magicos.hf.space/metrics) |
| Monitoramento de drift | [/monitoring/drift](https://henriquebap-datathon-educacao-passos-magicos.hf.space/monitoring/drift) |

---

## Forma mais fácil: Swagger UI

Abra no navegador:

```
https://henriquebap-datathon-educacao-passos-magicos.hf.space/docs
```

Lá você encontra todos os endpoints documentados. Para testar:

1. Clique no endpoint desejado (por exemplo, **POST /predict**)
2. Clique em **"Try it out"**
3. Edite o JSON de exemplo com os dados do aluno
4. Clique em **"Execute"**
5. A resposta aparece logo abaixo

Essa é a forma mais direta de testar a API sem precisar de nenhuma ferramenta externa.

---

## Endpoint `/predict` — Predição Individual

### O que faz

Recebe os indicadores educacionais de um estudante e retorna a classificação de risco de defasagem escolar.

### Request

**Método:** POST
**URL:** `https://henriquebap-datathon-educacao-passos-magicos.hf.space/predict`
**Content-Type:** `application/json`

### Campos de entrada (principais)

Os campos mais relevantes para a predição do ano 2022:

| Campo | Tipo | Faixa | Descrição |
|---|---|---|---|
| `INDE_2022` | float | 0–10 | Índice de Desenvolvimento Educacional |
| `IAA_2022` | float | 0–10 | Indicador de Autoavaliação |
| `IEG_2022` | float | 0–10 | Indicador de Engajamento |
| `IPS_2022` | float | 0–10 | Indicador Psicossocial |
| `IDA_2022` | float | 0–10 | Indicador de Adequação de Nível |
| `IPP_2022` | float | 0–10 | Indicador Psicopedagógico |
| `IPV_2022` | float | 0–10 | Indicador do Ponto de Virada |
| `IDADE_ALUNO_2022` | int | 5–25 | Idade do aluno |
| `PEDRA_2022` | string | — | Classificação PEDRA (Quartzo/Ágata/Ametista/Topázio) |
| `ANOS_PM_2022` | int | 0–15 | Anos no programa Passos Mágicos |

Todos os campos são opcionais — o modelo lida com dados parciais. Mas quanto mais indicadores você enviar, melhor a predição.

Também aceita indicadores de 2020 e 2021 para análise temporal (ex: `INDE_2020`, `IAA_2021`).

### Campos de saída

| Campo | Descrição |
|---|---|
| `prediction` | `0` = sem risco, `1` = em risco de defasagem |
| `risk_level` | `"LOW"` ou `"HIGH"` |
| `probability` | Probabilidades por classe (no_risk, at_risk) |
| `model_type` | Tipo do modelo (XGBClassifier) |
| `timestamp` | Data/hora da predição |

---

## Exemplos Práticos

### Exemplo 1 — curl (terminal)

Aluno com indicadores altos (provavelmente sem risco):

```bash
curl -X POST https://henriquebap-datathon-educacao-passos-magicos.hf.space/predict \
  -H "Content-Type: application/json" \
  -d '{
    "INDE_2022": 8.5,
    "IAA_2022": 8.0,
    "IEG_2022": 7.5,
    "IPS_2022": 7.0,
    "IDA_2022": 8.0,
    "IPP_2022": 6.5,
    "IPV_2022": 7.5,
    "IDADE_ALUNO_2022": 12,
    "PEDRA_2022": "Topázio",
    "ANOS_PM_2022": 4
  }'
```

Resposta esperada (exemplo):

```json
{
  "prediction": 0,
  "risk_level": "LOW",
  "probability": {
    "no_risk": 0.92,
    "at_risk": 0.08
  },
  "model_type": "XGBClassifier",
  "timestamp": "2026-02-26T15:30:00.000000+00:00"
}
```

### Exemplo 2 — curl (aluno em risco)

Aluno com indicadores baixos:

```bash
curl -X POST https://henriquebap-datathon-educacao-passos-magicos.hf.space/predict \
  -H "Content-Type: application/json" \
  -d '{
    "INDE_2022": 3.2,
    "IAA_2022": 2.5,
    "IEG_2022": 3.0,
    "IPS_2022": 4.0,
    "IDA_2022": 2.8,
    "IPP_2022": 3.5,
    "IPV_2022": 2.0,
    "IDADE_ALUNO_2022": 16,
    "PEDRA_2022": "Quartzo",
    "ANOS_PM_2022": 1
  }'
```

Resposta esperada:

```json
{
  "prediction": 1,
  "risk_level": "HIGH",
  "probability": {
    "no_risk": 0.03,
    "at_risk": 0.97
  },
  "model_type": "XGBClassifier",
  "timestamp": "2026-02-26T15:31:00.000000+00:00"
}
```

### Exemplo 3 — Python (requests)

```python
import requests

url = "https://henriquebap-datathon-educacao-passos-magicos.hf.space/predict"

aluno = {
    "INDE_2022": 6.0,
    "IAA_2022": 5.5,
    "IEG_2022": 6.2,
    "IPS_2022": 5.0,
    "IDA_2022": 5.8,
    "IPP_2022": 4.5,
    "IPV_2022": 5.0,
    "IDADE_ALUNO_2022": 14,
    "PEDRA_2022": "Ágata",
    "ANOS_PM_2022": 2
}

resp = requests.post(url, json=aluno)
resultado = resp.json()

print(f"Risco: {resultado['risk_level']}")
print(f"Probabilidade: {resultado['probability']['at_risk']:.1%}")
```

### Exemplo 4 — Predição em lote (POST /predict/batch)

```bash
curl -X POST https://henriquebap-datathon-educacao-passos-magicos.hf.space/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "students": [
      {
        "INDE_2022": 8.0, "IAA_2022": 7.5, "IEG_2022": 7.0,
        "IPS_2022": 6.5, "IDA_2022": 7.2, "IPP_2022": 6.0,
        "IPV_2022": 6.5, "IDADE_ALUNO_2022": 11, "PEDRA_2022": "Ametista"
      },
      {
        "INDE_2022": 3.0, "IAA_2022": 2.0, "IEG_2022": 2.5,
        "IPS_2022": 3.0, "IDA_2022": 2.2, "IPP_2022": 2.0,
        "IPV_2022": 1.5, "IDADE_ALUNO_2022": 17, "PEDRA_2022": "Quartzo"
      }
    ]
  }'
```

---

## Outros Endpoints

### GET /health

Verifica se a API está funcionando e se o modelo foi carregado.

```bash
curl https://henriquebap-datathon-educacao-passos-magicos.hf.space/health
```

### GET /metrics

Retorna as métricas de performance do modelo (F1-Score, acurácia, etc.) e estatísticas das predições feitas.

```bash
curl https://henriquebap-datathon-educacao-passos-magicos.hf.space/metrics
```

### GET /monitoring/drift

Verifica se houve drift nos dados de entrada em relação aos dados de treinamento — útil para monitoramento contínuo.

```bash
curl https://henriquebap-datathon-educacao-passos-magicos.hf.space/monitoring/drift
```

---

## Rodando Localmente (alternativa)

Caso queira rodar a API na sua máquina:

```bash
# Clonar o repositório
git clone https://github.com/henriquebap/datathon-educacao-passos-magicos.git
cd datathon-educacao-passos-magicos

# Instalar dependências
pip install -r requirements.txt

# Subir a API
uvicorn api.main:app --host 0.0.0.0 --port 8000

# Acessar: http://localhost:8000/docs
```

Ou via Docker:

```bash
docker compose up --build
# Acessar: http://localhost:8000/docs
```

---

## Resumo

- **URL base:** `https://henriquebap-datathon-educacao-passos-magicos.hf.space`
- **Documentação interativa:** `/docs` (Swagger UI — teste direto no navegador)
- **Predição:** `POST /predict` com JSON dos indicadores do aluno
- **Modelo:** XGBClassifier treinado com dados PEDE 2020-2022
- **Artefatos do modelo:** [HuggingFace Model Hub](https://huggingface.co/henriquebap/datathon-educacao-passos-magicos-model)
- **Código fonte:** [GitHub](https://github.com/henriquebap/datathon-educacao-passos-magicos)
