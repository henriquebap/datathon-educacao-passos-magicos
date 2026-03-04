# HuggingFace Spaces deploy — nginx + FastAPI + Streamlit via supervisord
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

COPY requirements_hf.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements_hf.txt

FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    API_ENV=production \
    HOME=/app

RUN apt-get update && \
    apt-get install -y --no-install-recommends nginx && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

RUN mkdir -p logs data/raw data/processed models monitoring/reports /tmp/.streamlit && \
    chmod -R 777 logs data models monitoring /tmp/.streamlit

COPY src/ ./src/
COPY api/ ./api/
COPY models/ ./models/
COPY streamlit_app/ ./streamlit_app/
COPY nginx.conf ./nginx.conf
COPY supervisord.conf ./supervisord.conf

RUN chmod -R 777 models logs

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["supervisord", "-c", "/app/supervisord.conf"]
