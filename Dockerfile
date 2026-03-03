# === Stage 1: build (instala dependencias com gcc) ===
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt

# === Stage 2: runtime (imagem enxuta, sem gcc) ===
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    API_ENV=production

COPY --from=builder /install /usr/local

RUN groupadd -g 1001 appgroup && \
    useradd -u 1001 -g appgroup -s /bin/false appuser && \
    mkdir -p logs data/raw data/processed monitoring/reports && \
    chown -R appuser:appgroup /app

COPY --chown=appuser:appgroup src/ ./src/
COPY --chown=appuser:appgroup api/ ./api/
COPY --chown=appuser:appgroup models/ ./models/
COPY --chown=appuser:appgroup monitoring/ ./monitoring/

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
