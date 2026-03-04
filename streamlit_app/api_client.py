"""Cliente HTTP para consumir a API FastAPI de predição."""

import os
import time
from typing import Any

import requests
import streamlit as st

_REMOTE_URL = "https://henriquebap-datathon-educacao-passos-magicos.hf.space"
_LOCAL_URL = "http://localhost:8000"

_default = _LOCAL_URL if os.getenv("API_ENV") == "production" else _REMOTE_URL
API_URL = os.getenv("API_URL", _default).rstrip("/")

_TIMEOUT = 30


def _handle_response(resp: requests.Response) -> dict:
    """Valida a resposta e retorna o JSON ou levanta erro legível."""
    if resp.status_code >= 400:
        detail = resp.text
        if resp.headers.get("content-type", "").startswith("application/json"):
            detail = resp.json().get("detail", resp.text)
        raise requests.HTTPError(f"[{resp.status_code}] {detail}", response=resp)
    return resp.json()


@st.cache_data(ttl=10)
def health_check(_retries: int = 3) -> dict[str, Any]:
    """Verifica saúde da API com retries para o startup do container."""
    last_err = None
    for attempt in range(_retries):
        try:
            resp = requests.get(f"{API_URL}/health", timeout=_TIMEOUT)
            return _handle_response(resp)
        except (requests.ConnectionError, requests.Timeout) as e:
            last_err = e
            if attempt < _retries - 1:
                time.sleep(2)
    raise last_err  # type: ignore[misc]


def predict(student_data: dict) -> dict[str, Any]:
    resp = requests.post(f"{API_URL}/predict", json=student_data, timeout=_TIMEOUT)
    return _handle_response(resp)


def predict_batch(students: list[dict]) -> dict[str, Any]:
    resp = requests.post(
        f"{API_URL}/predict/batch",
        json={"students": students},
        timeout=_TIMEOUT * 3,
    )
    return _handle_response(resp)


@st.cache_data(ttl=60)
def get_metrics() -> dict[str, Any]:
    resp = requests.get(f"{API_URL}/metrics", timeout=_TIMEOUT)
    return _handle_response(resp)


@st.cache_data(ttl=60)
def get_drift() -> dict[str, Any]:
    resp = requests.get(f"{API_URL}/monitoring/drift", timeout=_TIMEOUT)
    return _handle_response(resp)
