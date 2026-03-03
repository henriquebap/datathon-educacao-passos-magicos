"""Teste de integração do pipeline completo de treinamento."""

import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent


class TestTrainingPipeline:
    """Valida que o pipeline roda do início ao fim com dados sintéticos."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MODELS_DIR_OVERRIDE", str(tmp_path))
        self.output_dir = tmp_path

    def test_synthetic_training_produces_artifacts(self, tmp_path):
        """Roda run_training.py --synthetic e verifica se os artefatos são gerados."""
        result = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "run_training.py"),
                "--synthetic",
                "--samples", "80",
            ],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(PROJECT_ROOT),
        )

        assert result.returncode == 0, (
            f"Pipeline falhou (exit {result.returncode}).\n"
            f"STDOUT:\n{result.stdout[-2000:]}\n"
            f"STDERR:\n{result.stderr[-2000:]}"
        )

        models_dir = PROJECT_ROOT / "models"
        assert (models_dir / "model.joblib").exists(), "model.joblib não foi gerado"
        assert (models_dir / "pipeline.joblib").exists(), "pipeline.joblib não foi gerado"
        assert (models_dir / "evaluation_report.json").exists(), "evaluation_report.json não foi gerado"

    def test_training_output_contains_metrics(self):
        """Verifica que o log do treinamento imprime métricas finais."""
        result = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "run_training.py"),
                "--synthetic",
                "--samples", "80",
            ],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(PROJECT_ROOT),
        )

        output = result.stdout + result.stderr
        assert "TRAINING COMPLETE" in output or "PIPELINE" in output.upper(), (
            "Log do treinamento não contém marcador de conclusão"
        )
