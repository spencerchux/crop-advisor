"""Shared pytest fixtures and import configuration."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
for import_path in (PROJECT_ROOT, SRC_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from api.main import app  # noqa: E402
from crop_advisor.inference import PredictionService  # noqa: E402


@pytest.fixture(scope="session")
def prediction_service() -> PredictionService:
    return PredictionService()


@pytest.fixture()
def api_client() -> TestClient:
    with TestClient(app) as client:
        yield client


@pytest.fixture()
def valid_payload() -> dict[str, float]:
    return {
        "nitrogen": 70,
        "phosphorus": 40,
        "potassium": 45,
        "ph": 5.54,
        "rainfall": 75.32,
        "temperature": 22.676,
    }

