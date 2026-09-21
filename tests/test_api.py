"""Integration tests for FastAPI routes."""

from fastapi.testclient import TestClient


def test_health_endpoint(api_client: TestClient) -> None:
    response = api_client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_model_info_endpoint(api_client: TestClient) -> None:
    response = api_client.get("/model-info")
    body = response.json()

    assert response.status_code == 200
    assert body["model_name"] == "random_forest"
    assert body["crop_count"] == 40
    assert body["features"] == ["N", "P", "K", "pH", "rainfall", "temperature"]
    assert body["held_out_test_metrics"]["macro_f1"] > 0.90


def test_predict_endpoint(api_client: TestClient, valid_payload: dict[str, float]) -> None:
    response = api_client.post("/predict", json=valid_payload)
    body = response.json()

    assert response.status_code == 200
    assert body["recommended_crop"] == "barley"
    assert 0 <= body["confidence"] <= 1
    assert len(body["alternatives"]) == 2
    assert body["model_name"] == "random_forest"


def test_predict_reports_unused_context(api_client: TestClient, valid_payload: dict[str, float]) -> None:
    payload = {**valid_payload, "humidity": 70, "location": "Kano, Nigeria"}
    response = api_client.post("/predict", json=payload)

    assert response.status_code == 200
    assert len(response.json()["warnings"]) == 2


def test_predict_rejects_invalid_ph(api_client: TestClient, valid_payload: dict[str, float]) -> None:
    response = api_client.post("/predict", json={**valid_payload, "ph": 15})

    assert response.status_code == 422


def test_predict_rejects_unknown_fields(api_client: TestClient, valid_payload: dict[str, float]) -> None:
    response = api_client.post("/predict", json={**valid_payload, "unknown": 123})

    assert response.status_code == 422

