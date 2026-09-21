"""Unit tests for the Streamlit HTTP client wrapper."""

from __future__ import annotations

import pytest
import requests

from app.api_client import CropAdvisorAPIError, CropAdvisorClient


class FakeResponse:
    def __init__(self, status_code: int, payload: dict) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> dict:
        return self._payload


def test_client_normalizes_base_url(monkeypatch) -> None:
    captured: dict[str, str] = {}

    def fake_request(method, url, **kwargs):
        captured["url"] = url
        return FakeResponse(200, {"status": "ok"})

    monkeypatch.setattr(requests, "request", fake_request)
    client = CropAdvisorClient("http://localhost:8000/")

    assert client.health() == {"status": "ok"}
    assert captured["url"] == "http://localhost:8000/health"


def test_client_turns_validation_details_into_readable_error(monkeypatch) -> None:
    monkeypatch.setattr(
        requests,
        "request",
        lambda *args, **kwargs: FakeResponse(422, {"detail": [{"msg": "Input should be less than 14"}]}),
    )

    with pytest.raises(CropAdvisorAPIError, match="Please check your inputs"):
        CropAdvisorClient("http://localhost:8000").predict({})


def test_client_handles_connection_failure(monkeypatch) -> None:
    def fail(*args, **kwargs):
        raise requests.ConnectionError("offline")

    monkeypatch.setattr(requests, "request", fail)

    with pytest.raises(CropAdvisorAPIError, match="Could not reach"):
        CropAdvisorClient("http://localhost:8000").health()

