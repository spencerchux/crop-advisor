"""HTTP client used by the Streamlit frontend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests


class CropAdvisorAPIError(RuntimeError):
    """Raised when the prediction API cannot provide a usable response."""


@dataclass(frozen=True)
class CropAdvisorClient:
    """Small typed wrapper around the Crop Advisor REST API."""

    base_url: str
    timeout_seconds: float = 15.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "base_url", self.base_url.rstrip("/"))

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def model_info(self) -> dict[str, Any]:
        return self._request("GET", "/model-info")

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/predict", json=payload)

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        try:
            response = requests.request(
                method,
                f"{self.base_url}{path}",
                timeout=self.timeout_seconds,
                **kwargs,
            )
        except requests.RequestException as exc:
            raise CropAdvisorAPIError(
                "Could not reach the Crop Advisor API. Confirm that the API is running and the URL is correct."
            ) from exc

        if response.status_code == 422:
            try:
                detail = response.json().get("detail", [])
                messages = [item.get("msg", "Invalid value") for item in detail]
            except (ValueError, AttributeError):
                messages = ["One or more values were rejected."]
            raise CropAdvisorAPIError("Please check your inputs: " + "; ".join(messages))

        if not 200 <= response.status_code < 400:
            raise CropAdvisorAPIError(
                f"The API returned an error ({response.status_code}). Please try again later."
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise CropAdvisorAPIError("The API returned an unreadable response.") from exc
        if not isinstance(data, dict):
            raise CropAdvisorAPIError("The API returned an unexpected response format.")
        return data
