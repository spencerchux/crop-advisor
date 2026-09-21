"""Tests for model inference and warning behavior."""

from __future__ import annotations

import pytest

from crop_advisor.inference import PredictionService


VALID_FEATURES = {
    "N": 70.0,
    "P": 40.0,
    "K": 45.0,
    "pH": 5.54,
    "rainfall": 75.32,
    "temperature": 22.676,
}


def test_prediction_returns_ranked_probabilities(prediction_service: PredictionService) -> None:
    result = prediction_service.predict(VALID_FEATURES)

    assert result.recommended_crop == "barley"
    assert 0 <= result.confidence <= 1
    assert len(result.alternatives) == 2
    probabilities = [result.confidence, *(item.probability for item in result.alternatives)]
    assert probabilities == sorted(probabilities, reverse=True)
    assert result.warnings == []


def test_optional_unsupported_inputs_create_warnings(prediction_service: PredictionService) -> None:
    result = prediction_service.predict(
        VALID_FEATURES,
        humidity_provided=True,
        location_provided=True,
    )

    assert any("Humidity" in warning for warning in result.warnings)
    assert any("Location" in warning for warning in result.warnings)


def test_out_of_training_range_creates_warning(prediction_service: PredictionService) -> None:
    result = prediction_service.predict({**VALID_FEATURES, "N": 300.0})

    assert any("outside the training range" in warning for warning in result.warnings)


def test_invalid_feature_contract_is_rejected(prediction_service: PredictionService) -> None:
    features = VALID_FEATURES.copy()
    features.pop("K")

    with pytest.raises(ValueError, match="Invalid feature keys"):
        prediction_service.predict(features)

