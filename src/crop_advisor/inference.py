"""Model loading and prediction logic shared by API and frontend clients."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from crop_advisor.config import FEATURE_COLUMNS, METADATA_PATH, MODEL_PATH


@dataclass(frozen=True)
class CropProbability:
    """A crop and its model-estimated probability."""

    crop: str
    probability: float


@dataclass(frozen=True)
class PredictionResult:
    """Structured prediction returned by the inference service."""

    recommended_crop: str
    confidence: float
    alternatives: list[CropProbability]
    warnings: list[str]


class PredictionService:
    """Load the trained pipeline once and provide validated predictions."""

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        metadata_path: Path = METADATA_PATH,
    ) -> None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found at {model_path}. Run `python -m crop_advisor.train` first."
            )
        if not metadata_path.exists():
            raise FileNotFoundError(f"Model metadata not found at {metadata_path}.")

        self.model = joblib.load(model_path)
        self.metadata: dict[str, Any] = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata_features = self.metadata.get("features")
        if metadata_features != FEATURE_COLUMNS:
            raise ValueError(
                f"Model feature contract mismatch: expected {FEATURE_COLUMNS}, got {metadata_features}."
            )
        if not hasattr(self.model, "predict_proba"):
            raise TypeError("Saved model must implement predict_proba for confidence results.")

    def _range_warnings(self, features: dict[str, float]) -> list[str]:
        warnings: list[str] = []
        ranges = self.metadata.get("feature_ranges", {})
        for feature, value in features.items():
            limits = ranges.get(feature)
            if not limits:
                continue
            minimum = float(limits["min"])
            maximum = float(limits["max"])
            if value < minimum or value > maximum:
                warnings.append(
                    f"{feature}={value:g} is outside the training range [{minimum:g}, {maximum:g}]."
                )
        return warnings

    def predict(
        self,
        features: dict[str, float],
        *,
        humidity_provided: bool = False,
        location_provided: bool = False,
        top_k: int = 3,
    ) -> PredictionResult:
        """Predict a crop and return the highest-probability alternatives."""
        missing = sorted(set(FEATURE_COLUMNS) - set(features))
        extra = sorted(set(features) - set(FEATURE_COLUMNS))
        if missing or extra:
            raise ValueError(f"Invalid feature keys. Missing: {missing}; unexpected: {extra}.")

        frame = pd.DataFrame([[features[name] for name in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)
        probabilities = np.asarray(self.model.predict_proba(frame)[0], dtype=float)
        classes = np.asarray(self.model.classes_, dtype=str)
        ranked_indices = np.argsort(probabilities)[::-1][: max(1, min(top_k, len(classes)))]
        ranked = [
            CropProbability(crop=str(classes[index]), probability=float(probabilities[index]))
            for index in ranked_indices
        ]

        warnings = self._range_warnings(features)
        if humidity_provided:
            warnings.append("Humidity was accepted for context but is not used by this model.")
        if location_provided:
            warnings.append("Location was accepted for context but is not used by this model.")

        return PredictionResult(
            recommended_crop=ranked[0].crop,
            confidence=ranked[0].probability,
            alternatives=ranked[1:],
            warnings=warnings,
        )

