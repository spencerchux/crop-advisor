"""Application service dependencies."""

from functools import lru_cache

from crop_advisor.inference import PredictionService


@lru_cache(maxsize=1)
def get_prediction_service() -> PredictionService:
    """Create one model service per API process."""
    return PredictionService()

