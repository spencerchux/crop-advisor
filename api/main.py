"""FastAPI application for Crop Advisor predictions."""

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import Depends, FastAPI

from api.schemas import (
    CropScore,
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)
from api.services import get_prediction_service
from crop_advisor.inference import PredictionService


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Fail fast during startup if model artifacts cannot be loaded."""
    get_prediction_service()
    yield


app = FastAPI(
    title="Crop Advisor API",
    description="Recommend crops from soil nutrients, pH, rainfall, and temperature.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/", include_in_schema=False)
def root() -> dict[str, str]:
    return {"message": "Crop Advisor API", "docs": "/docs"}


@app.get("/health", response_model=HealthResponse, tags=["Operations"])
def health(service: PredictionService = Depends(get_prediction_service)) -> HealthResponse:
    return HealthResponse(status="ok", model_loaded=service.model is not None)


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Operations"])
def model_info(service: PredictionService = Depends(get_prediction_service)) -> ModelInfoResponse:
    metadata = service.metadata
    return ModelInfoResponse(
        model_name=metadata["model_name"],
        features=metadata["features"],
        crop_count=len(metadata["classes"]),
        trained_at_utc=metadata["trained_at_utc"],
        held_out_test_metrics=metadata["held_out_test_metrics"],
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
def predict(
    request: PredictionRequest,
    service: PredictionService = Depends(get_prediction_service),
) -> PredictionResponse:
    result = service.predict(
        request.model_features(),
        humidity_provided=request.humidity is not None,
        location_provided=request.location is not None,
    )
    return PredictionResponse(
        recommended_crop=result.recommended_crop,
        confidence=result.confidence,
        alternatives=[
            CropScore(crop=item.crop, probability=item.probability)
            for item in result.alternatives
        ],
        warnings=result.warnings,
        model_name=service.metadata["model_name"],
    )

