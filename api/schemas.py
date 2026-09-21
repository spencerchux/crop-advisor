"""Pydantic request and response contracts for the Crop Advisor API."""

from pydantic import BaseModel, ConfigDict, Field


class ApiModel(BaseModel):
    """Shared API model configuration."""

    model_config = ConfigDict(protected_namespaces=())


class PredictionRequest(ApiModel):
    """Farmer-provided soil and weather measurements."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        protected_namespaces=(),
    )

    nitrogen: float = Field(ge=0, le=500, description="Soil nitrogen value (N).", examples=[70])
    phosphorus: float = Field(ge=0, le=500, description="Soil phosphorus value (P).", examples=[40])
    potassium: float = Field(ge=0, le=500, description="Soil potassium value (K).", examples=[45])
    ph: float = Field(gt=0, le=14, description="Soil pH.", examples=[5.54])
    rainfall: float = Field(ge=0, le=10_000, description="Rainfall measurement in the dataset's unit.", examples=[75.32])
    temperature: float = Field(ge=-20, le=60, description="Temperature in degrees Celsius.", examples=[22.676])
    humidity: float | None = Field(default=None, ge=0, le=100, description="Optional context; not used by model v1.")
    location: str | None = Field(default=None, min_length=2, max_length=120, description="Optional context; not used by model v1.")

    def model_features(self) -> dict[str, float]:
        """Map public API names to the exact trained feature names."""
        return {
            "N": self.nitrogen,
            "P": self.phosphorus,
            "K": self.potassium,
            "pH": self.ph,
            "rainfall": self.rainfall,
            "temperature": self.temperature,
        }


class CropScore(ApiModel):
    crop: str
    probability: float = Field(ge=0, le=1)


class PredictionResponse(ApiModel):
    recommended_crop: str
    confidence: float = Field(ge=0, le=1)
    alternatives: list[CropScore]
    warnings: list[str]
    model_name: str


class HealthResponse(ApiModel):
    status: str
    model_loaded: bool


class ModelInfoResponse(ApiModel):
    model_name: str
    features: list[str]
    crop_count: int
    trained_at_utc: str
    held_out_test_metrics: dict[str, float]
