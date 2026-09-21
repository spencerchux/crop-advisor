"""Shared project configuration."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "crop_recommendation_clean.csv"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "model.joblib"
METADATA_PATH = ARTIFACT_DIR / "model_metadata.json"
COMPARISON_PATH = ARTIFACT_DIR / "model_comparison.csv"

TARGET_COLUMN = "Crop"
FEATURE_COLUMNS = ["N", "P", "K", "pH", "rainfall", "temperature"]
RANDOM_STATE = 42

