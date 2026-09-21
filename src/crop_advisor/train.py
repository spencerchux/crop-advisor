"""Train, compare, and save Crop Advisor classification pipelines."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from crop_advisor.config import (
    ARTIFACT_DIR,
    COMPARISON_PATH,
    DATA_PATH,
    FEATURE_COLUMNS,
    METADATA_PATH,
    MODEL_PATH,
    RANDOM_STATE,
)
from crop_advisor.data import DatasetSplit, load_clean_dataset, make_stratified_splits


def build_candidates() -> dict[str, Pipeline]:
    """Return the candidate models under consistent training interfaces."""
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=3_000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=500,
                        min_samples_leaf=1,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                )
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            steps=[
                (
                    "classifier",
                    HistGradientBoostingClassifier(
                        learning_rate=0.08,
                        max_iter=250,
                        l2_regularization=0.1,
                        random_state=RANDOM_STATE,
                    ),
                )
            ]
        ),
    }


def classification_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """Calculate metrics that remain informative under class imbalance."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def compare_models(split: DatasetSplit) -> tuple[pd.DataFrame, str, dict[str, Pipeline]]:
    """Fit candidates on training data and rank them on validation macro F1."""
    fitted: dict[str, Pipeline] = {}
    rows: list[dict[str, Any]] = []

    for name, model in build_candidates().items():
        print(f"Training {name}...")
        model.fit(split.X_train, split.y_train)
        predictions = model.predict(split.X_validation)
        rows.append({"model": name, **classification_metrics(split.y_validation, predictions)})
        fitted[name] = model

    comparison = (
        pd.DataFrame(rows)
        .sort_values(["macro_f1", "balanced_accuracy", "accuracy"], ascending=False)
        .reset_index(drop=True)
    )
    best_name = str(comparison.loc[0, "model"])
    return comparison, best_name, fitted


def sha256_file(path: Path) -> str:
    """Return a reproducibility hash for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def train_and_save(data_path: Path = DATA_PATH) -> dict[str, Any]:
    """Run model selection, final evaluation, refitting, and artifact export."""
    X, y = load_clean_dataset(data_path)
    split = make_stratified_splits(X, y)
    comparison, best_name, fitted = compare_models(split)

    best_validation_model = fitted[best_name]
    test_predictions = best_validation_model.predict(split.X_test)
    test_metrics = classification_metrics(split.y_test, test_predictions)

    X_train_final = pd.concat([split.X_train, split.X_validation])
    y_train_final = pd.concat([split.y_train, split.y_validation])
    final_model = clone(build_candidates()[best_name])
    final_model.fit(X_train_final, y_train_final)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    # Compression keeps the artifact comfortably below GitHub's 100 MB file limit.
    joblib.dump(final_model, MODEL_PATH, compress=3)
    comparison.to_csv(COMPARISON_PATH, index=False)

    feature_ranges = {
        feature: {"min": float(X[feature].min()), "max": float(X[feature].max())}
        for feature in FEATURE_COLUMNS
    }
    metadata: dict[str, Any] = {
        "model_name": best_name,
        "selection_metric": "validation_macro_f1",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "features": FEATURE_COLUMNS,
        "target": "Crop",
        "classes": sorted(y.unique().tolist()),
        "feature_ranges": feature_ranges,
        "dataset": {
            "path": str(data_path.relative_to(data_path.parents[2])),
            "sha256": sha256_file(data_path),
            "total_rows": len(X),
            "split_rows": {
                "train": len(split.X_train),
                "validation": len(split.X_validation),
                "test": len(split.X_test),
                "final_fit_train_plus_validation": len(X_train_final),
            },
        },
        "validation_results": comparison.to_dict(orient="records"),
        "held_out_test_metrics": test_metrics,
        "versions": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
        },
        "random_state": RANDOM_STATE,
        "notes": [
            "The supplied train and test CSVs contained the same records in different orders.",
            "Exact feature/target duplicates were removed before splitting.",
            "Humidity and location are not model features because they are absent from the dataset.",
        ],
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nValidation comparison:")
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSelected model: {best_name}")
    print("Held-out test metrics:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved metadata: {METADATA_PATH}")
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH, help="Path to the clean CSV dataset.")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    train_and_save(arguments.data.resolve())
