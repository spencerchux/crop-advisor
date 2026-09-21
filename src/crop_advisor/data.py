"""Dataset loading, validation, and splitting utilities."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from crop_advisor.config import FEATURE_COLUMNS, RANDOM_STATE, TARGET_COLUMN


@dataclass(frozen=True)
class DatasetSplit:
    """Train, validation, and test partitions."""

    X_train: pd.DataFrame
    X_validation: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_validation: pd.Series
    y_test: pd.Series


def load_clean_dataset(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load the cleaned dataset and enforce its modeling contract."""
    data = pd.read_csv(path)
    required = [TARGET_COLUMN, *FEATURE_COLUMNS]
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    data = data[required].copy()
    if data.isna().any().any():
        raise ValueError("Clean dataset contains missing values.")
    if data.duplicated(subset=required).any():
        raise ValueError("Clean dataset contains duplicate feature/target rows.")
    if data[TARGET_COLUMN].nunique() < 2:
        raise ValueError("Training requires at least two crop classes.")
    if data[TARGET_COLUMN].value_counts().min() < 3:
        raise ValueError("Every crop needs at least three records for stratified splitting.")

    X = data[FEATURE_COLUMNS].astype(float)
    y = data[TARGET_COLUMN].astype(str)
    return X, y


def make_stratified_splits(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = RANDOM_STATE,
) -> DatasetSplit:
    """Create 70% train, 15% validation, and 15% test partitions."""
    X_train, X_temporary, y_train, y_temporary = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=random_state,
        stratify=y,
    )
    X_validation, X_test, y_validation, y_test = train_test_split(
        X_temporary,
        y_temporary,
        test_size=0.50,
        random_state=random_state,
        stratify=y_temporary,
    )
    return DatasetSplit(
        X_train=X_train,
        X_validation=X_validation,
        X_test=X_test,
        y_train=y_train,
        y_validation=y_validation,
        y_test=y_test,
    )

