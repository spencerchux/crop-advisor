"""Tests for the cleaned dataset contract and stratified splitting."""

from __future__ import annotations

import pandas as pd
import pytest

from crop_advisor.config import DATA_PATH, FEATURE_COLUMNS
from crop_advisor.data import load_clean_dataset, make_stratified_splits


def test_project_clean_dataset_satisfies_contract() -> None:
    X, y = load_clean_dataset(DATA_PATH)

    assert X.columns.tolist() == FEATURE_COLUMNS
    assert len(X) == len(y) == 7_165
    assert y.nunique() == 40
    assert not X.isna().any().any()


def test_loader_rejects_missing_required_column(tmp_path) -> None:
    invalid = pd.DataFrame({"Crop": ["rice", "wheat", "maize"], "N": [1, 2, 3]})
    path = tmp_path / "invalid.csv"
    invalid.to_csv(path, index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        load_clean_dataset(path)


def test_loader_rejects_duplicate_observations(tmp_path) -> None:
    row = {
        "Crop": "rice",
        "N": 80,
        "P": 40,
        "K": 40,
        "pH": 6.0,
        "rainfall": 200,
        "temperature": 27,
    }
    path = tmp_path / "duplicates.csv"
    pd.DataFrame([row, row, {**row, "Crop": "wheat"}]).to_csv(path, index=False)

    with pytest.raises(ValueError, match="duplicate"):
        load_clean_dataset(path)


def test_stratified_split_is_complete_and_disjoint() -> None:
    X, y = load_clean_dataset(DATA_PATH)
    split = make_stratified_splits(X, y)

    all_indices = [
        set(split.X_train.index),
        set(split.X_validation.index),
        set(split.X_test.index),
    ]
    assert sum(map(len, all_indices)) == len(X)
    assert all_indices[0].isdisjoint(all_indices[1])
    assert all_indices[0].isdisjoint(all_indices[2])
    assert all_indices[1].isdisjoint(all_indices[2])
    assert set(split.y_train.unique()) == set(y.unique())
    assert set(split.y_validation.unique()) == set(y.unique())
    assert set(split.y_test.unique()) == set(y.unique())

