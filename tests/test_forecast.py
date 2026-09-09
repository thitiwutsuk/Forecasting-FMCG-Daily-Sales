"""Regression tests for src/models/forecast.py's Random Forest frame prep.

Locks in the isolation guarantee for make_rf_frame: it must not mutate the input
DataFrame, and its output must be numeric/NaN-free (sklearn's RandomForestRegressor
rejects both category dtype and NaN, unlike LightGBM/XGBoost).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models.forecast import CATEGORICAL_COLS, make_rf_frame


def _toy_frame() -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "sku": pd.Series(["SKU-1", "SKU-2", "SKU-1"], dtype="category"),
            "channel": pd.Series(["Retail", "Online", "Retail"], dtype="category"),
            "promo_recency": [3.0, np.nan, 1.0],
            "units_sold": [10, 20, 15],
        }
    )
    return df


def test_make_rf_frame_has_no_nan():
    df = _toy_frame()
    X = make_rf_frame(df, ["sku", "channel", "promo_recency", "units_sold"])
    assert X.isnull().sum().sum() == 0


def test_make_rf_frame_categoricals_are_numeric():
    df = _toy_frame()
    X = make_rf_frame(df, ["sku", "channel", "promo_recency", "units_sold"])
    for col in ["sku", "channel"]:
        assert pd.api.types.is_numeric_dtype(X[col])


def test_make_rf_frame_does_not_mutate_input():
    df = _toy_frame()
    original_promo_recency_nulls = df["promo_recency"].isnull().sum()
    original_sku_dtype = df["sku"].dtype

    make_rf_frame(df, ["sku", "channel", "promo_recency", "units_sold"])

    assert df["promo_recency"].isnull().sum() == original_promo_recency_nulls
    assert df["sku"].dtype == original_sku_dtype
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            assert isinstance(df[col].dtype, pd.CategoricalDtype)
