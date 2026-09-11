"""Regression tests for src/models/forecast.py's Random Forest frame prep.

Locks in the isolation guarantee for make_rf_frame: it must not mutate the input
DataFrame, and its output must be numeric/NaN-free (sklearn's RandomForestRegressor
rejects both category dtype and NaN, unlike LightGBM/XGBoost). Also locks in that
categorical codes stay consistent between train and validation frames when encoded
via fit_rf_categories()/make_rf_frame(..., categories=...) — the bug this file was
originally missing coverage for: encoding train_df and val_df independently let the
same code mean a different label on each side whenever a category was missing from
one of them (e.g. reordered categories, or a category absent from one split).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models.forecast import CATEGORICAL_COLS, fit_rf_categories, make_rf_frame


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


def test_fit_rf_categories_reads_vocabulary_from_train_only():
    train_df = pd.DataFrame({"channel": pd.Series(["Online", "Retail", "Online"], dtype="category")})
    categories = fit_rf_categories(train_df, ["channel"])
    assert list(categories["channel"]) == ["Online", "Retail"]


def test_make_rf_frame_codes_align_when_val_has_reordered_categories():
    # train sees Online/Retail in that order; val's own unique-value order is reversed.
    # Encoding each frame independently (the original bug) would give val's "Online" a
    # different code than train's "Online" purely from row order.
    train_df = pd.DataFrame({"channel": pd.Series(["Online", "Retail"], dtype="category")})
    val_df = pd.DataFrame({"channel": pd.Series(["Retail", "Online"], dtype="category")})

    categories = fit_rf_categories(train_df, ["channel"])
    X_train = make_rf_frame(train_df, ["channel"], categories=categories)
    X_val = make_rf_frame(val_df, ["channel"], categories=categories)

    online_code = X_train.loc[train_df["channel"] == "Online", "channel"].iloc[0]
    retail_code = X_train.loc[train_df["channel"] == "Retail", "channel"].iloc[0]
    assert X_val.loc[val_df["channel"] == "Online", "channel"].iloc[0] == online_code
    assert X_val.loc[val_df["channel"] == "Retail", "channel"].iloc[0] == retail_code
    assert online_code != retail_code


def test_make_rf_frame_codes_align_when_a_category_is_missing_from_val():
    # "Wholesale" only appears in train; val only has Online/Retail. Independently
    # re-deriving val's vocabulary from its own rows would shift Retail's code left.
    train_df = pd.DataFrame({"channel": pd.Series(["Online", "Retail", "Wholesale"], dtype="category")})
    val_df = pd.DataFrame({"channel": pd.Series(["Retail", "Retail"], dtype="category")})

    categories = fit_rf_categories(train_df, ["channel"])
    X_train = make_rf_frame(train_df, ["channel"], categories=categories)
    X_val = make_rf_frame(val_df, ["channel"], categories=categories)

    retail_code_train = X_train.loc[train_df["channel"] == "Retail", "channel"].iloc[0]
    assert (X_val["channel"] == retail_code_train).all()


def test_make_rf_frame_unseen_category_encodes_as_minus_one():
    train_df = pd.DataFrame({"channel": pd.Series(["Online", "Retail"], dtype="category")})
    val_df = pd.DataFrame({"channel": pd.Series(["Online", "NewChannel"], dtype="category")})

    categories = fit_rf_categories(train_df, ["channel"])
    X_val = make_rf_frame(val_df, ["channel"], categories=categories)

    assert X_val.loc[val_df["channel"] == "NewChannel", "channel"].iloc[0] == -1
    assert X_val.loc[val_df["channel"] == "Online", "channel"].iloc[0] != -1


def test_fit_rf_categories_ignores_unused_levels_declared_on_the_dtype():
    # Mirrors calling prepare_categoricals() on the FULL dataset (so the dtype carries
    # every label that appears anywhere) and then slicing out a train fold whose rows
    # happen not to contain one of those labels. train_df's own rows are Online/Retail
    # only, but its category dtype -- inherited from the full-dataset cast -- still
    # declares "FutureOnly" as a valid level. The vocabulary must reflect what's
    # actually present in train_df's rows, not what the dtype merely allows, or a
    # label with zero training rows would encode as a real code instead of -1.
    full_df = pd.DataFrame({"channel": pd.Series(["Online", "Retail", "FutureOnly", "Online"], dtype="category")})
    train_df = full_df.iloc[[0, 1, 3]]  # Online, Retail, Online -- FutureOnly never observed here
    assert "FutureOnly" not in train_df["channel"].unique()
    assert "FutureOnly" in train_df["channel"].cat.categories  # the dtype still declares it

    categories = fit_rf_categories(train_df, ["channel"])
    assert "FutureOnly" not in list(categories["channel"])

    val_df = pd.DataFrame({"channel": pd.Series(["FutureOnly"], dtype="category")})
    X_val = make_rf_frame(val_df, ["channel"], categories=categories)
    assert X_val["channel"].iloc[0] == -1
