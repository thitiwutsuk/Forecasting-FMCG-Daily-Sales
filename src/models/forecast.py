"""Core forecasting models (Phase 7): global pooled LightGBM, local per-SKU LightGBM,
global pooled XGBoost.

Feature/label split is centralized here so Phase 10 (cold-start) and Phase 11
(ablation) reuse the exact same column bookkeeping instead of redefining it.
"""

from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

TARGET = "target_next_week"

ID_COLS = ["sku", "week", "target_next_week"]

CATEGORICAL_COLS = ["sku", "channel", "region", "category", "segment", "brand", "lifecycle_stage"]

# Features that require the group's own sales/promo history and are therefore
# unavailable for a brand-new SKU (used to build the Phase 10 meta-learner).
HISTORY_DEPENDENT_COLS = [
    "lag_1",
    "lag_2",
    "rolling_mean_4",
    "rolling_std_4",
    "momentum",
    "promo_recency",
    "rolling_promo_rate",
]

ALL_FEATURE_COLS = [
    "sku", "channel", "region", "category", "segment", "brand", "lifecycle_stage",
    "units_sold", "stock_available", "promotion_flag", "price_unit", "delivery_days",
    "is_holiday_peak", "week_number", "month", "year", "is_holiday_week", "is_summer",
    "is_winter", "sku_age",
    "lag_1", "lag_2", "rolling_mean_4", "rolling_std_4", "momentum",
    "price_avg", "promo_rate", "stock_avg", "deliveries",
    "avg_temp", "inflation_index", "school_in_session", "event_score", "category_trend",
    "price_index", "promo_recency", "rolling_promo_rate", "cross_channel_demand_share",
]


def prepare_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Cast categorical columns to pandas 'category' dtype on the FULL dataset, once,
    before any train/val split — so every split shares the same category->code mapping
    (splitting after casting, via boolean/row-index slicing, preserves the category list).
    """
    out = df.copy()
    for col in CATEGORICAL_COLS:
        if col in out.columns:
            out[col] = out[col].astype("category")
    return out


def make_lgb_frame(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """Select feature columns, casting to 'category' only if not already categorical
    (prefer calling prepare_categoricals on the full df first to keep codes consistent).
    """
    X = df[feature_cols].copy()
    for col in CATEGORICAL_COLS:
        if col in X.columns and not isinstance(X[col].dtype, pd.CategoricalDtype):
            X[col] = X[col].astype("category")
    return X


DEFAULT_LGB_PARAMS = dict(
    # objective="regression" trains on L2 (squared error), not the WAPE-aligned L1
    # loss its name might suggest. We don't pass eval_set to .fit(), so a metric=
    # kwarg here would be inert (LightGBM only logs it against a held-out eval_set).
    # The XGBoost model below trains directly on reg:absoluteerror (true L1) as a
    # cross-check that the L2-trained LightGBM isn't materially worse on MAE/WAPE.
    objective="regression",
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=20,
    verbosity=-1,
    random_state=42,
)


def fit_predict_lgb(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: Optional[list] = None,
    params: Optional[dict] = None,
) -> np.ndarray:
    """Train one LightGBM regressor on train_df, return predictions for val_df."""
    feature_cols = feature_cols or ALL_FEATURE_COLS
    params = {**DEFAULT_LGB_PARAMS, **(params or {})}

    X_train = make_lgb_frame(train_df, feature_cols)
    y_train = train_df[TARGET].values
    X_val = make_lgb_frame(val_df, feature_cols)

    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train, categorical_feature=[c for c in CATEGORICAL_COLS if c in feature_cols])
    return model.predict(X_val)


def fit_lgb(train_df: pd.DataFrame, feature_cols: Optional[list] = None, params: Optional[dict] = None) -> lgb.LGBMRegressor:
    feature_cols = feature_cols or ALL_FEATURE_COLS
    params = {**DEFAULT_LGB_PARAMS, **(params or {})}
    X_train = make_lgb_frame(train_df, feature_cols)
    y_train = train_df[TARGET].values
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train, categorical_feature=[c for c in CATEGORICAL_COLS if c in feature_cols])
    return model


def predict_lgb(model: lgb.LGBMRegressor, df: pd.DataFrame, feature_cols: Optional[list] = None) -> np.ndarray:
    feature_cols = feature_cols or ALL_FEATURE_COLS
    X = make_lgb_frame(df, feature_cols)
    return model.predict(X)


DEFAULT_XGB_PARAMS = dict(
    objective="reg:absoluteerror",
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=20,
    tree_method="hist",
    enable_categorical=True,
    random_state=42,
)


def fit_predict_xgb(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: Optional[list] = None,
    params: Optional[dict] = None,
) -> np.ndarray:
    """Train one XGBoost regressor on train_df, return predictions for val_df.

    Mirrors fit_predict_lgb: same feature/categorical handling (native categorical
    support via enable_categorical, no one-hot), same global-pooled training scheme.
    """
    feature_cols = feature_cols or ALL_FEATURE_COLS
    params = {**DEFAULT_XGB_PARAMS, **(params or {})}

    X_train = make_lgb_frame(train_df, feature_cols)
    y_train = train_df[TARGET].values
    X_val = make_lgb_frame(val_df, feature_cols)

    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    return model.predict(X_val)


def fit_xgb(train_df: pd.DataFrame, feature_cols: Optional[list] = None, params: Optional[dict] = None) -> xgb.XGBRegressor:
    feature_cols = feature_cols or ALL_FEATURE_COLS
    params = {**DEFAULT_XGB_PARAMS, **(params or {})}
    X_train = make_lgb_frame(train_df, feature_cols)
    y_train = train_df[TARGET].values
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    return model


def fit_predict_local_per_sku(train_df: pd.DataFrame, val_df: pd.DataFrame, feature_cols: Optional[list] = None) -> np.ndarray:
    """Train one LightGBM per SKU (pooling its channel/region rows), predict val_df row-by-SKU.

    SKUs present in val_df but absent (or with <20 rows) in train_df fall back to the
    pooled-global prediction is not computed here; caller should combine with a global
    model for those cases. Returns NaN for SKUs with insufficient training rows.
    """
    feature_cols = feature_cols or ALL_FEATURE_COLS
    preds = pd.Series(index=val_df.index, dtype=float)

    for sku, val_sub in val_df.groupby("sku"):
        train_sub = train_df[train_df.sku == sku]
        if len(train_sub) < 20:
            continue
        preds.loc[val_sub.index] = fit_predict_lgb(train_sub, val_sub, feature_cols)

    return preds.values
