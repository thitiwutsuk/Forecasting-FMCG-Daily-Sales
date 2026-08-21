"""Cold-start forecasting (Phase 10): analog-based and meta-learner approaches
for SKUs with no sales history yet.
"""

import numpy as np
import pandas as pd

from .forecast import ALL_FEATURE_COLS, HISTORY_DEPENDENT_COLS, fit_predict_lgb, prepare_categoricals

NON_LAG_FEATURE_COLS = [c for c in ALL_FEATURE_COLS if c not in HISTORY_DEPENDENT_COLS]


def find_analog_sku(target_sku: str, sku_attrs: pd.DataFrame, candidate_skus: list) -> str:
    """Nearest existing SKU by category -> segment -> pack_type exact match, tie-broken
    by closest price_unit. Falls back to progressively coarser matches (category+pack_type,
    then category only) if no exact category+segment+pack_type match exists.
    """
    target = sku_attrs.loc[target_sku]
    pool = sku_attrs.loc[sku_attrs.index.isin(candidate_skus)]

    for keys in (["category", "segment", "pack_type"], ["category", "pack_type"], ["category"]):
        mask = (pool[keys] == target[keys]).all(axis=1)
        matches = pool[mask]
        if len(matches):
            return matches.assign(price_dist=(matches["price_unit"] - target["price_unit"]).abs())["price_dist"].idxmin()

    return pool.assign(price_dist=(pool["price_unit"] - target["price_unit"]).abs())["price_dist"].idxmin()


def analog_forecast(df: pd.DataFrame, cold_start_skus: list, sku_attrs: pd.DataFrame) -> pd.DataFrame:
    """For each cold-start SKU, forecast target_next_week using the matched analog SKU's
    own units_sold at the same (channel, region, sku_age) — i.e. its early trajectory.
    """
    mature_skus = [s for s in df["sku"].unique() if s not in cold_start_skus]
    analogs = {sku: find_analog_sku(sku, sku_attrs, mature_skus) for sku in cold_start_skus}

    analog_lookup = df[df.sku.isin(analogs.values())].set_index(["sku", "channel", "region", "sku_age"])["units_sold"]

    rows = df[df.sku.isin(cold_start_skus)].copy()
    rows["analog_sku"] = rows["sku"].map(analogs)

    def lookup_pred(row):
        key = (row["analog_sku"], row["channel"], row["region"], row["sku_age"])
        return analog_lookup.get(key, np.nan)

    rows["pred_analog"] = rows.apply(lookup_pred, axis=1)
    return rows


def meta_learner_forecast(df: pd.DataFrame, cold_start_skus: list) -> pd.DataFrame:
    """Train a LightGBM restricted to non-history-dependent features on mature SKUs only,
    predict target_next_week for the cold-start SKUs' rows (their entire lifetime, so the
    accuracy-vs-sku_age curve can be traced).
    """
    dfc = prepare_categoricals(df)
    train = dfc[~dfc.sku.isin(cold_start_skus)]
    test = dfc[dfc.sku.isin(cold_start_skus)].copy()

    pred = fit_predict_lgb(train, test, feature_cols=NON_LAG_FEATURE_COLS)
    test["pred_meta_learner"] = pred
    return test


def mature_model_forecast(df: pd.DataFrame, cold_start_skus: list) -> pd.DataFrame:
    """Same setup but with the FULL feature set (including lag/rolling) — used to show
    how much lag features would help once the cold-start SKU actually has history
    (only meaningful for rows where sku_age is large enough that lag_1/lag_2 aren't NaN).
    """
    dfc = prepare_categoricals(df)
    train = dfc[~dfc.sku.isin(cold_start_skus)]
    test = dfc[dfc.sku.isin(cold_start_skus)].copy()

    pred = fit_predict_lgb(train, test, feature_cols=ALL_FEATURE_COLS)
    test["pred_full_model"] = pred
    return test
