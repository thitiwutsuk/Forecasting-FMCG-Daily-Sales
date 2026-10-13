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

    `sku_attrs` must have a `first_seen` column (each SKU's first-observation date).
    The candidate pool is restricted to SKUs that launched strictly before target_sku
    -- a real cold-start forecast can only ever borrow from products that already
    existed at the time, so a candidate launched later (even if it's the closest match
    on category/segment/pack_type/price) is not a valid analog and must be excluded
    before similarity matching, not just deprioritized by it.
    """
    target = sku_attrs.loc[target_sku]
    pool = sku_attrs.loc[sku_attrs.index.isin(candidate_skus)]
    pool = pool[pool["first_seen"] < target["first_seen"]]
    if pool.empty:
        raise ValueError(
            f"no temporally-eligible analog for {target_sku!r}: none of the "
            f"{len(candidate_skus)} candidate SKUs launched before it "
            f"({pd.Timestamp(target['first_seen']).date()})"
        )

    for keys in (["category", "segment", "pack_type"], ["category", "pack_type"], ["category"]):
        mask = (pool[keys] == target[keys]).all(axis=1)
        matches = pool[mask]
        if len(matches):
            return matches.assign(price_dist=(matches["price_unit"] - target["price_unit"]).abs())["price_dist"].idxmin()

    return pool.assign(price_dist=(pool["price_unit"] - target["price_unit"]).abs())["price_dist"].idxmin()


def analog_forecast(df: pd.DataFrame, cold_start_skus: list, sku_attrs: pd.DataFrame) -> pd.DataFrame:
    """For each cold-start SKU, forecast target_next_week using the matched analog SKU's
    own units_sold at the same (channel, region, sku_age) — i.e. its early trajectory.

    `sku_attrs` must include a `first_seen` column (see find_analog_sku) so the match is
    restricted to SKUs that actually existed yet at the target's launch time.
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


def launch_week(df: pd.DataFrame, sku: str) -> pd.Timestamp:
    """Recover the SKU launch Monday from the original (untrimmed) age."""
    rows = df[df.sku == sku]
    if rows.empty:
        raise ValueError(f"no evaluation rows for {sku!r}")
    origins = pd.to_datetime(rows.week) - pd.to_timedelta(rows.sku_age * 7, unit="D")
    if origins.nunique() != 1:
        raise ValueError(f"inconsistent launch week for {sku!r}")
    return pd.Timestamp(origins.iloc[0])


def training_before_launch(df: pd.DataFrame, cold_start_skus: list, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Keep only labels fully observed before the launch Monday.

    Rows are Monday-labelled, left-closed weeks. A row at W predicts sales
    during [W+7d, W+14d), so its label is available only at W+14d.
    """
    available_at = pd.to_datetime(df.week) + pd.Timedelta(days=14)
    return df[(~df.sku.isin(cold_start_skus)) & (available_at <= cutoff)].copy()


def _launch_model_forecast(df, cold_start_skus, feature_cols, prediction_col):
    pieces = []
    for sku in cold_start_skus:
        cutoff = launch_week(df, sku)
        train = training_before_launch(df, cold_start_skus, cutoff)
        test = df[df.sku == sku].copy()
        if train.empty:
            raise ValueError(f"no training labels available before launch of {sku!r}")
        # Align category vocabularies using training values only. Unseen SKU IDs
        # become missing categories rather than invented training levels.
        train = prepare_categoricals(train)
        model_test = test.copy()
        for col in feature_cols:
            if isinstance(train[col].dtype, pd.CategoricalDtype):
                model_test[col] = pd.Categorical(test[col], categories=train[col].cat.categories)
        test[prediction_col] = fit_predict_lgb(train, model_test, feature_cols=feature_cols)
        test["train_cutoff"] = cutoff
        test["train_label_available_max"] = pd.to_datetime(train.week).max() + pd.Timedelta(days=14)
        test["train_rows"] = len(train)
        pieces.append(test)
    return pd.concat(pieces).sort_index()


def meta_learner_forecast(df: pd.DataFrame, cold_start_skus: list) -> pd.DataFrame:
    """Frozen launch-time model without lag features, for short-history rows.

    This still uses current-week sales: it is NOT a zero-history forecast.
    All evaluation SKUs are excluded; training labels precede each SKU launch.
    """
    return _launch_model_forecast(df, cold_start_skus, NON_LAG_FEATURE_COLS, "pred_meta_learner")


def mature_model_forecast(df: pd.DataFrame, cold_start_skus: list) -> pd.DataFrame:
    """Frozen launch-time model using history available at each scoring week."""
    return _launch_model_forecast(df, cold_start_skus, ALL_FEATURE_COLS, "pred_full_model")
