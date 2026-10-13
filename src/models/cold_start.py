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


def launch_analog_attributes(daily: pd.DataFrame, target_sku: str, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Candidate price averages from completed pre-launch observations only.

    Target price is a first-day observed-price proxy for a launch price supplied
    by a planner. It is not claimed to be known before launch in this dataset.
    Pack type is deliberately excluded: it varies within each SKU.
    """
    first_seen = daily.groupby("sku")["date"].min()
    history = daily[pd.to_datetime(daily.date) < cutoff]
    attrs = history.groupby("sku").agg(category=("category", "first"),
        segment=("segment", "first"), price_unit=("price_unit", "mean"))
    target = daily[(daily.sku == target_sku) & (daily.date == first_seen.loc[target_sku])]
    attrs.loc[target_sku] = [target.category.iloc[0], target.segment.iloc[0], target.price_unit.mean()]
    attrs["first_seen"] = first_seen.reindex(attrs.index)
    return attrs


def find_launch_analog(target_sku, attrs, candidate_skus, cutoff):
    pool = attrs.loc[attrs.index.isin(candidate_skus)]
    pool = pool[pool.first_seen < cutoff].sort_index()
    if pool.empty:
        raise ValueError(f"no temporally-eligible analog for {target_sku!r}")
    target = attrs.loc[target_sku]
    for keys in (["category", "segment"], ["category"], []):
        matches = pool if not keys else pool[(pool[keys] == target[keys]).all(axis=1)]
        if not matches.empty:
            return (matches.price_unit - target.price_unit).abs().idxmin()


def analog_forecast(df: pd.DataFrame, cold_start_skus: list, sku_attrs: pd.DataFrame,
                    daily: pd.DataFrame = None) -> pd.DataFrame:
    """Borrow the analog's NEXT-week sales at the same age/channel/region.

    With daily supplied, matching attributes are reconstructed at launch.
    Without daily, the caller must supply point-in-time price attributes.
    Trajectories are frozen at launch; unavailable labels remain NaN.
    """
    candidates = [s for s in df.sku.unique() if s not in cold_start_skus]
    pieces = []
    for sku in cold_start_skus:
        cutoff = launch_week(df, sku)
        attrs = launch_analog_attributes(daily, sku, cutoff) if daily is not None else sku_attrs
        analog = find_launch_analog(sku, attrs, candidates, cutoff)
        history = training_before_launch(df, cold_start_skus, cutoff)
        lookup = history[history.sku == analog].set_index(["channel", "region", "sku_age"])["target_next_week"]
        if not lookup.index.is_unique:
            raise ValueError("duplicate analog channel/region/age keys")
        rows = df[df.sku == sku].copy()
        keys = pd.MultiIndex.from_frame(rows[["channel", "region", "sku_age"]])
        rows["pred_analog"] = lookup.reindex(keys).to_numpy()
        rows["analog_sku"] = analog
        rows["train_cutoff"] = cutoff
        pieces.append(rows)
    return pd.concat(pieces).sort_index()


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
