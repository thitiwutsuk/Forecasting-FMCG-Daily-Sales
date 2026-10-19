"""Conditional pre-launch evaluation, isolated from the short-history protocol.

Launch date/assortment are retrospectively inferred, NOT verified launch plans.
No target-SKU sales, observed price/promo, stock or delivery inputs are features.
"""
import numpy as np
import pandas as pd

from .forecast import fit_predict_lgb, prepare_categoricals

ZERO_HISTORY_FEATURES = ["category", "segment", "brand", "channel", "region",
                         "month", "week_number", "year"]


def build_launch_records(daily: pd.DataFrame) -> pd.DataFrame:
    """One recorded launch-calendar-week total per active SKU/channel/region.

    Week W covers [W,W+7d). Some launches fall mid-week; observed_days is audit
    only and never a feature. Omitted combinations are NOT synthesized as zero.
    Static attributes and launch assortment are assumed known to a planner.
    """
    from data.build_weekly_base import _week_agg
    daily = daily.copy()
    daily["date"] = pd.to_datetime(daily.date)
    attrs = daily[["sku", "category", "segment", "brand"]].drop_duplicates()
    if attrs.sku.duplicated().any():
        raise ValueError("category/segment/brand must be static per SKU")
    first_seen = daily.groupby("sku").date.min()
    launch = first_seen - pd.to_timedelta(first_seen.dt.dayofweek, unit="D")
    weekly = _week_agg(daily)
    rows = weekly[weekly.week.eq(weekly.sku.map(launch))].copy()
    rows = rows.rename(columns={"units_sold": "launch_units"})
    rows = rows[["sku", "channel", "region", "week", "launch_units"]]
    rows = rows.merge(attrs, on="sku", validate="many_to_one")
    rows["label_available_at"] = rows.week + pd.Timedelta(days=7)
    rows = rows[rows.label_available_at <= daily.date.max().normalize() + pd.Timedelta(days=1)].copy()
    daily["week"] = daily.date - pd.to_timedelta(daily.date.dt.dayofweek, unit="D")
    days = daily.groupby(["sku", "channel", "region", "week"]).date.nunique().rename("observed_days")
    rows = rows.merge(days, on=["sku", "channel", "region", "week"], validate="one_to_one")
    rows["first_seen"] = rows.sku.map(first_seen)
    rows["month"] = rows.week.dt.month
    rows["week_number"] = rows.week.dt.isocalendar().week.astype(int)
    rows["year"] = rows.week.dt.year
    return rows.sort_values(["week", "sku", "channel", "region"]).reset_index(drop=True)


def build_seven_day_launch_records(daily: pd.DataFrame) -> pd.DataFrame:
    """Recorded units in [SKU first observation, first observation + 7 days).

    Every SKU has a seven-calendar-day target window. Active combinations are
    inferred retrospectively from this window, not from a verified launch plan.
    Days without records are not imputed. observed_days is audit only.
    'week' is the launch-window START, which need not be a Monday; it is retained
    for compatibility with zero_history_forecast, not a calendar-week label.
    """
    daily = daily.copy()
    daily["date"] = pd.to_datetime(daily.date).dt.normalize()
    attrs = daily[["sku", "category", "segment", "brand"]].drop_duplicates()
    if attrs.sku.duplicated().any():
        raise ValueError("category/segment/brand must be static per SKU")
    first_seen = daily.groupby("sku").date.min()
    start = daily.sku.map(first_seen)
    end = start + pd.Timedelta(days=7)
    window = daily[(daily.date >= start) & (daily.date < end) &
                   (end <= daily.date.max() + pd.Timedelta(days=1))]
    rows = window.groupby(["sku", "channel", "region"]).agg(
        launch_units=("units_sold", "sum"), observed_days=("date", "nunique")
    ).reset_index().merge(attrs, on="sku", validate="many_to_one")
    rows["first_seen"] = rows.sku.map(first_seen)
    rows["week"] = rows.first_seen
    rows["launch_window_start"] = rows.first_seen
    rows["label_available_at"] = rows.first_seen + pd.Timedelta(days=7)
    rows["month"] = rows.first_seen.dt.month
    rows["week_number"] = rows.first_seen.dt.isocalendar().week.astype(int)
    rows["year"] = rows.first_seen.dt.year
    return rows.sort_values(["week", "sku", "channel", "region"]).reset_index(drop=True)


def zero_history_forecast(records: pd.DataFrame, held_out_skus: list, *,
                          include_category_baseline: bool = False) -> pd.DataFrame:
    """Fit completed launch windows of OTHER SKUs before target window start.

    Analog: same channel/region, best category/segment/brand match, then most
    recent completed launch and SKU ID. No observed target price breaks ties.
    LightGBM: fit metadata/calendar -> launch_units; frozen before launch.
    """
    pieces = []
    for sku in held_out_skus:
        test = records[records.sku == sku].copy()
        if test.empty or test.week.nunique() != 1:
            raise ValueError(f"missing or inconsistent launch for {sku!r}")
        cutoff = test.week.iloc[0]
        train = records[(~records.sku.isin(held_out_skus)) &
                        (records.label_available_at <= cutoff)].copy()
        if train.empty:
            raise ValueError(f"no completed training launches before {sku!r}")
        # Restrict both model frames explicitly; audit columns and target sales
        # never enter prediction inputs, even if caller appends other columns.
        xtrain = prepare_categoricals(train[ZERO_HISTORY_FEATURES])
        xtest = test[ZERO_HISTORY_FEATURES].copy()
        for col in ZERO_HISTORY_FEATURES:
            if isinstance(xtrain[col].dtype, pd.CategoricalDtype):
                xtest[col] = pd.Categorical(xtest[col], categories=xtrain[col].cat.categories)
        xtrain["target_next_week"] = train.launch_units.to_numpy()
        # Shared helper's internal label name only; our target is launch week,
        # not next week. Exported ledger keeps the explicit launch_units name.
        test["pred_zero_lgb"] = fit_predict_lgb(xtrain, xtest, feature_cols=ZERO_HISTORY_FEATURES)
        analog_values, donors, donor_dates = [], [], []
        for row in test.itertuples():
            pool = train[(train.channel == row.channel) & (train.region == row.region)].copy()
            pool["match_score"] = ((pool.category == row.category).astype(int) * 4 +
                                   (pool.segment == row.segment).astype(int) * 2 +
                                   (pool.brand == row.brand).astype(int))
            pool = pool.sort_values(["match_score", "week", "sku"], ascending=[False, False, True])
            if pool.empty:
                analog_values.append(np.nan); donors.append(None); donor_dates.append(pd.NaT)
            else:
                donor = pool.iloc[0]
                analog_values.append(donor.launch_units); donors.append(donor.sku)
                donor_dates.append(donor.label_available_at)
        test["pred_zero_analog"] = analog_values
        test["analog_sku"] = donors
        test["analog_label_available_at"] = donor_dates
        if include_category_baseline:
            means, levels, counts = [], [], []
            for row in test.itertuples():
                tiers = [
                    ("category_channel_region", train[(train.category == row.category) &
                        (train.channel == row.channel) & (train.region == row.region)]),
                    ("category", train[train.category == row.category]),
                    ("channel_region", train[(train.channel == row.channel) & (train.region == row.region)]),
                    ("global", train),
                ]
                for level, pool in tiers:
                    if not pool.empty:
                        means.append(pool.launch_units.mean())
                        levels.append(level); counts.append(len(pool))
                        break
            test["pred_category_mean"] = means
            test["baseline_level"] = levels
            test["baseline_donors"] = counts
        test["train_cutoff"] = cutoff
        test["train_label_available_max"] = train.label_available_at.max()
        test["train_rows"] = len(train)
        test["train_skus"] = train.sku.nunique()
        pieces.append(test)
    return pd.concat(pieces).sort_index()
