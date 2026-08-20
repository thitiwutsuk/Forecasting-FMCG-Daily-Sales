"""External/contextual enrichment features (Phase 4).

The MI-006 prototype (`data/raw/df_weekly_MI-006_enriched.csv`) has a bug: `avg_temp`
and `inflation_index` vary by sales channel within the same week/region, even though
weather and national inflation cannot plausibly depend on which channel a product is
sold through. The within-week-across-channel noise in that prototype has a standard
deviation several times larger than its own trend/seasonal range (see
`notebooks/04_feature_engineering.ipynb`), so the true signal cannot be recovered from
it — these functions regenerate every enrichment column deterministically at the grain
where it plausibly belongs, instead of patching the buggy file.

Grains used (never `channel`, since none of these are channel-dependent in reality):
- avg_temp: region x week
- inflation_index: week (national, one value per week)
- school_in_session: week (calendar rule)
- event_score: week (national retail-holiday calendar)
- category_trend: category x region x week
"""

import numpy as np
import pandas as pd

# Polish Easter Sunday for the years covered by the dataset (2022-2024) — used to
# anchor the moving Easter/May-holiday cluster in the event calendar.
EASTER_SUNDAY = {2022: "2022-04-17", 2023: "2023-04-09", 2024: "2024-03-31"}

# Fixed-date Polish public/retail holidays relevant to FMCG demand.
FIXED_HOLIDAYS_MMDD = ["05-01", "05-03", "11-01", "12-24", "12-25", "12-26"]


def _week_starts(weeks: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(pd.to_datetime(weeks))


def generate_avg_temp(regions: list, weeks: pd.DatetimeIndex, seed: int = 42) -> pd.DataFrame:
    """Deterministic seasonal temperature (°C) per region x week, with small region-specific noise."""
    rng = np.random.default_rng(seed)
    week_s = _week_starts(weeks)
    day_of_year = week_s.dt.dayofyear.to_numpy()

    region_params = {
        region: {"base": 9.0 + i * 0.6, "amplitude": 9.0 + i * 0.5, "phase_days": 172}
        for i, region in enumerate(sorted(regions))
    }

    rows = []
    for region, params in region_params.items():
        seasonal = params["base"] + params["amplitude"] * np.sin(
            2 * np.pi * (day_of_year - params["phase_days"]) / 365.25
        )
        noise = rng.normal(0, 0.8, size=len(week_s))
        rows.append(pd.DataFrame({"region": region, "week": week_s.values, "avg_temp": seasonal + noise}))
    return pd.concat(rows, ignore_index=True)


def generate_inflation_index(weeks: pd.DatetimeIndex, seed: int = 7, start: float = 160.0, annual_rate: float = 0.018) -> pd.DataFrame:
    """Deterministic national inflation index, one value per week (no region/channel dependency)."""
    rng = np.random.default_rng(seed)
    week_s = _week_starts(weeks).drop_duplicates().sort_values().reset_index(drop=True)
    weeks_elapsed = (week_s - week_s.min()).dt.days / 7.0
    trend = start * (1 + annual_rate) ** (weeks_elapsed / 52.0)
    noise = rng.normal(0, 0.6, size=len(week_s))
    return pd.DataFrame({"week": week_s.values, "inflation_index": trend + noise})


def generate_school_in_session(weeks: pd.DatetimeIndex) -> pd.DataFrame:
    """Polish school calendar approximation: out of session during the July-August summer break."""
    week_s = _week_starts(weeks).drop_duplicates().sort_values().reset_index(drop=True)
    in_session = (~week_s.dt.month.isin([7, 8])).astype(int)
    return pd.DataFrame({"week": week_s.values, "school_in_session": in_session})


def _easter_holiday_weeks(years: list) -> list:
    dates = []
    for year in years:
        if year in EASTER_SUNDAY:
            dates.append(pd.Timestamp(EASTER_SUNDAY[year]))
    return dates


def generate_event_score(weeks: pd.DatetimeIndex, seed: int = 99) -> pd.DataFrame:
    """Deterministic proximity score to Polish retail-relevant holidays, one value per week."""
    rng = np.random.default_rng(seed)
    week_s = _week_starts(weeks).drop_duplicates().sort_values().reset_index(drop=True)
    years = sorted(week_s.dt.year.unique().tolist())

    holiday_dates = _easter_holiday_weeks(years)
    for year in years:
        for mmdd in FIXED_HOLIDAYS_MMDD:
            holiday_dates.append(pd.Timestamp(f"{year}-{mmdd}"))

    scores = np.zeros(len(week_s))
    for i, wk in enumerate(week_s):
        days_to_nearest = min(abs((wk - hd).days) for hd in holiday_dates)
        if days_to_nearest <= 7:
            scores[i] = 1.0 - (days_to_nearest / 7.0) * 0.15 + rng.normal(0, 0.05)
    scores = np.clip(scores, 0, None)
    return pd.DataFrame({"week": week_s.values, "event_score": scores})


def compute_category_trend(weekly: pd.DataFrame) -> pd.DataFrame:
    """Causal category-momentum index: category x region *average* sku-channel weekly sales,
    expanding-normalized to its own first week.

    Uses the per-sku-channel mean rather than the summed total specifically so the index
    tracks organic demand momentum, not the count of SKUs active in that category/region
    (which grows mechanically over 2022-2023 as more SKUs launch — see Phase 2 EDA note
    on the same effect in the raw sales trend). Uses only weeks up to and including the
    current one within each (category, region) group, so it carries no leakage into
    target_next_week.
    """
    cat_region_weekly = (
        weekly.groupby(["category", "region", "week"])["units_sold"].mean().reset_index().sort_values(["category", "region", "week"])
    )
    cat_region_weekly["category_trend"] = cat_region_weekly.groupby(["category", "region"])["units_sold"].transform(
        lambda s: s.expanding().mean() / s.expanding().mean().iloc[0]
    )
    return cat_region_weekly[["category", "region", "week", "category_trend"]]


def compute_internal_aggregates(daily: pd.DataFrame) -> pd.DataFrame:
    """price_avg, promo_rate, stock_avg, deliveries per sku x channel x region x week.

    Formula verified against the MI-006 prototype (0 mismatches over 1,349 rows) in
    notebooks/04_feature_engineering.ipynb before generalizing to all 30 SKUs.
    """
    out = []
    for keys, g in daily.groupby(["sku", "channel", "region"]):
        d = g.set_index("date").sort_index()
        agg = d.resample("W-MON", label="left", closed="left").agg(
            price_avg=("price_unit", "mean"),
            promo_rate=("promotion_flag", "mean"),
            stock_avg=("stock_available", "mean"),
            deliveries=("delivery_days", "count"),
        )
        agg = agg.dropna(subset=["price_avg"])
        agg["sku"], agg["channel"], agg["region"] = keys
        out.append(agg.reset_index().rename(columns={"date": "week"}))
    return pd.concat(out, ignore_index=True)


def build_enrichment_table(weekly: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """Assemble every enrichment column at its correct grain and merge onto the weekly base table.

    `weekly` (weekly_df_final_for_modeling.csv) has no `category`/`brand` columns, since
    those are static SKU attributes that only live in the daily file — they're attached
    here first so category_trend can be computed. `pack_type` is deliberately excluded:
    unlike brand/segment/category, it varies within a single SKU (a SKU is sold in
    multiple pack types), so it isn't a 1:1 SKU attribute and can't be joined on `sku` alone.
    """
    sku_attrs = daily[["sku", "brand", "segment", "category"]].drop_duplicates()
    weekly = weekly.merge(sku_attrs, on="sku", how="left")

    weeks = pd.to_datetime(weekly["week"].unique())
    regions = weekly["region"].unique().tolist()

    internal = compute_internal_aggregates(daily)
    temp = generate_avg_temp(regions, weeks)
    inflation = generate_inflation_index(weeks)
    school = generate_school_in_session(weeks)
    event = generate_event_score(weeks)
    cat_trend = compute_category_trend(weekly)

    out = weekly.merge(internal, on=["sku", "channel", "region", "week"], how="left")
    out = out.merge(temp, on=["region", "week"], how="left")
    out = out.merge(inflation, on="week", how="left")
    out = out.merge(school, on="week", how="left")
    out = out.merge(event, on="week", how="left")
    out = out.merge(cat_trend, on=["category", "region", "week"], how="left")
    return out
