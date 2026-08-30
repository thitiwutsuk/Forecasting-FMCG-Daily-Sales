"""Build the weekly base modeling table from raw daily data (self-built replacement
for the originally-given `data/raw/weekly_df_final_for_modeling.csv`).

Every derivation rule below was verified against the given table before being used
here (see `notebooks/04_feature_engineering.ipynb`, Section 0), except
`is_holiday_week`/`is_holiday_peak`: the given table's values for those two columns
follow no reconstructible rule (several offset/window hypotheses against a Polish
holiday calendar were tested and none reached more than ~95% agreement, with mismatch
patterns that don't fit a simple date-shift). Rather than chase an apparently-buggy
original, this module defines them from a correct Polish public-holiday calendar —
the same choice already made for `avg_temp`/`inflation_index` in `src/features/enrich.py`
when that prototype turned out to be unrecoverable.

Verified 1:1 against the given table (0 mismatches across all 31,027 rows unless noted):
- promotion_flag: max() per group-week, not mean() (confirmed via direct comparison)
- lifecycle_stage: Growth if sku_age <= 12, Mature if 13-52, Decline if >= 53
- is_summer: month in {6, 7, 8}; is_winter: month in {1, 2, 12}
- momentum: lag_1 - lag_2
- lag_1, lag_2, rolling_mean_4, rolling_std_4: shift-based, matching the same
  formulas `src/data/validate.py::check_lag_and_rolling_leakage` already validates
  against the given table
"""

import numpy as np
import pandas as pd

GROUP_KEYS = ["sku", "channel", "region"]

# Polish Easter Sunday for the years covered by the dataset (2022-2024) — matches
# src/features/enrich.py's EASTER_SUNDAY so both modules agree on the same calendar.
EASTER_SUNDAY = {2022: "2022-04-17", 2023: "2023-04-09", 2024: "2024-03-31"}

# Fixed-date Polish public holidays, with a flag for whether each is a "peak" retail
# holiday (own definition: the holidays with the largest FMCG demand swing — Christmas
# and New Year) versus a minor observance that still counts as a holiday week.
FIXED_HOLIDAYS = {
    "01-01": "peak",  # New Year's Day
    "01-06": "minor",  # Epiphany
    "05-01": "minor",  # Labour Day
    "05-03": "minor",  # Constitution Day
    "08-15": "minor",  # Assumption of Mary
    "11-01": "minor",  # All Saints' Day
    "11-11": "minor",  # Independence Day
    "12-24": "peak",  # Christmas Eve
    "12-25": "peak",  # Christmas Day
    "12-26": "peak",  # Second Day of Christmas
}
EASTER_OFFSET_KIND = {0: "peak", 1: "peak", 60: "minor"}  # Easter Sunday, Easter Monday, Corpus Christi


def _week_agg(daily: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily -> weekly per (sku, channel, region), same convention verified
    against the given table: W-MON, label='left', closed='left'."""
    rows = []
    for keys, g in daily.groupby(GROUP_KEYS):
        d = g.set_index("date").sort_index()
        agg = d.resample("W-MON", label="left", closed="left").agg(
            units_sold=("units_sold", "sum"),
            delivered_qty=("delivered_qty", "sum"),
            price_unit=("price_unit", "mean"),
            stock_available=("stock_available", "mean"),
            promotion_flag=("promotion_flag", "max"),
            delivery_days=("delivery_days", "mean"),
        )
        agg["promotion_flag"] = agg["promotion_flag"].astype("int64")
        agg["sku"], agg["channel"], agg["region"] = keys
        rows.append(agg.reset_index().rename(columns={"date": "week"}))
    return pd.concat(rows, ignore_index=True)


def _add_sku_age_and_lifecycle(weekly: pd.DataFrame) -> pd.DataFrame:
    """sku_age: 0-indexed weeks since this SKU's overall first observed week, across
    ALL channels/regions (0 at the launch week itself) — verified against the given
    table. Anchored at the SKU level, not per (sku, channel, region): a product's age
    is how long it's existed since its first-ever sale anywhere, not how long any one
    channel/region has carried it (14/270 groups start selling a week or more after
    their SKU's true launch elsewhere, which would silently miscompute age if anchored
    per-group instead). lifecycle_stage: deterministic bucket of sku_age."""
    weekly = weekly.sort_values(GROUP_KEYS + ["week"]).copy()
    first_week = weekly.groupby("sku")["week"].transform("min")
    weekly["sku_age"] = (weekly["week"] - first_week).dt.days // 7

    def bucket(age: int) -> str:
        if age <= 12:
            return "Growth"
        if age <= 52:
            return "Mature"
        return "Decline"

    weekly["lifecycle_stage"] = weekly["sku_age"].apply(bucket)
    return weekly


def _add_calendar_flags(weekly: pd.DataFrame) -> pd.DataFrame:
    """week_number/month/year from the week timestamp; is_summer/is_winter from month
    (both verified 100% against the given table)."""
    weekly = weekly.copy()
    weekly["week_number"] = weekly["week"].dt.isocalendar().week.astype("int64")
    weekly["month"] = weekly["week"].dt.month
    weekly["year"] = weekly["week"].dt.year
    weekly["is_summer"] = weekly["month"].isin([6, 7, 8]).astype("int64")
    weekly["is_winter"] = weekly["month"].isin([1, 2, 12]).astype("int64")
    return weekly


def _polish_holidays(years: list) -> dict:
    """Map each holiday date -> 'peak' or 'minor', for the given years."""
    holidays = {}
    for year in years:
        for mmdd, kind in FIXED_HOLIDAYS.items():
            holidays[pd.Timestamp(f"{year}-{mmdd}")] = kind
        if year in EASTER_SUNDAY:
            easter = pd.Timestamp(EASTER_SUNDAY[year])
            for offset, kind in EASTER_OFFSET_KIND.items():
                holidays[easter + pd.Timedelta(days=offset)] = kind
    return holidays


def _add_holiday_flags(weekly: pd.DataFrame) -> pd.DataFrame:
    """is_holiday_week / is_holiday_peak from a correct Polish public-holiday calendar
    (own definition — see module docstring for why this doesn't try to match the
    given table's values)."""
    weekly = weekly.copy()
    years = sorted(weekly["week"].dt.year.unique().tolist())
    holidays = _polish_holidays(years)

    def week_flags(week_start: pd.Timestamp) -> tuple:
        week_end = week_start + pd.Timedelta(days=6)
        kinds_in_week = [kind for date, kind in holidays.items() if week_start <= date <= week_end]
        is_holiday_week = int(len(kinds_in_week) > 0)
        is_holiday_peak = int("peak" in kinds_in_week)
        return is_holiday_week, is_holiday_peak

    unique_weeks = weekly[["week"]].drop_duplicates()
    unique_weeks[["is_holiday_week", "is_holiday_peak"]] = unique_weeks["week"].apply(
        lambda w: pd.Series(week_flags(w))
    )
    return weekly.merge(unique_weeks, on="week", how="left")


def _add_lag_rolling_target(weekly: pd.DataFrame) -> pd.DataFrame:
    """lag_1, lag_2, rolling_mean_4, rolling_std_4, momentum, target_next_week —
    identical formulas to the ones already validated against the given table by
    src/data/validate.py::check_lag_and_rolling_leakage."""
    weekly = weekly.sort_values(GROUP_KEYS + ["week"]).copy()
    g = weekly.groupby(GROUP_KEYS)["units_sold"]

    weekly["lag_1"] = g.shift(1)
    weekly["lag_2"] = g.shift(2)
    weekly["_shift1"] = weekly["lag_1"]
    weekly["rolling_mean_4"] = weekly.groupby(GROUP_KEYS)["_shift1"].rolling(4).mean().reset_index(level=GROUP_KEYS, drop=True)
    weekly["rolling_std_4"] = weekly.groupby(GROUP_KEYS)["_shift1"].rolling(4).std().reset_index(level=GROUP_KEYS, drop=True)
    weekly = weekly.drop(columns="_shift1")
    weekly["momentum"] = weekly["lag_1"] - weekly["lag_2"]
    weekly["target_next_week"] = weekly.groupby(GROUP_KEYS)["units_sold"].shift(-1)
    return weekly


BASE_COLUMNS = [
    "sku", "week", "channel", "region", "units_sold", "stock_available", "promotion_flag",
    "price_unit", "delivery_days", "is_holiday_peak", "week_number", "month", "year",
    "is_holiday_week", "is_summer", "is_winter", "sku_age", "lifecycle_stage",
    "lag_1", "lag_2", "rolling_mean_4", "rolling_std_4", "momentum", "target_next_week",
]


def build_weekly_base(daily: pd.DataFrame) -> pd.DataFrame:
    """Build the weekly base modeling table from raw daily data.

    Reproduces the schema of the originally-given `weekly_df_final_for_modeling.csv`
    (same 24 columns) so it slots into the existing Phase 4-11 pipeline (`enrich.py`,
    `engineer.py`, `walk_forward.py`, `models/*`, `causal/*`) unchanged.

    Drops leading/trailing rows per group where lag/rolling/target can't be fully
    computed, matching the given table's boundary-trim convention.
    """
    daily = daily.copy()
    daily["date"] = pd.to_datetime(daily["date"])

    weekly = _week_agg(daily)
    weekly = _add_sku_age_and_lifecycle(weekly)
    weekly = _add_calendar_flags(weekly)
    weekly = _add_holiday_flags(weekly)
    weekly = _add_lag_rolling_target(weekly)

    weekly = weekly.dropna(
        subset=["lag_1", "lag_2", "rolling_mean_4", "rolling_std_4", "momentum", "target_next_week"]
    ).reset_index(drop=True)

    return weekly[BASE_COLUMNS]
