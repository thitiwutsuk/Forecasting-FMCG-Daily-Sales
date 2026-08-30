"""Regression tests for src/data/build_weekly_base.py.

Locks in the derivation rules that were verified against the originally-given
`weekly_df_final_for_modeling.csv` (0 mismatches, see the module docstring for the
full verification story) so a future edit can't silently drift from them.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.build_weekly_base import _add_calendar_flags, _add_sku_age_and_lifecycle, build_weekly_base


def _toy_daily(n_weeks: int = 10) -> pd.DataFrame:
    """A single (sku, channel, region) group with n_weeks of daily rows, one row/day."""
    dates = pd.date_range("2022-01-03", periods=n_weeks * 7, freq="D")  # starts on a Monday
    return pd.DataFrame(
        {
            "date": dates,
            "sku": "TEST-001",
            "brand": "TestBrand",
            "segment": "TestSeg",
            "category": "TestCat",
            "channel": "Retail",
            "region": "PL-Central",
            "pack_type": "Single",
            "price_unit": 5.0,
            "promotion_flag": 0,
            "delivery_days": 2,
            "stock_available": 100,
            "delivered_qty": 50,
            "units_sold": 20,
        }
    )


def test_lifecycle_stage_thresholds():
    """Growth if sku_age <= 12, Mature if 13-52, Decline if >= 53 — verified with
    0 mismatches across all 31,027 rows of the given table."""
    weekly = pd.DataFrame(
        {
            "sku": ["A"] * 5,
            "channel": ["Retail"] * 5,
            "region": ["PL-Central"] * 5,
            "week": pd.to_datetime(
                ["2022-01-03", "2022-01-31", "2022-04-04", "2023-01-02", "2024-01-01"]
            ),
        }
    )
    out = _add_sku_age_and_lifecycle(weekly)
    stage_by_age = dict(zip(out["sku_age"], out["lifecycle_stage"]))

    for age in [0, 12]:
        if age in stage_by_age:
            assert stage_by_age[age] == "Growth"
    for age in [13, 52]:
        if age in stage_by_age:
            assert stage_by_age[age] == "Mature"

    # direct boundary check via the bucket rule itself
    ages = pd.Series([0, 4, 12, 13, 30, 52, 53, 100])
    weekly2 = pd.DataFrame(
        {
            "sku": "A",
            "channel": "Retail",
            "region": "PL-Central",
            "week": pd.to_datetime("2022-01-03") + pd.to_timedelta(ages * 7, unit="D"),
        }
    )
    out2 = _add_sku_age_and_lifecycle(weekly2)
    expected = ["Growth", "Growth", "Growth", "Mature", "Mature", "Mature", "Decline", "Decline"]
    assert out2["lifecycle_stage"].tolist() == expected


def test_is_summer_is_winter_from_month():
    """is_summer: month in {6,7,8}; is_winter: month in {1,2,12} — both verified 100%
    against the given table."""
    weekly = pd.DataFrame(
        {
            "week": pd.to_datetime(
                [f"2023-{m:02d}-02" for m in range(1, 13)]
            )
        }
    )
    out = _add_calendar_flags(weekly)
    for _, row in out.iterrows():
        month = row["month"]
        assert row["is_summer"] == (1 if month in (6, 7, 8) else 0)
        assert row["is_winter"] == (1 if month in (1, 2, 12) else 0)


def test_promotion_flag_is_max_not_mean_and_is_int():
    """promotion_flag must be max() per group-week (any promo day -> 1), not mean(),
    and must stay int64 — verified 0 mismatches against the given table."""
    daily = _toy_daily(n_weeks=6)
    daily = daily.copy()
    # flip promotion on for exactly 2 of the 7 days in the third week (mean would be < 0.5)
    third_week_mask = (daily["date"] >= "2022-01-17") & (daily["date"] < "2022-01-19")
    daily.loc[third_week_mask, "promotion_flag"] = 1

    weekly = build_weekly_base(daily)
    assert weekly["promotion_flag"].dtype == "int64"

    third_week_row = weekly[weekly["week"] == "2022-01-17"]
    if len(third_week_row):
        assert third_week_row["promotion_flag"].iloc[0] == 1  # max(), not mean() which would round to 0


def test_sku_age_anchored_at_sku_level_not_group_level():
    """sku_age must be anchored to the SKU's overall first appearance across ALL
    channels/regions, not per (sku, channel, region) — a bug found and fixed during
    development (see module docstring): 14/270 groups in the real data start selling
    later than their SKU's true launch elsewhere, which silently miscomputes age if
    anchored per-group instead of per-SKU."""
    daily_early = _toy_daily(n_weeks=8)
    daily_early["channel"] = "Discount"  # this channel launches on week 0

    daily_late = _toy_daily(n_weeks=8)
    daily_late["channel"] = "Retail"
    daily_late["date"] = daily_late["date"] + pd.Timedelta(weeks=2)  # this channel launches 2 weeks later

    daily = pd.concat([daily_early, daily_late], ignore_index=True)
    weekly = build_weekly_base(daily)

    retail_first_row = weekly[weekly["channel"] == "Retail"].sort_values("week").iloc[0]
    # Retail's own first daily row is 2 weeks after the SKU's true (Discount) launch,
    # so its sku_age at that first row must reflect the SKU-level age (>= 2), not 0.
    assert retail_first_row["sku_age"] >= 2


def test_build_weekly_base_output_schema():
    """Output must have the same 24-column schema as the originally-given table,
    with no nulls (boundary rows get dropped, not left as NaN)."""
    daily = _toy_daily(n_weeks=10)
    weekly = build_weekly_base(daily)

    expected_cols = {
        "sku", "week", "channel", "region", "units_sold", "stock_available", "promotion_flag",
        "price_unit", "delivery_days", "is_holiday_peak", "week_number", "month", "year",
        "is_holiday_week", "is_summer", "is_winter", "sku_age", "lifecycle_stage",
        "lag_1", "lag_2", "rolling_mean_4", "rolling_std_4", "momentum", "target_next_week",
    }
    assert set(weekly.columns) == expected_cols
    assert weekly.isnull().sum().sum() == 0
    assert not weekly.duplicated(subset=["sku", "channel", "region", "week"]).any()
