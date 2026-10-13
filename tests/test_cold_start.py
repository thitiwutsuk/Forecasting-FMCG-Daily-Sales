"""Regression tests for src/models/cold_start.py's analog-matching temporal-eligibility guard.

find_analog_sku() used to match a cold-start SKU to the closest existing SKU by
category/segment/pack_type/price alone, with no check that the candidate actually
existed yet at the target's launch time. A real cold-start forecast can only ever
borrow from products that already existed then, so a closer-matching SKU that
launched *later* is not a valid analog no matter how similar it looks -- it must be
excluded from the candidate pool before similarity matching runs, not just lose a
tiebreak to it.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models.cold_start import find_analog_sku


def _toy_sku_attrs() -> pd.DataFrame:
    # TARGET launches 2023-01-01.
    # CLOSE-LATE: exact category/segment/pack_type/price match, but launches AFTER
    # TARGET -- the closest possible match, and must still be excluded.
    # CLOSE-EARLY: same exact match, launches BEFORE TARGET -- the correct analog.
    # FAR-EARLY: only matches on category (coarsest tier), launches BEFORE TARGET --
    # eligible, but should lose to CLOSE-EARLY whenever both are in the pool.
    return pd.DataFrame(
        {
            "category": ["Milk", "Milk", "Milk", "Milk"],
            "segment": ["Standard", "Standard", "Standard", "Premium"],
            "pack_type": ["Bottle", "Bottle", "Bottle", "Carton"],
            "price_unit": [5.0, 5.0, 5.0, 8.0],
            "first_seen": pd.to_datetime(["2023-01-01", "2023-06-01", "2022-06-01", "2022-01-01"]),
        },
        index=["TARGET", "CLOSE-LATE", "CLOSE-EARLY", "FAR-EARLY"],
    )


def test_find_analog_sku_excludes_a_later_launched_closer_match():
    sku_attrs = _toy_sku_attrs()
    analog = find_analog_sku("TARGET", sku_attrs, ["CLOSE-LATE", "FAR-EARLY"])
    assert analog == "FAR-EARLY"


def test_find_analog_sku_selects_an_earlier_eligible_sku():
    # CLOSE-LATE is excluded despite being an exact match; among the two temporally
    # eligible candidates, the better match (CLOSE-EARLY) must still win the tiebreak,
    # confirming the eligibility filter doesn't just fall back to "whatever's left."
    sku_attrs = _toy_sku_attrs()
    analog = find_analog_sku("TARGET", sku_attrs, ["CLOSE-LATE", "CLOSE-EARLY", "FAR-EARLY"])
    assert analog == "CLOSE-EARLY"


def test_find_analog_sku_raises_when_no_candidate_is_temporally_eligible():
    sku_attrs = _toy_sku_attrs()
    with pytest.raises(ValueError, match="temporally-eligible"):
        find_analog_sku("TARGET", sku_attrs, ["CLOSE-LATE"])


def test_phase10_evaluation_skus_have_valid_temporal_analog_mappings():
    # Regression check on the actual project data: the 5 SKUs used as the Phase 10
    # cold-start evaluation set (notebooks/10_cold_start.ipynb) are the 5 latest-launching
    # SKUs in the whole dataset, so every other SKU should already be a temporally
    # eligible candidate for all of them -- this must keep holding even if the raw
    # data changes.
    data_path = Path(__file__).resolve().parents[1] / "data" / "raw" / "FMCG_2022_2024.csv"
    daily = pd.read_csv(data_path, parse_dates=["date"])
    sku_attrs = daily.groupby("sku").agg(
        pack_type=("pack_type", "first"),
        segment=("segment", "first"),
        category=("category", "first"),
        price_unit=("price_unit", "mean"),
        first_seen=("date", "min"),
    )

    cold_start_skus = ["YO-024", "MI-008", "SN-028", "YO-018", "SN-030"]
    mature_skus = [s for s in sku_attrs.index if s not in cold_start_skus]

    analogs = {sku: find_analog_sku(sku, sku_attrs, mature_skus) for sku in cold_start_skus}

    for target_sku, analog_sku in analogs.items():
        assert sku_attrs.loc[analog_sku, "first_seen"] < sku_attrs.loc[target_sku, "first_seen"]


def test_training_cutoff_requires_completed_target_week():
    from models.cold_start import training_before_launch
    df = pd.DataFrame({"sku": ["OLD", "OLD", "OLD", "NEW"],
                       "week": pd.to_datetime(["2023-01-02", "2023-01-09", "2023-01-16", "2023-01-02"])})
    train = training_before_launch(df, ["NEW"], pd.Timestamp("2023-01-16"))
    assert train.index.tolist() == [0]


@pytest.mark.parametrize("function_name", ["meta_learner_forecast", "mature_model_forecast"])
def test_models_use_individual_launch_cutoffs(monkeypatch, function_name):
    import models.cold_start as cold
    df = pd.DataFrame({"sku": ["OLD", "OLD", "OLD", "A", "B"],
        "week": pd.to_datetime(["2023-01-02", "2023-01-09", "2023-03-06", "2023-02-13", "2023-02-20"]),
        "sku_age": [0, 1, 9, 4, 4]})
    calls = []
    def fake_fit(train, test, feature_cols):
        cutoff = cold.launch_week(df, str(df.loc[test.index[0], "sku"]))
        assert (pd.to_datetime(train.week) + pd.Timedelta(days=14) <= cutoff).all()
        assert not train.sku.isin(["A", "B"]).any()
        calls.append(len(train))
        return [1.0] * len(test)
    monkeypatch.setattr(cold, "fit_predict_lgb", fake_fit)
    monkeypatch.setattr(cold, "NON_LAG_FEATURE_COLS", ["sku", "sku_age"])
    monkeypatch.setattr(cold, "ALL_FEATURE_COLS", ["sku", "sku_age"])
    result = getattr(cold, function_name)(df, ["A", "B"])
    assert calls == [1, 2]
    assert result.index.tolist() == [3, 4]


def test_launch_week_rejects_inconsistent_age():
    from models.cold_start import launch_week
    df = pd.DataFrame({"sku": ["A", "A"], "week": pd.to_datetime(["2023-02-06", "2023-02-13"]), "sku_age": [4, 4]})
    with pytest.raises(ValueError, match="inconsistent"):
        launch_week(df, "A")


def test_analog_uses_next_week_label_and_only_prelaunch_trajectory():
    from models.cold_start import analog_forecast
    df = pd.DataFrame({"sku": ["OLD", "OLD", "NEW", "NEW"],
        "week": pd.to_datetime(["2023-01-02", "2023-01-09", "2023-01-23", "2023-01-30"]),
        "sku_age": [0, 1, 0, 1], "channel": ["Retail"] * 4, "region": ["North"] * 4,
        "units_sold": [10, 20, 30, 40], "target_next_week": [20, 99, 40, 50]})
    attrs = pd.DataFrame({"category": ["Milk", "Milk"], "segment": ["S", "S"],
        "price_unit": [5., 5.], "first_seen": pd.to_datetime(["2023-01-02", "2023-01-23"])}, index=["OLD", "NEW"])
    result = analog_forecast(df, ["NEW"], attrs)
    assert result.pred_analog.tolist() == [20, 99]
    df.loc[1, "week"] = pd.Timestamp("2023-01-16")
    result = analog_forecast(df, ["NEW"], attrs)
    assert pd.isna(result.pred_analog.iloc[1])


def test_launch_attributes_ignore_future_prices_and_pack_types():
    from models.cold_start import launch_analog_attributes
    daily = pd.DataFrame({"sku": ["OLD", "OLD", "NEW", "NEW"],
        "date": pd.to_datetime(["2023-01-02", "2023-03-01", "2023-02-01", "2023-03-01"]),
        "category": ["Milk"] * 4, "segment": ["S"] * 4, "price_unit": [5., 999., 6., 999.]})
    attrs = launch_analog_attributes(daily, "NEW", pd.Timestamp("2023-01-30"))
    assert attrs.loc["OLD", "price_unit"] == 5.
    assert attrs.loc["NEW", "price_unit"] == 6.
    assert "pack_type" not in attrs.columns
