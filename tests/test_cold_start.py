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
