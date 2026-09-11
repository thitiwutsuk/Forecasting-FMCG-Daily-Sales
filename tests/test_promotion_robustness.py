"""Regression tests for src/causal/promotion_robustness.py's panel builder.

Locks in the validation preconditions the heterogeneity-robust DiD estimators
(LITERATURE_GROUNDING.md Sec3a) need before they'll run at all: one row per
(group_id, time_id), a binary treatment, and no internal time gaps within a
group's observed window.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from causal.promotion_robustness import build_group_panel


def _toy_df(**overrides) -> pd.DataFrame:
    base = pd.DataFrame(
        {
            "sku": ["A", "A", "A", "B", "B"],
            "channel": ["Retail"] * 5,
            "region": ["North"] * 5,
            "week": pd.to_datetime(["2022-01-03", "2022-01-10", "2022-01-17", "2022-01-10", "2022-01-17"]),
            "units_sold": [10, 12, 9, 20, 18],
            "promotion_flag": [0, 1, 0, 1, 0],
            "price_unit": [5.0, 5.0, 5.1, 3.0, 3.0],
        }
    )
    for k, v in overrides.items():
        base[k] = v
    return base


def test_build_group_panel_adds_expected_columns():
    panel = build_group_panel(_toy_df())
    assert set(["group_id", "time_id", "log_units"]) <= set(panel.columns)
    assert (panel["group_id"] == panel["sku"] + "|" + panel["channel"] + "|" + panel["region"]).all()


def test_build_group_panel_time_id_is_sequential_across_all_groups():
    panel = build_group_panel(_toy_df())
    # "A" starts at week 2022-01-03 (time_id 0); "B" starts at 2022-01-10 (time_id 1) --
    # time_id is a shared calendar index, not reset per group.
    a_ids = sorted(panel.loc[panel.sku == "A", "time_id"])
    b_ids = sorted(panel.loc[panel.sku == "B", "time_id"])
    assert a_ids == [0, 1, 2]
    assert b_ids == [1, 2]


def test_build_group_panel_rejects_duplicate_group_time_rows():
    df = _toy_df()
    dupe_row = df.iloc[[1]].copy()
    df = pd.concat([df, dupe_row], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        build_group_panel(df)


def test_build_group_panel_rejects_non_binary_treatment():
    df = _toy_df(promotion_flag=[0, 1, 2, 1, 0])
    with pytest.raises(ValueError, match="binary"):
        build_group_panel(df)


def test_build_group_panel_rejects_internal_time_gap():
    # "A" observed at weeks 0 and 2 but not 1 -- a hole inside its own window.
    df = _toy_df()
    df = df[~((df.sku == "A") & (df.week == pd.Timestamp("2022-01-10")))]
    with pytest.raises(ValueError, match="gap"):
        build_group_panel(df)
