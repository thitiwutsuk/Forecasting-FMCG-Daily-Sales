"""Regression tests for src/causal/promotion_robustness.py's panel builder.

Locks in the validation preconditions the heterogeneity-robust DiD estimators
(LITERATURE_GROUNDING.md Sec3a) need before they'll run at all: one row per
(group_id, time_id), a binary treatment, and no internal time gaps within a
group's observed window.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from causal.promotion_robustness import (
    build_group_panel,
    build_promotion_lead_lag_panel,
    compute_twfe_weight_diagnostic,
    fit_promotion_distributed_lag,
)


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
    assert set(["group_id", "group_label", "cluster_id", "time_id", "log_units"]) <= set(panel.columns)
    assert (panel["group_label"] == panel["sku"] + "|" + panel["channel"] + "|" + panel["region"]).all()
    assert pd.api.types.is_integer_dtype(panel["group_id"])
    assert pd.api.types.is_integer_dtype(panel["cluster_id"])
    assert panel["group_id"].min() == 1
    assert panel["cluster_id"].min() == 1


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


def test_twfe_weights_sum_to_one_and_reconstruct_coefficient():
    rows = []
    treatments = {
        "A": [0, 1, 0, 1],
        "B": [0, 0, 1, 1],
        "C": [1, 0, 1, 0],
        "D": [1, 1, 0, 0],
    }
    prices = {
        "A": [3.0, 3.2, 3.1, 3.5],
        "B": [3.4, 3.1, 3.6, 3.2],
        "C": [2.8, 3.3, 3.0, 3.4],
        "D": [3.6, 3.0, 3.5, 3.1],
    }
    for group_index, (sku, path) in enumerate(treatments.items()):
        for time_index, treatment in enumerate(path):
            price = prices[sku][time_index]
            log_units = 1.0 + 0.3 * group_index + 0.1 * time_index + 0.4 * treatment + 0.05 * price
            rows.append(
                {
                    "sku": sku,
                    "channel": "Retail",
                    "region": "North",
                    "week": pd.Timestamp("2022-01-03") + pd.Timedelta(weeks=time_index),
                    "units_sold": 0.0,  # replaced below after exponentiating
                    "promotion_flag": treatment,
                    "price_unit": price,
                    "_log_units": log_units,
                }
            )
    df = pd.DataFrame(rows)
    df["units_sold"] = np.expm1(df["_log_units"])
    panel = build_group_panel(df)

    summary, detail = compute_twfe_weight_diagnostic(panel)

    assert summary["sum_weights"] == pytest.approx(1.0)
    assert summary["reconstructed_twfe_coef"] == pytest.approx(0.4)
    assert detail.loc[detail.promotion_flag.eq(0), "twfe_weight"].eq(0).all()


def test_leads_and_lags_stay_within_group_boundaries():
    weeks = pd.date_range("2022-01-03", periods=6, freq="7D")
    df = pd.DataFrame(
        {
            "sku": ["A"] * 6 + ["B"] * 6,
            "channel": ["Retail"] * 12,
            "region": ["North"] * 12,
            "week": list(weeks) * 2,
            "units_sold": np.arange(12, dtype=float) + 10,
            "promotion_flag": [0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0],
            "price_unit": [3.0] * 12,
        }
    )
    panel = build_group_panel(df)

    shifted, terms = build_promotion_lead_lag_panel(panel, lags=2, leads=1)

    assert terms == ["promo_current", "promo_lag_1", "promo_lag_2", "promo_lead_1"]
    assert shifted.groupby("group_id").size().eq(3).all()
    a = shifted.loc[shifted.sku.eq("A")].sort_values("time_id")
    assert a["promo_current"].tolist() == [0.0, 0.0, 1.0]
    assert a["promo_lag_1"].tolist() == [1.0, 0.0, 0.0]
    assert a["promo_lag_2"].tolist() == [0.0, 1.0, 0.0]
    assert a["promo_lead_1"].tolist() == [0.0, 1.0, 1.0]


def test_distributed_lag_fit_recovers_known_current_and_lag_effects():
    rng = np.random.default_rng(42)
    rows = []
    weeks = pd.date_range("2022-01-03", periods=20, freq="7D")
    for group_index in range(10):
        treatment = rng.integers(0, 2, size=len(weeks))
        for time_index, week in enumerate(weeks):
            lag_effect = 0.1 * treatment[time_index - 1] if time_index > 0 else 0.0
            log_units = 2.0 + 0.05 * group_index + 0.02 * time_index + 0.4 * treatment[time_index] + lag_effect
            rows.append(
                {
                    "sku": f"S{group_index:02d}", "channel": "Retail", "region": "North",
                    "week": week, "units_sold": np.expm1(log_units),
                    "promotion_flag": treatment[time_index], "price_unit": 3.0,
                }
            )
    panel = build_group_panel(pd.DataFrame(rows))

    result = fit_promotion_distributed_lag(panel, lags=1, leads=1)

    assert result.params["promo_current"] == pytest.approx(0.4)
    assert result.params["promo_lag_1"] == pytest.approx(0.1)
    assert result.params["promo_lead_1"] == pytest.approx(0.0, abs=1e-10)
