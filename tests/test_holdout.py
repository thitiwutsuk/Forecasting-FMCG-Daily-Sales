"""Phase 13 tests: truncated labels, scoring, the batch contract audit, and the
January panel's arrival-order and history guarantees (the last two on real data)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from models.holdout import (KEYS, audit_data_contract, build_january_panel, build_ledger,
                            cluster_bootstrap_wape_diff, load_january_batches, replay_arrival_order,
                            score_ledger, target_is_complete)


def test_target_is_complete_flags_partial_label_week():
    # Label week of origin 2024-12-23 is Mon 12-30 .. Sun 2025-01-05; raw data ends 12-31.
    origins = pd.to_datetime(["2024-12-09", "2024-12-16", "2024-12-23"])
    assert target_is_complete(origins, "2024-12-31").tolist() == [True, True, False]
    assert target_is_complete(origins, "2025-01-05").tolist() == [True, True, True]


def _ledger():
    rows = pd.DataFrame(dict(sku=["A", "A", "B", "B"], channel="Retail", region="North",
                             week=pd.to_datetime(["2024-01-01", "2024-01-08"] * 2),
                             target_next_week=[10.0, 20.0, 30.0, 40.0]))
    baselines = rows[KEYS].assign(pred_naive=[10.0, 10.0, 30.0, 30.0], pred_seasonal_naive=[np.nan, 20.0, np.nan, 40.0],
                                  pred_moving_avg_4=[12.0, 18.0, 33.0, 37.0])
    return build_ledger(rows, {"global_lgb": np.array([11.0, 19.0, 31.0, 39.0])}, baselines)


def test_build_ledger_pairs_by_keys_and_keeps_missing_baselines():
    ledger = _ledger()
    assert set(ledger.model_id) == {"naive", "seasonal_naive", "moving_avg_4", "global_lgb"}
    assert ledger.groupby("model_id").size().eq(4).all()
    assert ledger[ledger.model_id.eq("seasonal_naive")].prediction.isna().sum() == 2


def test_score_ledger_pools_rows_and_reports_bias_and_coverage():
    scores = score_ledger(_ledger()).set_index("model_id")
    assert scores.loc["global_lgb", "WAPE"] == pytest.approx(4 / 100)
    assert scores.loc["naive", "bias"] == pytest.approx((80 - 100) / 100)
    assert scores.loc["seasonal_naive", "coverage"] == 0.5
    assert scores.loc["seasonal_naive", "WAPE"] == 0.0


def test_cluster_bootstrap_ci_contains_observed_difference():
    out = cluster_bootstrap_wape_diff(_ledger(), "global_lgb", "naive", n_boot=200)
    assert out["diff_b_minus_a"] == pytest.approx(20 / 100 - 4 / 100)
    assert out["ci_low"] <= out["diff_b_minus_a"] <= out["ci_high"]
    assert out["n_series"] == 2


def _daily(dates, rows_per_day=1, units=10):
    frame = pd.DataFrame([dict(date=d, sku="S", brand="B", segment="G", category="C", channel="Retail",
                               region="North", pack_type="Single", price_unit=5.0, promotion_flag=0,
                               delivery_days=2, stock_available=100, delivered_qty=100, units_sold=units)
                          for d in pd.to_datetime(dates) for _ in range(rows_per_day)])
    return frame


def test_contract_audit_flags_extra_rows_per_day():
    history = _daily(pd.date_range("2024-01-01", "2024-03-31"))
    same = _daily(pd.date_range("2024-04-01", "2024-04-14"))
    tripled = _daily(pd.date_range("2024-04-01", "2024-04-14"), rows_per_day=3)
    ok = audit_data_contract(history, same, "S").set_index("check")
    bad = audit_data_contract(history, tripled, "S").set_index("check")
    assert ok.loc["duplicate series-day rows", "consistent"]
    assert not bad.loc["duplicate series-day rows", "consistent"]
    assert not bad.loc["weekly units per series (full weeks)", "consistent"]


DATA_FILES = [ROOT / "data/interim/daily_validated.csv", ROOT / "data/processed/weekly_features.csv"]


@pytest.fixture(scope="module")
def real_inputs():
    if not all(p.exists() for p in DATA_FILES) or not list((ROOT / "data/raw").glob("batch_*.parquet")):
        pytest.skip("project data not available")
    daily = pd.read_csv(DATA_FILES[0], parse_dates=["date"])
    weekly = pd.read_csv(DATA_FILES[1], parse_dates=["week"])
    batches = load_january_batches(ROOT / "data/raw")
    panel, check = build_january_panel(daily, batches, weekly)
    return daily, weekly, batches, panel, check


def test_january_panel_reproduces_history_exactly(real_inputs):
    *_, check = real_inputs
    assert check.mismatches.sum() == 0
    assert check.rows_compared.min() > 1000


def test_january_panel_completes_truncated_label_and_has_no_future_labels(real_inputs):
    daily, weekly, batches, panel, _ = real_inputs
    last = panel[panel.week.eq("2024-12-23")]
    old = weekly[weekly.sku.eq("MI-006") & weekly.week.eq("2024-12-23")]
    assert (last.target_next_week.sum() > old.target_next_week.sum())
    assert (panel.label_week + pd.Timedelta(days=6)).max() <= batches.date.max()
    assert not panel.duplicated(KEYS).any() and panel.target_next_week.notna().all()


def test_january_features_do_not_depend_on_later_batches(real_inputs):
    daily, weekly, batches, panel, _ = real_inputs
    replay = replay_arrival_order(daily, batches, weekly, panel)
    assert replay.feature_mismatches.eq(0).all()
    assert replay.rows.iloc[-1] == len(panel)
