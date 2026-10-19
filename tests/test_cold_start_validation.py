import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from models.cold_start_validation import chronological_extension, sku_bootstrap_differences, summarize_predictions


def records():
    dates = pd.to_datetime(["2023-01-01", "2023-01-08", "2023-01-15", "2023-01-22"])
    return pd.DataFrame({"sku": ["A", "B", "C", "RESERVED"], "week": dates,
        "label_available_at": dates + pd.Timedelta(days=7), "launch_units": [10,20,30,40]})


def test_selection_is_temporal_excludes_reserved_and_ignores_outcomes(monkeypatch):
    import models.cold_start_validation as validation
    seen = []
    def fake_forecast(history, held_out_skus, **kwargs):
        assert "RESERVED" not in history.sku.tolist()
        seen.append(held_out_skus)
        return history[history.sku.isin(held_out_skus)].copy()
    monkeypatch.setattr(validation, "zero_history_forecast", fake_forecast)
    before, audit = chronological_extension(records(), ["RESERVED"], min_train_skus=2)
    assert before.sku.tolist() == ["C"]
    assert audit.set_index("sku").loc["B", "status"] == "insufficient_prior_launches"
    assert audit.set_index("sku").loc["RESERVED", "status"] == "reserved_existing_test"
    changed = records()
    changed.launch_units = [999,888,777,666]
    after, audit_after = chronological_extension(changed, ["RESERVED"], min_train_skus=2)
    pd.testing.assert_frame_equal(audit, audit_after)
    assert seen == [["C"], ["C"]]


def test_sku_bootstrap_keeps_paired_comparison_and_is_reproducible():
    p = pd.DataFrame({"sku": ["A","A","B","B"], "launch_units": [10.,20.,30.,40.],
        "pred_category_mean": [10.,20.,30.,40.], "pred_zero_analog": [9.,18.,27.,36.],
        "pred_zero_lgb": [8.,16.,24.,32.]})
    result = sku_bootstrap_differences(p, repetitions=100, seed=7)
    pd.testing.assert_frame_equal(result, sku_bootstrap_differences(p, repetitions=100, seed=7))
    np.testing.assert_allclose(result.wape_difference, [-.1,-.2])
    np.testing.assert_allclose(result.ci95_low, [-.1,-.2])
    np.testing.assert_allclose(result.ci95_high, [-.1,-.2])


def test_summaries_compare_same_rows_and_report_coverage():
    p = pd.DataFrame({"sku": ["A","B"], "launch_units": [10.,20.],
        "pred_category_mean": [10.,20.], "pred_zero_analog": [9.,np.nan], "pred_zero_lgb": [8.,16.]})
    scores, per_sku = summarize_predictions(p)
    assert scores.n.tolist() == [1,1,1]
    assert scores.coverage.tolist() == [.5,.5,.5]
    assert per_sku[per_sku.sku == "B"].n.eq(0).all()


def test_bootstrap_rejects_single_sku():
    p = pd.DataFrame({"sku":["A"],"launch_units":[10.],"pred_category_mean":[10.],
        "pred_zero_analog":[9.],"pred_zero_lgb":[8.]})
    with pytest.raises(ValueError, match="two SKUs"):
        sku_bootstrap_differences(p)
