import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from models.zero_history import build_launch_records, zero_history_forecast, ZERO_HISTORY_FEATURES


def toy_records():
    return pd.DataFrame({"sku": ["OLD", "LATE", "NEW"],
        "week": pd.to_datetime(["2023-01-02", "2023-01-09", "2023-01-09"]),
        "label_available_at": pd.to_datetime(["2023-01-09", "2023-01-16", "2023-01-16"]),
        "category": ["Milk"] * 3, "segment": ["S"] * 3, "brand": ["B"] * 3,
        "channel": ["Retail"] * 3, "region": ["North"] * 3,
        "launch_units": [70., 999., 100.], "month": [1]*3, "year": [2023]*3, "week_number": [1,2,2]})


def test_model_inputs_exclude_sales_and_future_training(monkeypatch):
    import models.zero_history as zero
    seen = []
    def fake_fit(train, test, feature_cols):
        assert train.target_next_week.tolist() == [70.]
        assert set(test.columns) == set(ZERO_HISTORY_FEATURES)
        assert not {"units_sold", "launch_units", "observed_days", "price_unit", "promotion_flag", "sku"} & set(feature_cols)
        seen.append(test.copy())
        return np.array([80.])
    monkeypatch.setattr(zero, "fit_predict_lgb", fake_fit)
    records = toy_records()
    before = zero_history_forecast(records, ["NEW"])
    records.loc[records.sku == "NEW", "launch_units"] = 99999.
    records["price_unit"] = 99999.
    after = zero_history_forecast(records, ["NEW"])
    pd.testing.assert_frame_equal(seen[0], seen[1])
    assert before.pred_zero_lgb.tolist() == after.pred_zero_lgb.tolist() == [80.]
    assert before.pred_zero_analog.tolist() == after.pred_zero_analog.tolist() == [70.]
    assert before.analog_sku.tolist() == ["OLD"]
    assert before.train_rows.tolist() == [1]


def test_analog_reports_missing_channel_coverage(monkeypatch):
    import models.zero_history as zero
    monkeypatch.setattr(zero, "fit_predict_lgb", lambda *a, **kw: np.array([1.]))
    records = toy_records()
    records.loc[records.sku == "NEW", "channel"] = "Online"
    out = zero_history_forecast(records, ["NEW"])
    assert out.pred_zero_analog.isna().all()


def test_no_completed_launches_raises():
    records = toy_records()
    records.loc[records.sku == "OLD", "label_available_at"] = pd.Timestamp("2023-01-10")
    with pytest.raises(ValueError, match="no completed"):
        zero_history_forecast(records, ["NEW"])


def test_launch_label_is_current_partial_week_not_next_week():
    dates = pd.date_range("2023-01-04", "2023-01-22")
    daily = pd.DataFrame({"date": dates, "sku": "A", "channel": "Retail", "region": "North",
        "brand": "B", "category": "Milk", "segment": "S", "units_sold": 10,
        "delivered_qty": 20, "price_unit": 5., "stock_available": 30., "promotion_flag": 0, "delivery_days": 1})
    out = build_launch_records(daily)
    assert len(out) == 1
    assert out.launch_units.iloc[0] == 50.
    assert out.observed_days.iloc[0] == 5
    assert out.label_available_at.iloc[0] == pd.Timestamp("2023-01-09")
    daily.loc[daily.date >= "2023-01-09", "units_sold"] = 99999
    pd.testing.assert_frame_equal(out, build_launch_records(daily))


def test_all_held_out_skus_excluded(monkeypatch):
    import models.zero_history as zero
    seen = []
    def fake_fit(train, test, **kw):
        seen.append(train.target_next_week.tolist())
        return np.ones(len(test))
    monkeypatch.setattr(zero, "fit_predict_lgb", fake_fit)
    zero_history_forecast(toy_records(), ["LATE", "NEW"])
    assert seen == [[70.], [70.]]


def seven_day_daily():
    dates = pd.date_range("2023-01-04", "2023-01-22")
    return pd.DataFrame({"date": dates, "sku": "A", "channel": "Retail", "region": "North",
        "brand": "B", "category": "Milk", "segment": "S", "units_sold": 10})


def test_seven_days_cross_calendar_week_and_exclude_eighth_day():
    from models.zero_history import build_seven_day_launch_records
    daily = seven_day_daily()
    before = build_seven_day_launch_records(daily)
    assert before.launch_units.tolist() == [70]
    assert before.observed_days.tolist() == [7]
    assert before.week.iloc[0] == pd.Timestamp("2023-01-04")
    assert before.label_available_at.iloc[0] == pd.Timestamp("2023-01-11")
    daily.loc[daily.date >= "2023-01-11", "units_sold"] = 9999
    pd.testing.assert_frame_equal(before, build_seven_day_launch_records(daily))


def test_seven_days_does_not_invent_missing_records_or_reset_channel_launch():
    from models.zero_history import build_seven_day_launch_records
    daily = seven_day_daily()
    late = daily[daily.date >= "2023-01-08"].copy()
    late["channel"] = "Online"
    daily = daily[daily.date != "2023-01-06"]
    out = build_seven_day_launch_records(pd.concat([daily, late]))
    online = out[out.channel == "Online"].iloc[0]
    retail = out[out.channel == "Retail"].iloc[0]
    assert online.launch_units == 30 and online.observed_days == 3
    assert retail.launch_units == 60 and retail.observed_days == 6
    assert out.week.nunique() == 1
    assert set(out.channel) == {"Retail", "Online"}


def test_seven_days_excludes_right_censored_target():
    from models.zero_history import build_seven_day_launch_records
    daily = seven_day_daily().iloc[:6]
    assert build_seven_day_launch_records(daily).empty


def test_category_mean_uses_only_completed_other_launches(monkeypatch):
    import models.zero_history as zero
    monkeypatch.setattr(zero, "fit_predict_lgb", lambda *a, **kw: np.array([1.]))
    records = toy_records()
    extra = records.iloc[[0]].copy()
    extra.sku = "OLD2"
    extra.launch_units = 90.
    records = pd.concat([records, extra], ignore_index=True)
    out = zero_history_forecast(records, ["NEW"], include_category_baseline=True)
    assert out.pred_category_mean.tolist() == [80.]
    assert out.baseline_donors.tolist() == [2]
    assert out.baseline_level.tolist() == ["category_channel_region"]
    records.loc[records.sku == "NEW", "launch_units"] = 99999.
    changed = zero_history_forecast(records, ["NEW"], include_category_baseline=True)
    assert changed.pred_category_mean.tolist() == [80.]


def test_category_mean_fallback_is_recorded(monkeypatch):
    import models.zero_history as zero
    monkeypatch.setattr(zero, "fit_predict_lgb", lambda *a, **kw: np.array([1.]))
    records = toy_records()
    records.loc[records.sku == "NEW", ["category", "channel", "region"]] = ["Juice", "Online", "South"]
    out = zero_history_forecast(records, ["NEW"], include_category_baseline=True)
    assert out.pred_category_mean.tolist() == [70.]
    assert out.baseline_level.tolist() == ["global"]
    assert out.baseline_donors.tolist() == [1]


def test_seven_day_cutoff_accepts_only_completed_windows(monkeypatch):
    import models.zero_history as zero
    monkeypatch.setattr(zero, "fit_predict_lgb", lambda *a, **kw: np.array([1.]))
    records = toy_records()
    records.loc[records.sku == "NEW", "week"] = pd.Timestamp("2023-01-11")
    records.loc[records.sku == "OLD", "label_available_at"] = pd.Timestamp("2023-01-11")
    out = zero_history_forecast(records, ["NEW"], include_category_baseline=True)
    assert out.train_rows.tolist() == [1]
    assert out.train_label_available_max.tolist() == [pd.Timestamp("2023-01-11")]
