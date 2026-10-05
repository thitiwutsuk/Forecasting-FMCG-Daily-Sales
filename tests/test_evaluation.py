"""Phase 12 tests protect key alignment, fair cohorts and held-out data."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from models.evaluation import (KEYS, CORE, prediction_rows, summarize_predictions,
                               validate_ledger, recorded_table, load_recorded_results, run_cv)


def val(week="2024-01-01", actual=(10.0, 20.0)):
    return pd.DataFrame(dict(sku=["A", "B"], channel=["Retail"] * 2, region=["North"] * 2,
                             week=pd.to_datetime([week] * 2), target_next_week=actual), index=[30, 10])


def registry():
    return {m: dict(model=CORE[m][0], phase=str(CORE[m][1]), comparison_group="core_cv",
                    source_notebook="test.ipynb") for m in ("global_lgb", "global_xgb", "seasonal_naive")}


def ledger():
    pieces = []
    for fold, week, actual, lgb, xgb, seasonal in [
        (1, "2024-01-01", (10, 20), (10, 18), (9, 19), (np.nan, 16)),
        (2, "2024-01-08", (100, 200), (95, 199), (97, 195), (np.nan, 190)),
    ]:
        frame = val(week, actual)
        for m, pred in [("global_lgb", lgb), ("global_xgb", xgb), ("seasonal_naive", seasonal)]:
            pieces.append(prediction_rows(frame, pred, m, fold))
    return pd.concat(pieces, ignore_index=True)


def test_predictions_follow_keys_after_dl_style_reorder():
    frame = val()
    pred = frame[KEYS].assign(prediction=[11.0, 23.0]).iloc[::-1].reset_index(drop=True)
    result = prediction_rows(frame, pred, "global_lgb", 1).set_index("sku")
    assert result.loc["A", "prediction"] == 11
    assert result.loc["A", "actual"] == 10
    assert result.loc["B", "prediction"] == 23
    pd.testing.assert_frame_equal(frame, val())


@pytest.mark.parametrize("kind", ["duplicate", "missing", "extra", "infinity"])
def test_invalid_predictions_are_not_silently_dropped(kind):
    frame = val()
    pred = frame[KEYS].assign(prediction=[11.0, 23.0])
    if kind == "duplicate":
        pred = pd.concat([pred, pred.iloc[:1]])
    elif kind == "missing":
        pred = pred.iloc[:1]
    elif kind == "extra":
        pred.loc[30, "sku"] = "Extra"
    else:
        pred.loc[30, "prediction"] = np.inf
    with pytest.raises(ValueError):
        prediction_rows(frame, pred, "global_lgb", 1)


def test_macro_mean_sample_std_and_common_coverage_are_correct():
    summary, scores = summarize_predictions(ledger(), registry())
    own = summary[summary.comparison_group.eq("core_cv")].set_index("model_id")
    lgb_wapes = np.array([2 / 30, 6 / 300])
    assert own.loc["global_lgb", "WAPE"] == pytest.approx(lgb_wapes.mean())
    assert own.loc["global_lgb", "WAPE"] != pytest.approx(8 / 330)  # pooled WAPE is different
    assert own.loc["global_lgb", "WAPE_std"] == pytest.approx(lgb_wapes.std(ddof=1))
    assert own.loc["seasonal_naive", "coverage"] == 0.5
    assert own.loc["seasonal_naive", "n_scored"] == 2
    assert pd.isna(own.loc["seasonal_naive", "rank"])
    assert own.loc["global_lgb", "rank_group"] == own.loc["global_xgb", "rank_group"]
    common = summary[summary.comparison_group.eq("core_common")]
    assert common.cohort_hash.nunique() == 1
    assert common.n_scored.eq(2).all()
    assert common.coverage.eq(0.5).all()
    assert scores[scores.comparison_group.eq("core_common")].n_scored.eq(1).all()


def test_common_table_is_not_duplicated_when_coverage_matches():
    data = ledger()
    data.loc[data.prediction.isna(), "prediction"] = 10
    summary, _ = summarize_predictions(data, registry())
    assert not summary.comparison_group.eq("core_common").any()


def test_empty_common_fold_is_an_error():
    data = ledger()
    data.loc[data.model_id.eq("global_xgb") & data.sku.eq("B"), "prediction"] = np.nan
    with pytest.raises(ValueError, match="no common"):
        summarize_predictions(data, registry())


def test_missing_model_fold_is_an_error():
    data = ledger()
    data = data[~(data.model_id.eq("global_xgb") & data.fold_id.eq(2))]
    with pytest.raises(ValueError, match="missing model/fold"):
        summarize_predictions(data, registry())


def test_zero_actuals_leave_wape_undefined():
    data = ledger()
    data.loc[data.fold_id.eq(1), "actual"] = 0
    summary, _ = summarize_predictions(data, registry())
    assert summary.WAPE.isna().all()  # an undefined fold must not be silently skipped
    assert summary.WAPE_std.isna().all()
    assert summary["rank"].isna().all()


def test_ablation_delta_uses_reused_full_model_predictions():
    data = ledger()
    variant = data[data.model_id.eq("global_lgb")].copy()
    variant["model_id"] = "without_lag_rolling"
    variant["prediction"] += 10
    reg = registry()
    reg["without_lag_rolling"] = dict(model="Without lag", phase="11", comparison_group="ablation_cv", source_notebook="test.ipynb")
    summary, _ = summarize_predictions(pd.concat([data, variant]), reg)
    rows = summary[summary.comparison_group.eq("ablation_cv")].set_index("model_id")
    assert rows.loc["global_lgb", "delta_WAPE"] == 0
    assert rows.loc["without_lag_rolling", "delta_WAPE"] == pytest.approx(
        rows.loc["without_lag_rolling", "WAPE"] - rows.loc["global_lgb", "WAPE"])
    assert rows["rank"].isna().all()


def folds():
    return [SimpleNamespace(train_weeks=np.array(["2023-12-25"], dtype="datetime64[ns]"),
                            val_weeks=np.array([week], dtype="datetime64[ns]"))
            for week in ("2024-01-01", "2024-01-08")]


@pytest.mark.parametrize("kind", ["duplicate", "holdout", "overlap", "wrong_actual", "missing_model"])
def test_ledger_validation_rejects_bad_panels(kind):
    data = ledger()
    fs = folds()
    holdout = np.array(["2024-01-15"], dtype="datetime64[ns]")
    if kind == "duplicate":
        data = pd.concat([data, data.iloc[:1]])
    elif kind == "holdout":
        data.loc[0, "week"] = pd.Timestamp("2024-01-15")
    elif kind == "overlap":
        fs[0].train_weeks = fs[0].val_weeks
    elif kind == "wrong_actual":
        data.loc[0, "actual"] = 999
    else:
        data = data[data.model_id.ne("global_xgb")]
    with pytest.raises(ValueError):
        validate_ledger(data, fs, holdout, list(registry()))


def test_valid_ledger_is_accepted():
    validate_ledger(ledger(), folds(), np.array(["2024-01-15"], dtype="datetime64[ns]"), list(registry()))


def test_recorded_results_have_provenance_and_no_fabricated_metrics():
    recorded, sources = load_recorded_results(ROOT)
    assert len(recorded) == 8
    assert recorded.source.eq("recorded").all()
    assert len(sources) == 2 and all(len(s["sha256"]) == 64 for s in sources)
    assert all(s["recorded_values"] for s in sources)
    cold = recorded[recorded.phase.eq("10")]
    assert cold[["MAE", "RMSE", "SMAPE"]].isna().all().all()
    assert set(cold.comparison_group) == {"recorded_cold_0_7", "recorded_cold_8_plus"}


def test_missing_recorded_output_fails_instead_of_reading_readme(tmp_path):
    path = tmp_path / "missing.ipynb"
    path.write_text(json.dumps(dict(cells=[])), encoding="utf-8")
    with pytest.raises(ValueError, match="found 0"):
        recorded_table(path, ["model", "WAPE"])


def test_cv_fits_once_per_variant_and_dl_merge_preserves_keys(monkeypatch, tmp_path):
    from models import forecast
    from features import dl_embeddings
    from splits.walk_forward import WalkForwardSplitter

    frame = pd.read_csv(ROOT / "data/processed/weekly_features.csv", parse_dates=["week"])
    fs, holdout = WalkForwardSplitter(initial_train_end="2023-09-30", val_fold_weeks=8, final_test_weeks=10).get_folds(frame)
    counts = dict(lgb=0, xgb=0, rf=0, local=0, embeddings=0)

    def fake_fit(name):
        def fit(train, val, **kwargs):
            counts[name] += 1
            assert not train.week.isin(holdout).any()
            assert not val.week.isin(holdout).any()
            assert train.week.max() < val.week.min()
            return val.target_next_week.values + 1
        return fit

    for name, fn in [("lgb", "fit_predict_lgb"), ("xgb", "fit_predict_xgb"),
                     ("rf", "fit_predict_rf"), ("local", "fit_predict_local_per_sku")]:
        monkeypatch.setattr(forecast, fn, fake_fit(name))

    def fake_embeddings(train):
        counts["embeddings"] += 1
        assert not train.week.isin(holdout).any()
        assert set(train.week) == set(pd.to_datetime(fs[counts["embeddings"] - 1].train_weeks))
        return {}

    monkeypatch.setattr(dl_embeddings, "fit_fold_embeddings", fake_embeddings)
    monkeypatch.setattr(dl_embeddings, "transform_with_fold_embeddings", lambda df, tables: df.iloc[::-1].reset_index(drop=True))
    result, reg, _, _, _ = run_cv(ROOT, progress=lambda msg: None, checkpoint_dir=tmp_path)
    assert counts == dict(lgb=56, xgb=7, rf=7, local=7, embeddings=7)
    assert len(reg) == 14
    dl = result[result.model_id.eq("lgb_dl")]
    assert np.allclose(dl.prediction, dl.actual + 1)
    assert not result.week.isin(holdout).any()
    restored, restored_reg, _, _, _ = run_cv(ROOT, progress=lambda msg: None, checkpoint_dir=tmp_path)
    assert counts == dict(lgb=56, xgb=7, rf=7, local=7, embeddings=7)
    pd.testing.assert_frame_equal(restored, result, check_dtype=False, check_categorical=False)
    assert all(meta['restored_folds'] == list(range(1, 8)) for meta in restored_reg.values())
    checkpoint = next(tmp_path.glob('*/global_lgb_1.csv'))
    checkpoint.write_text('damaged', encoding='utf-8')
    with pytest.raises(ValueError, match='checksum mismatch'):
        run_cv(ROOT, progress=lambda msg: None, checkpoint_dir=tmp_path)
