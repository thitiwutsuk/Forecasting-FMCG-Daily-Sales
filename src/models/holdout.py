"""Phase 13 future holdout backtest.

Two protocols, kept separate because they answer different questions:

Part A — final holdout (pre-registered in Phase 5). Every model is fitted ONCE on all
CV origin weeks and scored once on the 10 reserved origin weeks (all 30 SKUs). Nothing
here is tuned on the holdout. Origin 2024-12-23 is scored separately: its label is the
week of 2024-12-30, but the raw file ends on 2024-12-31, so that label holds 2 of 7 days.

Part B — January 2025 batches (MI-006 only). The four weekly batch files are audited
against the 2022-2024 data contract first, then appended to the raw daily history,
rebuilt with the same Phase 4 code, and scored as one-step-ahead forecasts from a model
refitted on every complete 2022-2024 origin week. The model is frozen during January.

Features that depend on OTHER SKUs (category_trend, price_index's category average
price) can't be recomputed from a MI-006-only batch, so their last complete-week value
is carried forward. avg_temp/inflation_index/event_score are regenerated with the same
deterministic Phase 4 generators over the extended week range.
"""

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import score_all

KEYS = ["sku", "channel", "region", "week"]
GROUP_KEYS = ["sku", "channel", "region"]
METRICS = ["MAE", "RMSE", "WAPE", "SMAPE"]
SPLIT_CONFIG = dict(initial_train_end="2023-09-30", val_fold_weeks=8, final_test_weeks=10)
PRIMARY = "global_lgb"
MODELS = {
    "naive": "Naive",
    "seasonal_naive": "Seasonal Naive",
    "moving_avg_4": "Moving Average (4w)",
    "global_lgb": "Global LightGBM",
    "global_xgb": "Global XGBoost",
    "global_cb": "Global CatBoost",
    "global_rf": "Global Random Forest",
}
ML_MODELS = ["global_lgb", "global_xgb", "global_cb", "global_rf"]
BASELINE_COLS = {"naive": "pred_naive", "seasonal_naive": "pred_seasonal_naive", "moving_avg_4": "pred_moving_avg_4"}
# Columns whose January values would need other SKUs' January data.
CARRIED_FORWARD = ["category_trend", "price_index"]
# The Phase 4 generators draw noise sequentially over the weeks they're given, so
# avg_temp for a week would otherwise depend on how many later weeks had arrived.
# Generating over a fixed horizon makes January values independent of arrival order.
ENRICHMENT_HORIZON_END = "2025-12-29"


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------- labels / splits

def target_is_complete(weeks, last_observed_date) -> np.ndarray:
    """True where the label week (origin + 7 days, Mon-Sun) ends on or before the last
    observed raw date. Weeks are labelled by their Monday (W-MON, label/closed left)."""
    weeks = pd.to_datetime(pd.Series(weeks)).to_numpy()
    target_end = weeks + np.timedelta64(13, "D")
    return target_end <= np.datetime64(pd.Timestamp(last_observed_date))


def holdout_split(df: pd.DataFrame):
    """(train, holdout, holdout_weeks) using the unchanged Phase 5 splitter."""
    from splits.walk_forward import WalkForwardSplitter

    folds, holdout_weeks = WalkForwardSplitter(**SPLIT_CONFIG).get_folds(df)
    if len(folds) != 7 or len(holdout_weeks) != 10:
        raise ValueError("Phase 13 expects the Phase 5 7-fold/10-week protocol")
    in_holdout = df.week.isin(holdout_weeks)
    train, test = df[~in_holdout], df[in_holdout]
    if train.week.max() >= test.week.min():
        raise ValueError("training weeks must precede holdout weeks")
    return train, test, pd.to_datetime(holdout_weeks)


# ---------------------------------------------------------------- models

def _xgb_predict(model, df, feature_cols):
    from .forecast import make_lgb_frame
    return model.predict(make_lgb_frame(df, feature_cols))


def fit_models(train: pd.DataFrame, model_ids=ML_MODELS, feature_cols=None) -> dict:
    """Fit each global model once with its Phase 7 default parameters."""
    from .forecast import ALL_FEATURE_COLS, fit_cb, fit_lgb, fit_rf, fit_xgb

    feature_cols = feature_cols or ALL_FEATURE_COLS
    if train.target_next_week.isna().any():
        raise ValueError("training labels must be complete")
    fitters = {"global_lgb": fit_lgb, "global_xgb": fit_xgb, "global_cb": fit_cb, "global_rf": fit_rf}
    return {m: fitters[m](train, feature_cols) for m in model_ids}


def predict_models(fitted: dict, df: pd.DataFrame, feature_cols=None) -> dict:
    from .forecast import ALL_FEATURE_COLS, predict_cb, predict_lgb, predict_rf

    feature_cols = feature_cols or ALL_FEATURE_COLS
    predictors = {"global_lgb": predict_lgb, "global_xgb": _xgb_predict, "global_cb": predict_cb, "global_rf": predict_rf}
    return {m: np.asarray(predictors[m](model, df, feature_cols), dtype=float) for m, model in fitted.items()}


def baseline_predictions(panel: pd.DataFrame) -> pd.DataFrame:
    """Phase 6 baselines on the full panel (each uses only past/current units_sold)."""
    from .baseline import add_baseline_predictions

    out = add_baseline_predictions(panel[KEYS + ["units_sold"]].copy())
    return out[KEYS + list(BASELINE_COLS.values())]


def build_ledger(eval_rows: pd.DataFrame, ml_preds: dict, baselines: pd.DataFrame, extra_cols=()) -> pd.DataFrame:
    """Long table: one row per (model, series, origin week), paired by panel keys."""
    gold = eval_rows[KEYS + ["target_next_week", *extra_cols]].rename(columns={"target_next_week": "actual"})
    if gold.duplicated(KEYS).any():
        raise ValueError("evaluation keys must be unique")
    base = gold.merge(baselines, on=KEYS, how="left", validate="one_to_one")
    pieces = [base[KEYS + ["actual", *extra_cols]].assign(model_id=m, prediction=base[c].to_numpy())
              for m, c in BASELINE_COLS.items()]
    pieces += [gold.assign(model_id=m, prediction=p) for m, p in ml_preds.items()]
    ledger = pd.concat(pieces, ignore_index=True)
    ledger["model"] = ledger.model_id.map(MODELS)
    return ledger


# ---------------------------------------------------------------- scoring

def score_ledger(ledger: pd.DataFrame, by=()) -> pd.DataFrame:
    """Pooled metrics per model (and optional grouping). bias = sum(pred-actual)/sum(actual).

    Rows without a prediction (seasonal-naive warm-up) are dropped and counted via
    coverage; compare models only where coverage matches.
    """
    rows = []
    for key, part in ledger.groupby(["model_id", *by], sort=False):
        key = key if isinstance(key, tuple) else (key,)
        valid = part[np.isfinite(part.prediction)]
        score = score_all(valid.actual.values, valid.prediction.values) if len(valid) else dict.fromkeys(METRICS, np.nan)
        bias = (valid.prediction.sum() - valid.actual.sum()) / valid.actual.sum() if len(valid) else np.nan
        rows.append(dict(zip(["model_id", *by], key), model=MODELS[key[0]], n_scored=len(valid),
                         coverage=len(valid) / len(part), **score, bias=float(bias)))
    return pd.DataFrame(rows)


def cluster_bootstrap_wape_diff(ledger, model_a, model_b, n_boot=2000, seed=42, cluster=GROUP_KEYS):
    """Percentile CI for WAPE(model_b) - WAPE(model_a), resampling whole series.

    Resampling series (not rows) keeps each series' weeks together, so serial
    correlation within a series isn't treated as independent evidence.
    """
    a = ledger[ledger.model_id.eq(model_a)].set_index(KEYS)
    b = ledger[ledger.model_id.eq(model_b)].set_index(KEYS)
    if not a.index.sort_values().equals(b.index.sort_values()):
        raise ValueError("models must be scored on identical keys")
    b = b.loc[a.index]
    frame = pd.DataFrame({"actual": a.actual, "err_a": (a.prediction - a.actual).abs(),
                          "err_b": (b.prediction - b.actual).abs()}).reset_index()
    per_series = frame.groupby(cluster)[["actual", "err_a", "err_b"]].sum().to_numpy()
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(per_series), size=(n_boot, len(per_series)))
    sums = per_series[draws].sum(axis=1)
    diffs = (sums[:, 2] - sums[:, 1]) / sums[:, 0]
    observed = (per_series[:, 2].sum() - per_series[:, 1].sum()) / per_series[:, 0].sum()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return dict(model_a=model_a, model_b=model_b, WAPE_a=per_series[:, 1].sum() / per_series[:, 0].sum(),
                WAPE_b=per_series[:, 2].sum() / per_series[:, 0].sum(), diff_b_minus_a=float(observed),
                ci_low=float(lo), ci_high=float(hi), n_series=len(per_series), n_boot=n_boot)


# ---------------------------------------------------------------- January batches

def load_january_batches(raw_dir) -> pd.DataFrame:
    """Read every batch_*.parquet in arrival order, tagging each row with its file."""
    files = sorted(Path(raw_dir).glob("batch_*.parquet"))
    if not files:
        raise FileNotFoundError("no batch_*.parquet files found")
    frames = [pd.read_parquet(f).assign(batch_file=f.name) for f in files]
    out = pd.concat(frames, ignore_index=True)
    # Parquet stores microsecond timestamps; match the CSV history's nanosecond dtype.
    out["date"] = pd.to_datetime(out["date"]).astype("datetime64[ns]")
    return out


def _weekly_series_totals(daily):
    return (daily.set_index("date").groupby(GROUP_KEYS)["units_sold"]
            .resample("W-MON", label="left", closed="left").sum())


def audit_data_contract(history: pd.DataFrame, batches: pd.DataFrame, sku: str) -> pd.DataFrame:
    """Compare the batch files with the same SKU's 2022-2024 daily history.

    Every check states the property the downstream weekly roll-up silently relies on;
    `consistent` is a descriptive flag for that property, not a hypothesis test.
    """
    from scipy.stats import ks_2samp

    hist = history[history.sku.eq(sku)]
    new = batches[batches.columns.intersection(history.columns)]
    day_keys = GROUP_KEYS + ["date"]
    rows = []

    def add(check, hist_value, batch_value, consistent, why):
        rows.append(dict(check=check, history=hist_value, batch=batch_value, consistent=bool(consistent), why_it_matters=why))

    same_schema = list(new.columns) == list(history.columns) and (new.dtypes.values == history.dtypes.values).all()
    add("columns and dtypes", f"{len(history.columns)} columns", "identical" if same_schema else "differs",
        same_schema, "Pipeline code must read the batch unchanged.")
    add("SKUs in batch", sku, ", ".join(sorted(batches.sku.unique())), set(batches.sku) == {sku},
        "Cross-SKU features can't be recomputed from a single-SKU batch.")
    add("date range", f"{hist.date.min().date()} → {hist.date.max().date()}",
        f"{batches.date.min().date()} → {batches.date.max().date()}",
        batches.date.min() > hist.date.max() and batches.date.nunique() == (batches.date.max() - batches.date.min()).days + 1,
        "Batches must continue the history without overlap or gaps.")
    hist_dup, new_dup = int(hist.duplicated(day_keys).sum()), int(batches.duplicated(day_keys).sum())
    add("duplicate series-day rows", hist_dup, new_dup, new_dup == hist_dup == 0,
        "Weekly units_sold is a SUM of daily rows; extra rows per day inflate it.")
    hist_rpd = hist.groupby(day_keys).size()
    new_rpd = batches.groupby(day_keys).size()
    add("rows per series-day", f"{hist_rpd.min()}–{hist_rpd.max()}", f"{new_rpd.min()}–{new_rpd.max()}",
        new_rpd.max() <= hist_rpd.max(), "Same as above.")
    hist_cover = hist.groupby(GROUP_KEYS).date.nunique() / ((hist.date.max() - hist.date.min()).days + 1)
    new_cover = batches.groupby(GROUP_KEYS).date.nunique() / ((batches.date.max() - batches.date.min()).days + 1)
    add("share of calendar days with a row", f"{hist_cover.mean():.2f}", f"{new_cover.mean():.2f}",
        abs(new_cover.mean() - hist_cover.mean()) < 0.1, "Weekly sums depend on how many days are recorded.")

    for col in ["units_sold", "price_unit", "stock_available", "delivered_qty", "delivery_days"]:
        ks = ks_2samp(hist[col], batches[col])
        add(f"{col} per row: mean [min–max]",
            f"{hist[col].mean():.2f} [{hist[col].min():g}–{hist[col].max():g}]",
            f"{batches[col].mean():.2f} [{batches[col].min():g}–{batches[col].max():g}] (KS D={ks.statistic:.2f})",
            ks.statistic < 0.2, "Per-row distribution the model learned from.")
    add("promotion rate per row", f"{hist.promotion_flag.mean():.2f}", f"{batches.promotion_flag.mean():.2f}",
        abs(batches.promotion_flag.mean() - hist.promotion_flag.mean()) < 0.1, "Promo frequency drives promo features.")
    # NaN (not assessable, so not "consistent") when either side has no promo or no non-promo rows.
    lift = lambda d: d.units_sold[d.promotion_flag.eq(1)].mean() / d.units_sold[d.promotion_flag.eq(0)].mean()
    h_lift, n_lift = lift(hist), lift(batches)
    add("units per row, promo / no-promo", f"{h_lift:.2f}×", f"{n_lift:.2f}×",
        abs(n_lift - h_lift) < 0.25, "The learned promo response (Phase 8: ≈+28%).")
    full_weeks = _weekly_series_totals(batches[batches.date >= batches.date.min() + pd.offsets.Week(weekday=0)])
    h_weeks = _weekly_series_totals(hist)
    add("weekly units per series (full weeks)", f"median {h_weeks.median():.0f}, p99 {h_weeks.quantile(0.99):.0f}",
        f"median {full_weeks.median():.0f}, range {full_weeks.min():.0f}–{full_weeks.max():.0f}",
        full_weeks.median() <= h_weeks.quantile(0.99), "The quantity the model forecasts.")
    return pd.DataFrame(rows)


def build_january_panel(daily: pd.DataFrame, batches: pd.DataFrame, weekly_features: pd.DataFrame,
                        sku: str = "MI-006"):
    """Weekly feature rows for every origin whose label involves January data.

    Returns (panel, check) where `panel` holds the evaluation origins (the last
    2024 origin whose label is now complete, plus each new origin with a complete
    label) and `check` compares the rebuilt history with weekly_features.csv.
    """
    from data.build_weekly_base import build_weekly_base
    from features.engineer import add_cross_channel_demand_share, add_promo_recency, add_rolling_promo_rate
    from features.enrich import (compute_internal_aggregates, generate_avg_temp, generate_event_score,
                                 generate_inflation_index, generate_school_in_session)

    hist_daily = daily[daily.sku.eq(sku)]
    combined = pd.concat([hist_daily, batches[daily.columns]], ignore_index=True).sort_values(GROUP_KEYS + ["date"])
    base = build_weekly_base(combined)
    attrs = hist_daily[["sku", "brand", "segment", "category"]].drop_duplicates()
    if len(attrs) != 1:
        raise ValueError("SKU attributes must be unique")
    base = base.merge(attrs, on="sku", how="left")
    base = base.merge(compute_internal_aggregates(combined), on=GROUP_KEYS + ["week"], how="left")

    hist_weeks = pd.DatetimeIndex(sorted(weekly_features.week.unique()))
    new_weeks = pd.DatetimeIndex(sorted(set(base.week) - set(hist_weeks)))
    all_weeks = hist_weeks.append(pd.date_range(hist_weeks.max() + pd.Timedelta(days=7), ENRICHMENT_HORIZON_END, freq="7D"))
    if not set(new_weeks) <= set(all_weeks):
        raise ValueError("new weeks fall outside the enrichment horizon")
    regions = weekly_features.region.unique().tolist()  # same order Phase 4 passed in
    base = (base.merge(generate_avg_temp(regions, all_weeks), on=["region", "week"], how="left")
                .merge(generate_inflation_index(all_weeks), on="week", how="left")
                .merge(generate_school_in_session(all_weeks), on="week", how="left")
                .merge(generate_event_score(all_weeks), on="week", how="left"))

    # Cross-SKU columns: values as of the last complete history week, never January data.
    last_week = hist_weeks.max()
    category = attrs.category.iloc[0]
    last_hist = weekly_features[weekly_features.week.eq(last_week)]
    trend = last_hist[last_hist.category.eq(category)].groupby("region").category_trend.first()
    base["category_trend"] = base.region.map(trend)
    base["price_index"] = base.price_unit / last_hist[last_hist.category.eq(category)].price_unit.mean()

    base = add_promo_recency(base)
    base = add_rolling_promo_rate(base)
    base = add_cross_channel_demand_share(base)
    base = base[weekly_features.columns]

    # History check: per-SKU columns must reproduce weekly_features.csv exactly.
    file_rows = weekly_features[weekly_features.sku.eq(sku) & weekly_features.week.lt(last_week)]
    rebuilt = base[base.week.isin(file_rows.week.unique())]
    merged = file_rows.merge(rebuilt, on=KEYS, suffixes=("_file", "_rebuilt"), validate="one_to_one")
    per_sku_cols = [c for c in weekly_features.columns
                    if c not in KEYS + CARRIED_FORWARD + ["avg_temp", "brand", "segment", "category"]]
    check = []
    for col in per_sku_cols:
        a, b = merged[f"{col}_file"], merged[f"{col}_rebuilt"]
        if pd.api.types.is_numeric_dtype(a):
            n_bad = int(((a - b).abs() > 1e-9).sum() + (a.isna() != b.isna()).sum())
        else:
            n_bad = int((a.astype(str) != b.astype(str)).sum())
        check.append(dict(column=col, rows_compared=len(merged), mismatches=n_bad))
    check = pd.DataFrame(check)
    if len(merged) != len(file_rows):
        raise ValueError("rebuilt history is missing rows")

    # Evaluation origins: last 2024 origin (now with a complete label) + every new origin.
    last_file_row = weekly_features[weekly_features.sku.eq(sku) & weekly_features.week.eq(last_week)]
    completed = last_file_row.drop(columns="target_next_week").merge(
        base[KEYS + ["target_next_week"]], on=KEYS, how="left", validate="one_to_one")
    panel = pd.concat([completed[weekly_features.columns], base[base.week.isin(new_weeks)]], ignore_index=True)
    panel["label_week"] = panel.week + pd.Timedelta(days=7)
    return panel.sort_values(KEYS).reset_index(drop=True), check


def replay_arrival_order(daily, batches, weekly_features, panel, sku="MI-006") -> pd.DataFrame:
    """Rebuild the panel after each batch arrives and check that every origin already
    scorable at that point has exactly the same features as in the full build — i.e.
    no feature of an origin depends on a batch that arrived after its forecast date."""
    from .forecast import ALL_FEATURE_COLS

    rows = []
    cols = [c for c in ALL_FEATURE_COLS + ["target_next_week"] if c not in KEYS]
    for n, name in enumerate(sorted(batches.batch_file.unique()), 1):
        available = batches[batches.batch_file.isin(sorted(batches.batch_file.unique())[:n])]
        partial, _ = build_january_panel(daily, available, weekly_features, sku)
        partial = partial[(partial.label_week + pd.Timedelta(days=6)).le(available.date.max())]
        joined = partial.merge(panel, on=KEYS, suffixes=("_p", "_f"), validate="one_to_one")
        n_bad = 0
        for c in cols:
            a, b = joined[f"{c}_p"], joined[f"{c}_f"]
            n_bad += int((a.astype(str) != b.astype(str)).sum()) if not pd.api.types.is_numeric_dtype(a) \
                else int(((a - b).abs() > 1e-9).sum() + (a.isna() != b.isna()).sum())
        rows.append(dict(batches_available=n, last_file=name, data_through=str(available.date.max().date()),
                         scorable_origins=", ".join(str(w.date()) for w in sorted(partial.week.unique())),
                         rows=len(joined), feature_mismatches=n_bad))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- manifest

def make_manifest(root, **facts) -> dict:
    from .forecast import DEFAULT_CB_PARAMS, DEFAULT_LGB_PARAMS, DEFAULT_RF_PARAMS, DEFAULT_XGB_PARAMS

    root = Path(root)
    def git(*args):
        try:
            return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    packages = {p: importlib.metadata.version(p) for p in
                ["numpy", "pandas", "scikit-learn", "scipy", "lightgbm", "xgboost", "catboost", "pyarrow", "matplotlib"]}
    files = ["data/interim/daily_validated.csv", "data/processed/weekly_features.csv", "src/models/holdout.py",
             "src/models/forecast.py", "src/models/baseline.py", "src/models/metrics.py",
             "src/data/build_weekly_base.py", "src/features/enrich.py", "src/features/engineer.py",
             "src/splits/walk_forward.py", "requirements-lock.txt",
             *sorted(str(p.relative_to(root)) for p in (root / "data/raw").glob("batch_*.parquet"))]
    return dict(
        created_at_utc=datetime.now(timezone.utc).isoformat(), source_commit=git("rev-parse", "HEAD"),
        git_status_at_run=git("status", "--porcelain"), python=platform.python_version(), os=platform.platform(),
        packages=packages, input_hashes={p: file_hash(root / p) for p in files}, split_config=SPLIT_CONFIG,
        primary_model=PRIMARY, models=MODELS,
        model_parameters=dict(lightgbm=DEFAULT_LGB_PARAMS, xgboost=DEFAULT_XGB_PARAMS,
                              catboost=DEFAULT_CB_PARAMS, random_forest=DEFAULT_RF_PARAMS),
        metric_aggregation="Pooled over all scored rows (not a mean of per-week scores); bias = sum(pred-actual)/sum(actual).",
        **facts,
    )
