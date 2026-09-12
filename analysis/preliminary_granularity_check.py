"""Preliminary sensitivity analysis: weekly vs. daily forecasting grain.

Status: preliminary / exploratory, not a confirmatory result and not a replacement
for the Phase 7 model. See analysis/GRANULARITY_AND_STOCK_FINDINGS.md for the full
write-up, hedged claims, and open caveats -- read that before quoting any number
printed by this script in the final report.

Three LightGBM specifications are compared on the existing Phase 5 walk-forward folds:

1. direct weekly (this script's compact baseline): one next-week total per
   SKU/channel/region, fixed before that week begins.
2. daily fixed-origin: seven daily forecasts made from information available at the
   end of the origin week only, then summed to the same next-week total. Source-date
   assertions check the daily lag timestamps; a separate assertion checks that the
   seven targets reconstruct the project's own target_next_week. This is the fair,
   same-origin comparison against (1).
3. daily rolling: one-day-ahead forecasts that use actual sales from days already
   past within the target week. This is not leakage -- a system that reforecasts
   daily is legitimately allowed to see those days -- but it answers a different
   question (a reforecast-every-day service) than a plan frozen before the week
   starts, so it is not a fair comparison to (1) or (2).

Missing daily rows are filled as zero sales because the project's weekly sum treats
an absent row as contributing zero. That is an assumption, not proof that the source
data records true zero-demand days this way -- daily_panel_completeness() quantifies
how much of the daily targets this assumption actually touches (~15% of rows).

7 walk-forward folds share overlapping expanding-window training data, so they are
not independent samples; p-values below should be read as exploratory evidence of
direction and rough magnitude, not as a confirmatory significance test.
"""

from pathlib import Path
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from splits.walk_forward import WalkForwardSplitter  # noqa: E402
from models.forecast import (  # noqa: E402
    ALL_FEATURE_COLS,
    fit_predict_lgb as fit_predict_project_lgb,
    prepare_categoricals,
)


KEYS = ["sku", "channel", "region"]
STATIC = ["category", "segment", "brand"]
CAT_COLS = KEYS + STATIC

MODEL_PARAMS = {
    "objective": "regression",
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 20,
    "verbosity": -1,
    "random_state": 42,
    "n_jobs": 4,
    "deterministic": True,
    "force_row_wise": True,
}


def wape(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.abs(actual - predicted).sum() / np.abs(actual).sum())


def prepare_weekly(weekly: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = weekly.copy()
    out["week"] = pd.to_datetime(out["week"])
    out["target_week"] = out["week"] + pd.Timedelta(days=7)
    out["target_week_number"] = out["target_week"].dt.isocalendar().week.astype(int)
    out["target_month"] = out["target_week"].dt.month
    out["target_year"] = out["target_week"].dt.year
    features = CAT_COLS + [
        "units_sold",
        "lag_1",
        "lag_2",
        "rolling_mean_4",
        "rolling_std_4",
        "target_week_number",
        "target_month",
        "target_year",
    ]
    for col in CAT_COLS:
        out[col] = out[col].astype("category")
    return out, features


def complete_daily_panel(daily: pd.DataFrame) -> pd.DataFrame:
    daily = daily.copy()
    daily["date"] = pd.to_datetime(daily["date"])
    attrs = daily.groupby(KEYS, observed=True)[STATIC].first()
    pieces = []
    for keys, group in daily.groupby(KEYS, observed=True, sort=False):
        full_dates = pd.date_range(group["date"].min(), group["date"].max(), freq="D")
        series = group.set_index("date")["units_sold"].reindex(full_dates, fill_value=0.0)
        frame = pd.DataFrame({"date": full_dates, "units_sold_daily": series.to_numpy(float)})
        for name, value in zip(KEYS, keys):
            frame[name] = value
        for name in STATIC:
            frame[name] = attrs.loc[keys, name]
        pieces.append(frame)
    return pd.concat(pieces, ignore_index=True)


def prepare_daily_fixed_origin(
    daily: pd.DataFrame, weekly: pd.DataFrame
) -> tuple[pd.DataFrame, list[str]]:
    panel = complete_daily_panel(daily)
    value_lookup = panel.set_index(KEYS + ["date"])["units_sold_daily"]

    base_cols = KEYS + STATIC + [
        "week",
        "units_sold",
        "lag_1",
        "lag_2",
        "rolling_mean_4",
        "rolling_std_4",
        "target_next_week",
    ]
    origins = weekly[base_cols].copy()
    rows = []
    for horizon in range(1, 8):
        part = origins.copy()
        part["horizon"] = horizon
        part["target_date"] = part["week"] + pd.Timedelta(days=7 + horizon - 1)
        rows.append(part)
    out = pd.concat(rows, ignore_index=True)
    out = out.sort_values(KEYS + ["week", "horizon"]).reset_index(drop=True)

    # Every daily lag must come from no later than Sunday at the end of the
    # origin week. This is the actual temporal-leakage assertion; the target-sum
    # assertion below checks outcome alignment, which is a separate property.
    origin_end = out["week"] + pd.Timedelta(days=6)
    for offset in [7, 14, 28]:
        source_date = out["target_date"] - pd.Timedelta(days=offset)
        if (source_date > origin_end).any():
            raise AssertionError(f"daily_lag_{offset} uses data after the forecast origin")

    def lookup_for(offset: int) -> np.ndarray:
        idx = pd.MultiIndex.from_frame(
            out[KEYS].assign(date=out["target_date"] - pd.Timedelta(days=offset))
        )
        return value_lookup.reindex(idx).fillna(0.0).to_numpy()

    out["target_daily"] = lookup_for(0)
    out["daily_lag_7"] = lookup_for(7)
    out["daily_lag_14"] = lookup_for(14)
    out["daily_lag_28"] = lookup_for(28)
    out["target_dow"] = out["target_date"].dt.dayofweek
    out["target_month"] = out["target_date"].dt.month
    out["target_week_number"] = out["target_date"].dt.isocalendar().week.astype(int)

    # Verify that seven reconstructed daily targets equal the project's weekly target.
    reconstructed = out.groupby(KEYS + ["week"], observed=True)["target_daily"].sum()
    expected = origins.set_index(KEYS + ["week"])["target_next_week"]
    aligned = reconstructed.reindex(expected.index)
    if not np.allclose(aligned.to_numpy(), expected.to_numpy()):
        raise AssertionError("Daily targets do not reconstruct target_next_week")

    features = CAT_COLS + [
        "units_sold",
        "lag_1",
        "lag_2",
        "rolling_mean_4",
        "rolling_std_4",
        "daily_lag_7",
        "daily_lag_14",
        "daily_lag_28",
        "horizon",
        "target_dow",
        "target_month",
        "target_week_number",
    ]
    for col in CAT_COLS:
        out[col] = out[col].astype("category")
    return out, features


def prepare_daily_rolling(daily: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """One-day-ahead rows using actual sales already observed earlier that week.

    Not leakage: a system that reforecasts daily is legitimately entitled to see
    days that have already happened. It benefits from a later forecast origin and
    daily information updates, so it answers a different question (a
    reforecast-every-day service level) than a plan frozen before the week begins --
    not directly comparable to the weekly or daily-fixed-origin variants above.
    """
    out = complete_daily_panel(daily).sort_values(KEYS + ["date"]).reset_index(drop=True)
    grouped = out.groupby(KEYS, observed=True, sort=False)["units_sold_daily"]
    for lag in [1, 7, 14, 28]:
        out[f"daily_lag_{lag}"] = grouped.shift(lag)
    shifted = grouped.shift(1)
    out["daily_rolling_mean_7"] = (
        shifted.groupby([out[k] for k in KEYS], observed=True)
        .rolling(7)
        .mean()
        .reset_index(level=KEYS, drop=True)
    )
    out["daily_rolling_mean_28"] = (
        shifted.groupby([out[k] for k in KEYS], observed=True)
        .rolling(28)
        .mean()
        .reset_index(level=KEYS, drop=True)
    )
    out["target_dow"] = out["date"].dt.dayofweek
    out["target_month"] = out["date"].dt.month
    out["target_week_number"] = out["date"].dt.isocalendar().week.astype(int)
    out["target_week"] = out["date"] - pd.to_timedelta(out["date"].dt.dayofweek, unit="D")
    out["week"] = out["target_week"] - pd.Timedelta(days=7)
    out = out.dropna().reset_index(drop=True)
    features = CAT_COLS + [
        "daily_lag_1",
        "daily_lag_7",
        "daily_lag_14",
        "daily_lag_28",
        "daily_rolling_mean_7",
        "daily_rolling_mean_28",
        "target_dow",
        "target_month",
        "target_week_number",
    ]
    for col in CAT_COLS:
        out[col] = out[col].astype("category")
    return out, features


def fit_predict(train: pd.DataFrame, valid: pd.DataFrame, features: list[str], target: str) -> np.ndarray:
    model = lgb.LGBMRegressor(**MODEL_PARAMS)
    model.fit(train[features], train[target])
    return np.maximum(0.0, model.predict(valid[features]))


def stock_audit(daily: pd.DataFrame) -> dict[str, float]:
    """Checks two narrow claims only -- see the module docstring for what this does
    and does not establish about how stock_available was actually generated."""
    ordered = daily.sort_values(["sku", "region", "date"]).copy()
    group = ordered.groupby(["sku", "region"], observed=True, sort=False)
    expected_next = np.maximum(
        0.0, ordered["stock_available"] + ordered["delivered_qty"] - ordered["units_sold"]
    )
    actual_next = group["stock_available"].shift(-1)
    next_date = group["date"].shift(-1)
    day_delta = (next_date - ordered["date"]).dt.days
    comparable = actual_next.notna()

    def transition_counts(mask: pd.Series) -> tuple[int, int]:
        return (
            int(mask.sum()),
            int(np.isclose(actual_next[mask], expected_next[mask]).sum()),
        )

    adjacent_n, adjacent_exact = transition_counts(comparable)
    same_date_n, same_date_exact = transition_counts(comparable & day_delta.eq(0))
    next_day_n, next_day_exact = transition_counts(comparable & day_delta.eq(1))
    gap_n, gap_exact = transition_counts(comparable & day_delta.gt(1))

    channel_stock = daily.groupby(["sku", "region", "date"], observed=True).agg(
        channels=("channel", "nunique"), stocks=("stock_available", "nunique")
    )
    multi = channel_stock["channels"] >= 2
    return {
        "adjacent_row_transitions": adjacent_n,
        "adjacent_row_exact": adjacent_exact,
        "same_date_transitions": same_date_n,
        "same_date_exact": same_date_exact,
        "next_calendar_day_transitions": next_day_n,
        "next_calendar_day_exact": next_day_exact,
        "gap_over_one_day_transitions": gap_n,
        "gap_over_one_day_exact": gap_exact,
        "multi_channel_dates": int(multi.sum()),
        "same_stock_dates": int((channel_stock.loc[multi, "stocks"] == 1).sum()),
    }


def daily_panel_completeness(daily: pd.DataFrame) -> dict[str, float]:
    """Fraction of the (sku, channel, region) daily panel actually present.

    The daily_then_sum and daily_rolling targets fill absent dates as zero sales
    (see complete_daily_panel). This quantifies how much of that target is real
    versus that filled-in assumption.
    """
    span = daily.groupby(KEYS)["date"].agg(lambda s: (s.max() - s.min()).days + 1)
    actual = daily.groupby(KEYS)["date"].nunique()
    return {
        "expected_days": int(span.sum()),
        "actual_days": int(actual.sum()),
        "present_fraction": float(actual.sum() / span.sum()),
    }


def daily_vs_weekly_noise(daily: pd.DataFrame, weekly: pd.DataFrame) -> dict[str, float]:
    """Coefficient of variation of units_sold at each grain, per (sku, channel, region).

    Supporting evidence for -- not proof of -- the hypothesis that a noisier daily
    target partly explains daily_then_sum's higher WAPE. A higher CV does not by
    itself establish that noise (rather than optimization-target mismatch, untuned
    hyperparameters, or the ~15% filled-zero rows) is the mechanism.
    """
    def cv(s: pd.Series) -> float:
        return float(s.std() / s.mean()) if s.mean() > 0 else np.nan

    daily_cv = daily.groupby(KEYS)["units_sold"].apply(cv).mean()
    weekly_cv = weekly.groupby(KEYS)["units_sold"].apply(cv).mean()
    return {
        "daily_cv": float(daily_cv),
        "weekly_cv": float(weekly_cv),
        "ratio": float(daily_cv / weekly_cv),
    }


def enrichment_semantics_audit(daily: pd.DataFrame, weekly_features_path: Path) -> dict[str, object]:
    """Checks two claims about src/features/enrich.py's compute_internal_aggregates:

    1. `deliveries=("delivery_days", "count")` -- delivery_days is a per-row lead-time
       figure and never null, so the aggregation literally counts observed rows.
       Because delivered_qty is positive on all but three validated rows, that count
       is numerically almost the same as positive-delivery days. Whether it represents
       delivery frequency or data completeness depends on what absent dates mean.
    2. Both `stock_avg` and the weekly `stock_available` are independently checked
       against a fresh group-week mean across every modeled row. Their calculation is
       correct, but the two resulting model features are exact duplicates.
    """
    delivered_qty_positive_rate = float((daily["delivered_qty"] > 0).mean())
    delivery_days_null_rate = float(daily["delivery_days"].isna().mean())

    weekly = pd.read_csv(weekly_features_path, parse_dates=["week"])
    daily_for_agg = daily.copy()
    daily_for_agg["week"] = daily_for_agg["date"] - pd.to_timedelta(
        daily_for_agg["date"].dt.dayofweek, unit="D"
    )
    daily_for_agg["positive_delivery"] = daily_for_agg["delivered_qty"].gt(0).astype(int)
    recomputed = daily_for_agg.groupby(KEYS + ["week"], observed=True).agg(
        recomputed_stock_mean=("stock_available", "mean"),
        observed_rows=("delivery_days", "count"),
        positive_delivery_days=("positive_delivery", "sum"),
    ).reset_index()
    audited = weekly.merge(
        recomputed,
        on=KEYS + ["week"],
        how="left",
        validate="one_to_one",
    )

    return {
        "delivered_qty_positive_rate": delivered_qty_positive_rate,
        "delivery_days_null_rate": delivery_days_null_rate,
        "deliveries_value_counts": weekly["deliveries"].value_counts().sort_index().to_dict(),
        "deliveries_equals_observed_rows_all": bool(
            np.array_equal(audited["deliveries"], audited["observed_rows"])
        ),
        "deliveries_vs_positive_days_mismatched_rows": int(
            (audited["deliveries"] != audited["positive_delivery_days"]).sum()
        ),
        "stock_avg_matches_recomputed_all": bool(
            np.allclose(audited["stock_avg"], audited["recomputed_stock_mean"])
        ),
        "stock_available_matches_recomputed_all": bool(
            np.allclose(audited["stock_available"], audited["recomputed_stock_mean"])
        ),
        "stock_avg_duplicates_stock_available_all": bool(
            np.allclose(audited["stock_avg"], audited["stock_available"])
        ),
    }


def main() -> None:
    daily = pd.read_csv(ROOT / "data/interim/daily_validated.csv", parse_dates=["date"])
    weekly = pd.read_csv(ROOT / "data/processed/weekly_features.csv", parse_dates=["week"])
    weekly = prepare_categoricals(weekly)
    weekly, weekly_features = prepare_weekly(weekly)
    daily_fixed, daily_features = prepare_daily_fixed_origin(daily, weekly)
    daily_rolling, daily_rolling_features = prepare_daily_rolling(daily)
    # Keep exactly the same eligible group-origin rows as the weekly experiment.
    daily_rolling = daily_rolling.merge(
        weekly[KEYS + ["week"]].drop_duplicates(),
        on=KEYS + ["week"],
        how="inner",
        validate="many_to_one",
    )

    splitter = WalkForwardSplitter(
        initial_train_end="2023-09-30", val_fold_weeks=8, final_test_weeks=10
    )
    folds, _ = splitter.get_folds(weekly)
    fold_rows = []
    for fold in folds:
        weekly_train = weekly[weekly["week"].isin(fold.train_weeks)]
        weekly_valid = weekly[weekly["week"].isin(fold.val_weeks)]
        weekly_pred = fit_predict(
            weekly_train, weekly_valid, weekly_features, "target_next_week"
        )
        project_weekly_pred = fit_predict_project_lgb(
            weekly_train, weekly_valid, feature_cols=ALL_FEATURE_COLS
        )

        daily_train = daily_fixed[daily_fixed["week"].isin(fold.train_weeks)]
        daily_valid = daily_fixed[daily_fixed["week"].isin(fold.val_weeks)].copy()
        daily_valid["prediction"] = fit_predict(
            daily_train, daily_valid, daily_features, "target_daily"
        )
        daily_aggregated = daily_valid.groupby(KEYS + ["week"], observed=True).agg(
            actual=("target_daily", "sum"), prediction=("prediction", "sum")
        )

        rolling_train = daily_rolling[daily_rolling["week"].isin(fold.train_weeks)]
        rolling_valid = daily_rolling[daily_rolling["week"].isin(fold.val_weeks)].copy()
        rolling_valid["prediction"] = fit_predict(
            rolling_train,
            rolling_valid,
            daily_rolling_features,
            "units_sold_daily",
        )
        rolling_aggregated = rolling_valid.groupby(KEYS + ["week"], observed=True).agg(
            actual=("units_sold_daily", "sum"), prediction=("prediction", "sum")
        )
        weekly_expected = weekly_valid.set_index(KEYS + ["week"])["target_next_week"]
        rolling_actual = rolling_aggregated["actual"].reindex(weekly_expected.index)
        if rolling_actual.isna().any() or not np.allclose(
            rolling_actual.to_numpy(), weekly_expected.to_numpy()
        ):
            raise AssertionError("Rolling daily targets do not match the weekly validation target")

        fold_rows.append(
            {
                "fold": fold.fold_id,
                "weekly_wape": wape(
                    weekly_valid["target_next_week"].to_numpy(), weekly_pred
                ),
                "project_weekly_wape": wape(
                    weekly_valid["target_next_week"].to_numpy(), project_weekly_pred
                ),
                "daily_then_sum_wape": wape(
                    daily_aggregated["actual"].to_numpy(),
                    daily_aggregated["prediction"].to_numpy(),
                ),
                "daily_rolling_wape": wape(
                    rolling_aggregated["actual"].to_numpy(),
                    rolling_aggregated["prediction"].to_numpy(),
                ),
            }
        )
        print(f"completed fold {fold.fold_id + 1}/{len(folds)}", flush=True)

    results = pd.DataFrame(fold_rows)
    results["daily_minus_weekly"] = results["daily_then_sum_wape"] - results["weekly_wape"]
    results["rolling_minus_weekly"] = results["daily_rolling_wape"] - results["weekly_wape"]
    results["daily_minus_project"] = (
        results["daily_then_sum_wape"] - results["project_weekly_wape"]
    )
    results["rolling_minus_project"] = (
        results["daily_rolling_wape"] - results["project_weekly_wape"]
    )
    print("\nFOLD_RESULTS")
    print(results.to_string(index=False))
    print("\nMEANS")
    print(results.mean(numeric_only=True).to_string())
    print(
        "\nCAVEAT: 7 expanding-window folds share overlapping training data -- they are "
        "not independent samples. The t-test below assumes independence and so overstates "
        "confidence; Wilcoxon and the sign test are reported alongside it but do not fix "
        "that non-independence either. Read all three as exploratory, not confirmatory."
    )
    for comparison in [
        "daily_minus_weekly",
        "rolling_minus_weekly",
        "daily_minus_project",
        "rolling_minus_project",
    ]:
        diffs = results[comparison].to_numpy()
        t_test = stats.ttest_1samp(diffs, popmean=0.0)
        wilcoxon = stats.wilcoxon(diffs)
        n_pos = int((diffs > 0).sum())
        sign = stats.binomtest(n_pos, len(diffs), 0.5, alternative="two-sided")
        print(
            f"{comparison}: {n_pos}/{len(diffs)} folds positive, "
            f"paired t={t_test.statistic:.4f} (p={t_test.pvalue:.4f}), "
            f"wilcoxon p={wilcoxon.pvalue:.4f}, sign-test p={sign.pvalue:.4f}",
            flush=True,
        )
    print(
        "\nNOTE: daily_minus_project and rolling_minus_project compare against the "
        "project's full-feature production model, which uses a different feature set "
        "than the daily variants -- these two are confounded by feature-set differences, "
        "not a clean read on granularity alone. daily_minus_weekly and rolling_minus_weekly "
        "compare against this script's own compact weekly baseline (same feature philosophy, "
        "same code path) and are the closer-to-apples-to-apples pair for a granularity claim."
    )

    print("\nDAILY_PANEL_COMPLETENESS")
    print(daily_panel_completeness(daily))
    print("\nDAILY_VS_WEEKLY_NOISE (coefficient of variation)")
    print(daily_vs_weekly_noise(daily, weekly))
    print("\nSTOCK_AUDIT")
    print(
        "Rules out a single shared stock-available scalar copied across every channel, "
        "and rules out Beata Faron's shared-pool snippet as the literal generator. Does "
        "NOT establish that each channel's stock is fully independent -- a more complex "
        "shared-allocation mechanism is not ruled out by this data alone."
    )
    print(stock_audit(daily))
    print("\nENRICHMENT_SEMANTICS_AUDIT (src/features/enrich.py compute_internal_aggregates)")
    print(enrichment_semantics_audit(daily, ROOT / "data/processed/weekly_features.csv"))


if __name__ == "__main__":
    main()
