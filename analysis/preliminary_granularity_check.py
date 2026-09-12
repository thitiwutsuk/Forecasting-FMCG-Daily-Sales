"""Preliminary, leakage-aware comparison of weekly and daily forecasting grains.

This is deliberately a diagnostic rather than a replacement for the Phase 7 model.
It compares two compact LightGBM specifications on the existing Phase 5 folds:

1. direct weekly: one next-week total per SKU/channel/region;
2. daily fixed-origin: seven daily forecasts made from information available at the
   end of the origin week, then summed to the same next-week total.

Missing daily rows are filled as zero sales because the project's weekly sum treats
an absent row as contributing zero. That is an assumption, not proof that the source
data records true zero-demand days this way.
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
    """One-day-ahead rows that may use actual sales from earlier validation days.

    This represents a forecast refreshed every day. Its weekly aggregate is useful as
    a diagnostic, but it is not the same service level as a next-week forecast fixed
    before that week begins.
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
    ordered = daily.sort_values(["sku", "region", "date"]).copy()
    group = ordered.groupby(["sku", "region"], observed=True, sort=False)
    expected_next = np.maximum(
        0.0, ordered["stock_available"] + ordered["delivered_qty"] - ordered["units_sold"]
    )
    actual_next = group["stock_available"].shift(-1)
    comparable = actual_next.notna()

    channel_stock = daily.groupby(["sku", "region", "date"], observed=True).agg(
        channels=("channel", "nunique"), stocks=("stock_available", "nunique")
    )
    multi = channel_stock["channels"] >= 2
    return {
        "transition_n": int(comparable.sum()),
        "transition_exact": int(np.isclose(actual_next[comparable], expected_next[comparable]).sum()),
        "multi_channel_dates": int(multi.sum()),
        "same_stock_dates": int((channel_stock.loc[multi, "stocks"] == 1).sum()),
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
    for comparison in [
        "daily_minus_weekly",
        "rolling_minus_weekly",
        "daily_minus_project",
        "rolling_minus_project",
    ]:
        test = stats.ttest_1samp(results[comparison], popmean=0.0)
        print(
            f"{comparison}: paired t={test.statistic:.4f}, p={test.pvalue:.6f}",
            flush=True,
        )
    print("\nSTOCK_AUDIT")
    print(stock_audit(daily))


if __name__ == "__main__":
    main()
