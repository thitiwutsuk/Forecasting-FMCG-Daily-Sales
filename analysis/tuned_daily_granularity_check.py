"""Time-separated tuning check for fixed-origin daily-then-sum forecasting.

This follow-up addresses two limitations of ``preliminary_granularity_check.py``:

* the daily model is given its own feature engineering and hyperparameter search;
* model selection uses only the first three walk-forward validation blocks, while
  the last four blocks are reserved for a locked, later-time evaluation.

The project's final 10-week Phase 13 holdout remains untouched.  This is still a
sensitivity analysis, not proof that one forecasting grain is universally better.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "analysis"))

from models.forecast import (  # noqa: E402
    ALL_FEATURE_COLS,
    fit_predict_lgb as fit_predict_project_lgb,
    prepare_categoricals,
)
from preliminary_granularity_check import (  # noqa: E402
    CAT_COLS,
    KEYS,
    MODEL_PARAMS,
    complete_daily_panel,
    fit_predict,
    prepare_daily_fixed_origin,
    prepare_weekly,
    wape,
)
from splits.walk_forward import WalkForwardSplitter  # noqa: E402


DEV_FOLDS = 3
BOOTSTRAP_REPS = 20_000
BOOTSTRAP_BLOCK_WEEKS = 4


@dataclass(frozen=True)
class Candidate:
    name: str
    feature_set: str
    params: dict[str, object]


CANDIDATES = [
    Candidate("base_l2", "base", {}),
    Candidate("enhanced_l2", "enhanced", {}),
    Candidate("enhanced_l1", "enhanced", {"objective": "regression_l1"}),
    Candidate("enhanced_huber", "enhanced", {"objective": "huber"}),
    Candidate("enhanced_poisson", "enhanced", {"objective": "poisson"}),
    Candidate(
        "enhanced_l1_small_leaves",
        "enhanced",
        {"objective": "regression_l1", "num_leaves": 15, "min_child_samples": 40},
    ),
    Candidate(
        "enhanced_l1_flexible",
        "enhanced",
        {"objective": "regression_l1", "num_leaves": 63, "min_child_samples": 10},
    ),
    Candidate(
        "enhanced_l1_slow",
        "enhanced",
        {"objective": "regression_l1", "n_estimators": 500, "learning_rate": 0.03},
    ),
    Candidate(
        "enhanced_l2_flexible",
        "enhanced",
        {"num_leaves": 63, "min_child_samples": 10},
    ),
]


def add_enhanced_daily_features(
    daily: pd.DataFrame, daily_fixed: pd.DataFrame
) -> tuple[pd.DataFrame, list[str]]:
    """Add fixed-origin same-weekday history and missing-row indicators.

    Every added lag is at least seven days behind its target date, so it is known
    by Sunday at the end of the origin week.  The observed indicators allow the
    model to distinguish a recorded zero from the panel-completion zero assumption.
    """
    out = daily_fixed.copy()
    panel = complete_daily_panel(daily)
    value_lookup = panel.set_index(KEYS + ["date"])["units_sold_daily"]
    observed = daily.assign(_observed=1.0).set_index(KEYS + ["date"])["_observed"]
    origin_end = out["week"] + pd.Timedelta(days=6)

    def lookup(series: pd.Series, source_date: pd.Series, fill: float) -> np.ndarray:
        idx = pd.MultiIndex.from_frame(out[KEYS].assign(date=source_date))
        return series.reindex(idx).fillna(fill).to_numpy(float)

    lag_cols = []
    observed_cols = []
    for offset in range(7, 57, 7):
        source_date = out["target_date"] - pd.Timedelta(days=offset)
        if (source_date > origin_end).any():
            raise AssertionError(f"enhanced daily lag {offset} crosses forecast origin")
        lag_col = f"daily_lag_{offset}"
        observed_col = f"daily_lag_{offset}_observed"
        out[lag_col] = lookup(value_lookup, source_date, 0.0)
        out[observed_col] = lookup(observed, source_date, 0.0)
        lag_cols.append(lag_col)
        observed_cols.append(observed_col)

    out["same_dow_mean_4"] = out[lag_cols[:4]].mean(axis=1)
    out["same_dow_mean_8"] = out[lag_cols].mean(axis=1)
    out["same_dow_median_8"] = out[lag_cols].median(axis=1)
    out["same_dow_std_8"] = out[lag_cols].std(axis=1)
    out["last_week_share"] = np.divide(
        out["daily_lag_7"],
        out["units_sold"],
        out=np.zeros(len(out), dtype=float),
        where=out["units_sold"].to_numpy() > 0,
    )
    share_2 = np.divide(
        out["daily_lag_14"],
        out["lag_1"],
        out=np.zeros(len(out), dtype=float),
        where=out["lag_1"].to_numpy() > 0,
    )
    share_3 = np.divide(
        out["daily_lag_21"],
        out["lag_2"],
        out=np.zeros(len(out), dtype=float),
        where=out["lag_2"].to_numpy() > 0,
    )
    out["same_dow_share_mean_3"] = np.column_stack(
        [out["last_week_share"].to_numpy(), share_2, share_3]
    ).mean(axis=1)

    features = CAT_COLS + [
        "units_sold",
        "lag_1",
        "lag_2",
        "rolling_mean_4",
        "rolling_std_4",
        *lag_cols,
        *observed_cols,
        "same_dow_mean_4",
        "same_dow_mean_8",
        "same_dow_median_8",
        "same_dow_std_8",
        "last_week_share",
        "same_dow_share_mean_3",
        "horizon",
        "target_dow",
        "target_month",
        "target_week_number",
    ]
    return out, features


def fit_predict_candidate(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    features: list[str],
    candidate: Candidate,
) -> np.ndarray:
    params = {**MODEL_PARAMS, **candidate.params}
    model = lgb.LGBMRegressor(**params)
    model.fit(train[features], train["target_daily"])
    return np.maximum(0.0, model.predict(valid[features]))


def aggregate_daily_predictions(frame: pd.DataFrame, prediction: np.ndarray) -> pd.DataFrame:
    scored = frame[KEYS + ["week", "target_daily"]].copy()
    scored["prediction"] = prediction
    return scored.groupby(KEYS + ["week"], observed=True).agg(
        actual=("target_daily", "sum"), prediction=("prediction", "sum")
    )


def pooled_wape(parts: list[pd.DataFrame]) -> float:
    scored = pd.concat(parts)
    return wape(scored["actual"].to_numpy(), scored["prediction"].to_numpy())


def tune_daily(
    daily_base: pd.DataFrame,
    base_features: list[str],
    daily_enhanced: pd.DataFrame,
    enhanced_features: list[str],
    folds: list,
) -> tuple[Candidate, pd.DataFrame]:
    rows = []
    for candidate in CANDIDATES:
        frame, features = (
            (daily_base, base_features)
            if candidate.feature_set == "base"
            else (daily_enhanced, enhanced_features)
        )
        fold_scores = []
        scored_parts = []
        for fold in folds[:DEV_FOLDS]:
            train = frame[frame["week"].isin(fold.train_weeks)]
            valid = frame[frame["week"].isin(fold.val_weeks)]
            aggregated = aggregate_daily_predictions(
                valid, fit_predict_candidate(train, valid, features, candidate)
            )
            scored_parts.append(aggregated)
            fold_scores.append(
                wape(aggregated["actual"].to_numpy(), aggregated["prediction"].to_numpy())
            )
        rows.append(
            {
                "candidate": candidate.name,
                "feature_set": candidate.feature_set,
                "development_pooled_wape": pooled_wape(scored_parts),
                "development_mean_fold_wape": float(np.mean(fold_scores)),
                "development_fold_wapes": ",".join(f"{x:.6f}" for x in fold_scores),
            }
        )
        print(f"tuned {candidate.name}", flush=True)
    leaderboard = pd.DataFrame(rows).sort_values(
        ["development_pooled_wape", "candidate"], ignore_index=True
    )
    winner_name = leaderboard.iloc[0]["candidate"]
    winner = next(candidate for candidate in CANDIDATES if candidate.name == winner_name)
    return winner, leaderboard


def circular_block_bootstrap(
    weekly_errors: pd.DataFrame,
    challenger: str,
    baseline: str,
    reps: int = BOOTSTRAP_REPS,
    block_weeks: int = BOOTSTRAP_BLOCK_WEEKS,
) -> dict[str, float]:
    """Bootstrap paired WAPE differences while resampling contiguous week blocks."""
    ordered = weekly_errors.sort_values("target_week").reset_index(drop=True)
    n = len(ordered)
    rng = np.random.default_rng(42)
    blocks_needed = int(np.ceil(n / block_weeks))
    starts = rng.integers(0, n, size=(reps, blocks_needed))
    offsets = np.arange(block_weeks)
    idx = ((starts[:, :, None] + offsets) % n).reshape(reps, -1)[:, :n]
    challenger_errors = ordered[challenger].to_numpy()
    baseline_errors = ordered[baseline].to_numpy()
    actuals = ordered["actual_total"].to_numpy()
    diffs = (
        challenger_errors[idx].sum(axis=1) - baseline_errors[idx].sum(axis=1)
    ) / actuals[idx].sum(axis=1)
    point = (ordered[challenger].sum() - ordered[baseline].sum()) / ordered[
        "actual_total"
    ].sum()
    return {
        "point_difference": float(point),
        "ci_2.5%": float(np.quantile(diffs, 0.025)),
        "ci_97.5%": float(np.quantile(diffs, 0.975)),
        "bootstrap_probability_challenger_better": float((diffs < 0).mean()),
    }


def score_development_weekly_baselines(
    weekly: pd.DataFrame, weekly_features: list[str], folds: list
) -> dict[str, float]:
    """Context only; these scores do not participate in candidate selection."""
    actual_parts = []
    compact_parts = []
    project_parts = []
    for fold in folds[:DEV_FOLDS]:
        train = weekly[weekly["week"].isin(fold.train_weeks)]
        valid = weekly[weekly["week"].isin(fold.val_weeks)]
        actual_parts.append(valid["target_next_week"].to_numpy())
        compact_parts.append(
            fit_predict(train, valid, weekly_features, "target_next_week")
        )
        project_parts.append(
            np.maximum(
                0.0,
                fit_predict_project_lgb(train, valid, feature_cols=ALL_FEATURE_COLS),
            )
        )
    actual = np.concatenate(actual_parts)
    return {
        "compact_weekly": wape(actual, np.concatenate(compact_parts)),
        "project_weekly": wape(actual, np.concatenate(project_parts)),
    }


def main() -> None:
    daily = pd.read_csv(ROOT / "data/interim/daily_validated.csv", parse_dates=["date"])
    weekly = pd.read_csv(ROOT / "data/processed/weekly_features.csv", parse_dates=["week"])
    weekly = prepare_categoricals(weekly)
    weekly, weekly_features = prepare_weekly(weekly)
    daily_base, base_features = prepare_daily_fixed_origin(daily, weekly)
    daily_enhanced, enhanced_features = add_enhanced_daily_features(daily, daily_base)

    splitter = WalkForwardSplitter(
        initial_train_end="2023-09-30", val_fold_weeks=8, final_test_weeks=10
    )
    folds, final_test_weeks = splitter.get_folds(weekly)
    if len(folds) <= DEV_FOLDS:
        raise AssertionError("not enough folds for time-separated development/evaluation")

    winner, leaderboard = tune_daily(
        daily_base,
        base_features,
        daily_enhanced,
        enhanced_features,
        folds,
    )
    development_weekly = score_development_weekly_baselines(
        weekly, weekly_features, folds
    )
    winner_frame, winner_features = (
        (daily_base, base_features)
        if winner.feature_set == "base"
        else (daily_enhanced, enhanced_features)
    )

    evaluation_rows = []
    weekly_error_rows = []
    for fold in folds[DEV_FOLDS:]:
        weekly_train = weekly[weekly["week"].isin(fold.train_weeks)]
        weekly_valid = weekly[weekly["week"].isin(fold.val_weeks)].copy()
        compact_pred = fit_predict(
            weekly_train, weekly_valid, weekly_features, "target_next_week"
        )
        project_pred = np.maximum(
            0.0,
            fit_predict_project_lgb(
                weekly_train, weekly_valid, feature_cols=ALL_FEATURE_COLS
            ),
        )

        daily_train = winner_frame[winner_frame["week"].isin(fold.train_weeks)]
        daily_valid = winner_frame[winner_frame["week"].isin(fold.val_weeks)]
        daily_aggregated = aggregate_daily_predictions(
            daily_valid,
            fit_predict_candidate(
                daily_train, daily_valid, winner_features, winner
            ),
        )
        expected = weekly_valid.set_index(KEYS + ["week"])["target_next_week"]
        daily_aggregated = daily_aggregated.reindex(expected.index)
        if daily_aggregated.isna().any().any() or not np.allclose(
            daily_aggregated["actual"].to_numpy(), expected.to_numpy()
        ):
            raise AssertionError("daily and weekly evaluation targets are misaligned")

        evaluation_rows.append(
            {
                "fold": fold.fold_id,
                "origin_week_start": pd.Timestamp(fold.val_weeks[0]).date(),
                "origin_week_end": pd.Timestamp(fold.val_weeks[-1]).date(),
                "compact_weekly_wape": wape(expected.to_numpy(), compact_pred),
                "project_weekly_wape": wape(expected.to_numpy(), project_pred),
                "tuned_daily_then_sum_wape": wape(
                    expected.to_numpy(), daily_aggregated["prediction"].to_numpy()
                ),
            }
        )

        panel_scores = pd.DataFrame(
            {
                "week": weekly_valid["week"].to_numpy(),
                "target_week": (weekly_valid["week"] + pd.Timedelta(days=7)).to_numpy(),
                "actual": expected.to_numpy(),
                "compact_abs_error": np.abs(expected.to_numpy() - compact_pred),
                "project_abs_error": np.abs(expected.to_numpy() - project_pred),
                "daily_abs_error": np.abs(
                    expected.to_numpy() - daily_aggregated["prediction"].to_numpy()
                ),
            }
        )
        weekly_error_rows.append(
            panel_scores.groupby("target_week", as_index=False).agg(
                actual_total=("actual", "sum"),
                compact_abs_error=("compact_abs_error", "sum"),
                project_abs_error=("project_abs_error", "sum"),
                daily_abs_error=("daily_abs_error", "sum"),
            )
        )
        print(f"evaluated locked winner on fold {fold.fold_id}", flush=True)

    evaluation = pd.DataFrame(evaluation_rows)
    weekly_errors = pd.concat(weekly_error_rows, ignore_index=True)
    totals = weekly_errors[[
        "actual_total", "compact_abs_error", "project_abs_error", "daily_abs_error"
    ]].sum()

    print("\nDEVELOPMENT_LEADERBOARD (first 3 folds only)")
    print(leaderboard.to_string(index=False))
    print("\nDEVELOPMENT_WEEKLY_CONTEXT (not used for candidate selection)")
    print(development_weekly)
    print(f"\nLOCKED_WINNER\n{winner}")
    print("\nLATER_TIME_EVALUATION (last 4 folds only)")
    print(evaluation.to_string(index=False))
    print("\nPOOLED_EVALUATION_WAPE")
    print(
        {
            "compact_weekly": float(totals["compact_abs_error"] / totals["actual_total"]),
            "project_weekly": float(totals["project_abs_error"] / totals["actual_total"]),
            "tuned_daily_then_sum": float(totals["daily_abs_error"] / totals["actual_total"]),
            "evaluation_weeks": int(len(weekly_errors)),
        }
    )
    print("\nPAIRED_CIRCULAR_BLOCK_BOOTSTRAP (20,000 reps)")
    for block_weeks in [2, 4, 8]:
        print(
            f"daily minus compact weekly ({block_weeks}-week blocks):",
            circular_block_bootstrap(
                weekly_errors,
                "daily_abs_error",
                "compact_abs_error",
                block_weeks=block_weeks,
            ),
        )
        print(
            f"daily minus project weekly ({block_weeks}-week blocks):",
            circular_block_bootstrap(
                weekly_errors,
                "daily_abs_error",
                "project_abs_error",
                block_weeks=block_weeks,
            ),
        )
    print("\nFINAL_HOLDOUT_UNTOUCHED")
    print(
        f"{len(final_test_weeks)} weeks: "
        f"{pd.Timestamp(final_test_weeks[0]).date()} to "
        f"{pd.Timestamp(final_test_weeks[-1]).date()}"
    )


if __name__ == "__main__":
    main()
