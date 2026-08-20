"""Formal data-validation checks for the FMCG forecasting pipeline (Phase 3).

Covers three things: schema/range sanity checks, duplicate-key checks, and
leakage checks that confirm every lag/rolling/target feature in the weekly
modeling table is derived only from past weeks within its own
sku-channel-region group.
"""

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str
    n_failures: int = 0


@dataclass
class ValidationReport:
    results: list = field(default_factory=list)

    def add(self, result: CheckResult) -> None:
        self.results.append(result)

    @property
    def all_passed(self) -> bool:
        return all(r.passed for r in self.results)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [{"check": r.name, "passed": r.passed, "n_failures": r.n_failures, "detail": r.detail} for r in self.results]
        )


GROUP_KEYS = ["sku", "channel", "region"]


def check_no_duplicate_keys(df: pd.DataFrame, keys: list) -> CheckResult:
    n_dupes = int(df.duplicated(subset=keys).sum())
    return CheckResult(
        name=f"no duplicate {'-'.join(keys)} rows",
        passed=n_dupes == 0,
        detail=f"{n_dupes} duplicate row(s) found" if n_dupes else "no duplicates",
        n_failures=n_dupes,
    )


def check_no_negative(df: pd.DataFrame, column: str) -> CheckResult:
    n_bad = int((df[column] < 0).sum())
    return CheckResult(
        name=f"{column} >= 0",
        passed=n_bad == 0,
        detail=f"{n_bad} row(s) with negative {column}" if n_bad else "no negative values",
        n_failures=n_bad,
    )


def check_positive(df: pd.DataFrame, column: str) -> CheckResult:
    n_bad = int((df[column] <= 0).sum())
    return CheckResult(
        name=f"{column} > 0",
        passed=n_bad == 0,
        detail=f"{n_bad} row(s) with non-positive {column}" if n_bad else "all values positive",
        n_failures=n_bad,
    )


def _weekly_rollup(daily_group: pd.DataFrame) -> pd.Series:
    """Reproduce the weekly units_sold roll-up convention used by weekly_df_final_for_modeling.csv."""
    s = daily_group.set_index("date").sort_index()["units_sold"]
    return s.resample("W-MON", label="left", closed="left").sum()


def check_weekly_rollup(daily: pd.DataFrame, weekly: pd.DataFrame) -> CheckResult:
    """Recompute units_sold per sku-channel-region-week from the daily file and diff against weekly file."""
    n_compared = 0
    n_mismatch = 0
    for keys, g in daily.groupby(GROUP_KEYS):
        recomputed = _weekly_rollup(g)
        recomputed.index.name = "week"
        existing = weekly[
            (weekly.sku == keys[0]) & (weekly.channel == keys[1]) & (weekly.region == keys[2])
        ].set_index("week")["units_sold"]
        common = recomputed.index.intersection(existing.index)
        n_compared += len(common)
        n_mismatch += int((recomputed.loc[common] != existing.loc[common]).sum())
    return CheckResult(
        name="daily->weekly roll-up convention (W-MON, label=left, closed=left)",
        passed=n_mismatch == 0,
        detail=f"{n_mismatch}/{n_compared} weeks mismatched",
        n_failures=n_mismatch,
    )


def check_target_leakage(weekly: pd.DataFrame) -> CheckResult:
    """target_next_week must equal units_sold of the next week within the same group, never from another group."""
    sorted_df = weekly.sort_values(GROUP_KEYS + ["week"]).copy()
    sorted_df["next_actual"] = sorted_df.groupby(GROUP_KEYS)["units_sold"].shift(-1)
    comparable = sorted_df.dropna(subset=["next_actual", "target_next_week"])
    n_mismatch = int((comparable["target_next_week"] != comparable["next_actual"]).sum())
    return CheckResult(
        name="target_next_week == next week's units_sold (same group)",
        passed=n_mismatch == 0,
        detail=f"{n_mismatch}/{len(comparable)} rows mismatched",
        n_failures=n_mismatch,
    )


def check_lag_and_rolling_leakage(daily: pd.DataFrame, weekly: pd.DataFrame) -> list:
    """lag_1, lag_2, rolling_mean_4, rolling_std_4, momentum must use only weeks strictly before the current row."""
    checks = {"lag_1": 0, "lag_2": 0, "rolling_mean_4": 0, "rolling_std_4": 0, "momentum": 0}
    n_compared = 0

    for keys, g in daily.groupby(GROUP_KEYS):
        weekly_full = _weekly_rollup(g)
        weekly_full.index.name = "week"

        lag1 = weekly_full.shift(1)
        lag2 = weekly_full.shift(2)
        roll_mean = weekly_full.shift(1).rolling(4).mean()
        roll_std = weekly_full.shift(1).rolling(4).std()
        momentum = lag1 - lag2

        existing = weekly[
            (weekly.sku == keys[0]) & (weekly.channel == keys[1]) & (weekly.region == keys[2])
        ].set_index("week")
        common = existing.index.intersection(lag1.index)
        n_compared += len(common)

        checks["lag_1"] += int((existing.loc[common, "lag_1"].round(3) != lag1.loc[common].round(3)).sum())
        checks["lag_2"] += int((existing.loc[common, "lag_2"].round(3) != lag2.loc[common].round(3)).sum())
        checks["rolling_mean_4"] += int(
            (existing.loc[common, "rolling_mean_4"] - roll_mean.loc[common]).abs().gt(1e-6).sum()
        )
        checks["rolling_std_4"] += int(
            (existing.loc[common, "rolling_std_4"] - roll_std.loc[common]).abs().gt(1e-6).sum()
        )
        checks["momentum"] += int((existing.loc[common, "momentum"] - momentum.loc[common]).abs().gt(1e-6).sum())

    return [
        CheckResult(
            name=f"{col} derived only from past weeks (no leakage)",
            passed=n_bad == 0,
            detail=f"{n_bad}/{n_compared} rows mismatched vs. shift-based recomputation",
            n_failures=n_bad,
        )
        for col, n_bad in checks.items()
    ]


def run_all_checks(daily: pd.DataFrame, weekly: pd.DataFrame) -> ValidationReport:
    report = ValidationReport()

    report.add(check_no_duplicate_keys(daily, GROUP_KEYS + ["date"]))
    report.add(check_no_duplicate_keys(weekly, GROUP_KEYS + ["week"]))

    report.add(check_no_negative(daily, "units_sold"))
    report.add(check_no_negative(daily, "stock_available"))
    report.add(check_no_negative(daily, "delivered_qty"))
    report.add(check_positive(daily, "price_unit"))

    report.add(check_no_negative(weekly, "units_sold"))
    report.add(check_no_negative(weekly, "stock_available"))
    report.add(check_positive(weekly, "price_unit"))
    report.add(check_no_negative(weekly, "sku_age"))

    report.add(check_weekly_rollup(daily, weekly))
    report.add(check_target_leakage(weekly))
    for result in check_lag_and_rolling_leakage(daily, weekly):
        report.add(result)

    return report
