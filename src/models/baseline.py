"""Baseline forecasters for target_next_week (Phase 6).

Each function returns predictions for `target_next_week` computed only from
`units_sold` within the same sku-channel-region group, recomputed independently of
any precomputed lag columns so the baseline is self-contained and trivially auditable.
"""

import numpy as np
import pandas as pd

GROUP_KEYS = ["sku", "channel", "region"]


def _sorted(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(GROUP_KEYS + ["week"])


def naive_forecast(df: pd.DataFrame) -> pd.Series:
    """Persistence: next week's forecast = this week's actual units_sold."""
    return _sorted(df)["units_sold"]


def seasonal_naive_forecast(df: pd.DataFrame) -> pd.Series:
    """Forecast = units_sold from 52 weeks earlier in the same group (NaN if unavailable)."""
    s = _sorted(df)
    return s.groupby(GROUP_KEYS)["units_sold"].shift(52)


def moving_average_forecast(df: pd.DataFrame, window: int = 4) -> pd.Series:
    """Forecast = mean of the most recent `window` completed weeks, including the current one."""
    s = _sorted(df)
    return s.groupby(GROUP_KEYS)["units_sold"].transform(lambda x: x.rolling(window).mean())


def add_baseline_predictions(df: pd.DataFrame) -> pd.DataFrame:
    out = _sorted(df).reset_index(drop=True)
    out["pred_naive"] = naive_forecast(out).reset_index(drop=True)
    out["pred_seasonal_naive"] = seasonal_naive_forecast(out).reset_index(drop=True)
    out["pred_moving_avg_4"] = moving_average_forecast(out).reset_index(drop=True)
    return out
