"""Shared forecast-accuracy metrics (Phase 6+).

WAPE and SMAPE are the primary metrics for this project (see README) because MAPE is
unstable when many sku-channel-region-weeks have low or zero actual sales.
"""

import numpy as np


def mae(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - pred)))


def rmse(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - pred) ** 2)))


def wape(actual: np.ndarray, pred: np.ndarray) -> float:
    denom = np.sum(np.abs(actual))
    if denom == 0:
        return float("nan")
    return float(np.sum(np.abs(actual - pred)) / denom)


def smape(actual: np.ndarray, pred: np.ndarray, eps: float = 1e-8) -> float:
    denom = np.abs(actual) + np.abs(pred) + eps
    return float(np.mean(2 * np.abs(actual - pred) / denom))


def score_all(actual: np.ndarray, pred: np.ndarray) -> dict:
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return {
        "MAE": mae(actual, pred),
        "RMSE": rmse(actual, pred),
        "WAPE": wape(actual, pred),
        "SMAPE": smape(actual, pred),
    }
