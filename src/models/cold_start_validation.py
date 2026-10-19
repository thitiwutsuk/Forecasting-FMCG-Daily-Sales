"""Chronological robustness check; no tuning on the five existing test SKUs.

This is a retrospective extension within the SAME simulated dataset, not an
independent external test. Eligibility depends on dates and counts, not outcomes.
"""
import numpy as np
import pandas as pd

from .metrics import score_all
from .zero_history import zero_history_forecast

METHODS = {"Analog no-price": "pred_zero_analog", "Zero-history LightGBM": "pred_zero_lgb",
           "Category mean": "pred_category_mean"}


def chronological_extension(records, reserved_skus, min_train_skus=10):
    if min_train_skus < 1:
        raise ValueError("min_train_skus must be positive")
    history = records[~records.sku.isin(reserved_skus)].copy()
    audit, pieces = [], []
    for sku, rows in records.groupby("sku", sort=True):
        if rows.week.nunique() != 1:
            raise ValueError(f"inconsistent launch window for {sku!r}")
        cutoff = rows.week.iloc[0]
        train = history[(history.sku != sku) & (history.label_available_at <= cutoff)]
        count = train.sku.nunique()
        reason = ("reserved_existing_test" if sku in reserved_skus else
                  "eligible" if count >= min_train_skus else "insufficient_prior_launches")
        audit.append({"sku": sku, "cutoff": cutoff, "prior_train_skus": count,
                      "evaluation_rows": len(rows), "status": reason})
        if reason == "eligible":
            pred = zero_history_forecast(history, [sku], include_category_baseline=True)
            pred["cohort"] = "chronological_extension"
            pieces.append(pred)
    predictions = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    return predictions, pd.DataFrame(audit).sort_values(["cutoff", "sku"]).reset_index(drop=True)


def summarize_predictions(predictions):
    if predictions.empty:
        return pd.DataFrame(), pd.DataFrame()
    common = predictions.dropna(subset=list(METHODS.values()))
    scores, by_sku = [], []
    for model, col in METHODS.items():
        scores.append({"model": model, "scope": "common rows", "n": len(common),
            "skus": common.sku.nunique(), "available_n": len(predictions),
            "coverage": len(common)/len(predictions),
            **score_all(common.launch_units, common[col])})
        for sku, available in predictions.groupby("sku"):
            part = common[common.sku == sku]
            by_sku.append({"sku": sku, "model": model, "n": len(part),
                "available_n": len(available), "coverage": len(part)/len(available),
                **score_all(part.launch_units, part[col])})
    return pd.DataFrame(scores), pd.DataFrame(by_sku)


def sku_bootstrap_differences(predictions, repetitions=2000, seed=42):
    """Paired SKU-cluster bootstrap of WAPE(Category mean) - WAPE(other).

    Negative means baseline lower error. Sample whole SKUs with replacement,
    retaining their channel/region dependence. Exploratory percentile intervals
    condition on these observed SKUs; they do not prove external generalization.
    """
    common = predictions.dropna(subset=list(METHODS.values())).copy()
    if common.sku.nunique() < 2 or repetitions < 1:
        raise ValueError("need at least two SKUs and positive repetitions")
    common["denominator"] = common.launch_units.abs()
    for model, col in METHODS.items():
        common[model] = (common.launch_units - common[col]).abs()
    blocks = common.groupby("sku", sort=True)[["denominator"] + list(METHODS)].sum()
    if blocks.denominator.sum() <= 0:
        raise ValueError("WAPE needs a positive total actual")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(blocks), size=(repetitions, len(blocks)))
    totals = blocks.to_numpy()[indices].sum(axis=1)
    valid = totals[:, 0] > 0
    totals = totals[valid]
    result = []
    base_idx = blocks.columns.get_loc("Category mean")
    for other in ("Analog no-price", "Zero-history LightGBM"):
        idx = blocks.columns.get_loc(other)
        differences = (totals[:, base_idx] - totals[:, idx])/totals[:, 0]
        low, high = np.quantile(differences, [0.025, 0.975])
        point = (blocks["Category mean"].sum() - blocks[other].sum())/blocks.denominator.sum()
        result.append({"comparison": "Category mean minus " + other, "wape_difference": point,
            "ci95_low": low, "ci95_high": high, "skus": len(blocks),
            "repetitions": repetitions, "valid_repetitions": len(totals), "seed": seed})
    return pd.DataFrame(result)
