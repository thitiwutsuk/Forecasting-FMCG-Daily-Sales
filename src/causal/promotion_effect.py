"""Two-way fixed-effects promotion-uplift estimator (Phase 8, Tier 2 causal analysis).

Specification: log(1 + units_sold) ~ promotion_flag + price_unit + channel + region,
with SKU entity fixed effects and week time fixed effects, clustered SEs by SKU.

Entity FE absorbs any fixed unobserved SKU characteristic (brand strength, base
demand level); time FE absorbs any common week-level shock (macro, week-in-year
seasonality shared across all SKUs). The promotion coefficient is a log-linear
effect: % uplift = exp(coef) - 1.
"""

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS


def _design_matrix(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.get_dummies(df[["channel", "region"]], drop_first=True).astype(float)
    X["price_unit"] = df["price_unit"].astype(float)
    X["promotion_flag"] = df["promotion_flag"].astype(float)
    return X


def fit_promotion_effect(df: pd.DataFrame, entity_effects: bool = True, time_effects: bool = True):
    """df must have a MultiIndex (sku, week). Returns the fitted PanelOLS result.

    Clustered-by-entity SEs require more than 1 cluster to be well-defined; with a
    single SKU (e.g. the Juice category) they degenerate to a singular covariance
    matrix, so we fall back to heteroskedasticity-robust SEs in that case.
    """
    y = np.log1p(df["units_sold"])
    X = _design_matrix(df)
    model = PanelOLS(y, X, entity_effects=entity_effects, time_effects=time_effects)
    n_entities = df.index.get_level_values(0).nunique()
    if n_entities > 1:
        return model.fit(cov_type="clustered", cluster_entity=True)
    return model.fit(cov_type="robust")


def uplift_summary(result) -> dict:
    """Convert the log-linear promotion_flag coefficient to a % uplift with CI."""
    coef = result.params["promotion_flag"]
    lo, hi = result.conf_int().loc["promotion_flag"]
    return {
        "uplift_pct": float((np.expm1(coef)) * 100),
        "ci_low_pct": float((np.expm1(lo)) * 100),
        "ci_high_pct": float((np.expm1(hi)) * 100),
        "p_value": float(result.pvalues["promotion_flag"]),
        "n_obs": int(result.nobs),
    }


def fit_by_category(df: pd.DataFrame, category_col: str = "category") -> pd.DataFrame:
    """Fit one FE regression per category; falls back to time-effects-only when a
    category has just 1 SKU (entity FE is not separately identified with 1 entity).
    """
    rows = []
    for cat, sub in df.groupby(category_col):
        n_sku = sub["sku"].nunique()
        panel = sub.set_index(["sku", "week"])
        use_entity_fe = n_sku > 1
        try:
            res = fit_promotion_effect(panel, entity_effects=use_entity_fe, time_effects=True)
            summary = uplift_summary(res)
            summary["category"] = cat
            summary["n_sku"] = n_sku
            summary["entity_fe"] = use_entity_fe
            rows.append(summary)
        except Exception as e:
            rows.append({"category": cat, "n_sku": n_sku, "entity_fe": use_entity_fe, "error": str(e)})
    return pd.DataFrame(rows)
