"""Robustness-audit helpers for the Phase 8 promotion TWFE estimate.

Phase 8 (src/causal/promotion_effect.py) fits SKU-level fixed effects. The
heterogeneity-robust diagnostics recommended in LITERATURE_GROUNDING.md Sec3a --
de Chaisemartin & D'Haultfoeuille (2020)'s negative-weights check and its
`DID_M`/WAS alternative estimator -- need a finer entity unit: group = (sku,
channel, region), since `promotion_flag` varies at that level, not just by sku.
This module builds that panel, refits a parity TWFE, and computes the treated-cell
TWFE weights directly from the residualized treatment.  The latter is also the
calculation performed by `TwoWayFEWeights` for an unweighted feTR regression.
"""

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS


def build_group_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Adds numeric group/time/cluster IDs, a readable group label, and log_units.

    Validates the preconditions the downstream heterogeneity-robust estimators
    assume: exactly one row per (group_id, time_id), a binary treatment, and no
    internal gaps in any group's observed time range (staggered launches are
    fine -- a group's window can start late -- but it can't have a hole once
    started).  IDs are deliberately numeric: `did-multiplegt-stat` coerces its
    ID column to numeric internally, so passing the former string label silently
    converted every ID to missing and produced a zero-weight error.
    """
    out = df.copy()
    required = {"sku", "channel", "region", "week", "units_sold", "promotion_flag"}
    missing = required.difference(out.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    if out[list(required)].isna().any().any():
        raise ValueError("required panel columns must not contain missing values")

    out["group_label"] = out["sku"] + "|" + out["channel"] + "|" + out["region"]
    out["group_id"] = pd.factorize(out["group_label"], sort=True)[0] + 1
    out["cluster_id"] = pd.factorize(out["sku"], sort=True)[0] + 1
    all_weeks = sorted(out["week"].unique())
    week_to_id = {w: i for i, w in enumerate(all_weeks)}
    out["time_id"] = out["week"].map(week_to_id).astype("int64")
    out["log_units"] = np.log1p(out["units_sold"])

    if out.duplicated(subset=["group_id", "time_id"]).any():
        raise ValueError("duplicate (group_id, time_id) rows -- estimators need exactly one row per cell")
    if not set(out["promotion_flag"].unique()) <= {0, 1}:
        raise ValueError("promotion_flag must be binary for these estimators")
    for gid, sub in out.groupby("group_id"):
        tids = sorted(sub["time_id"].unique())
        if set(range(tids[0], tids[-1] + 1)) != set(tids):
            raise ValueError(f"internal time gap in group {gid!r}")

    return out


def fit_parity_twfe(panel: pd.DataFrame):
    """Refits the promotion TWFE with group_id (sku x channel x region) fixed
    effects instead of Phase 8's sku-level entity FE, clustered by sku.

    This is the apples-to-apples comparison basis for the heterogeneity-robust
    tools, which operate on group_id as the unit. Confirms the finer FE
    granularity doesn't itself move the estimate before attributing any
    difference to heterogeneity-robustness.
    """
    indexed = panel.set_index(["group_id", "time_id"])
    model = PanelOLS(
        indexed["log_units"],
        indexed[["promotion_flag", "price_unit"]].astype(float),
        entity_effects=True,
        time_effects=True,
    )
    clusters = indexed["sku"]
    return model.fit(cov_type="clustered", clusters=clusters)


def compute_twfe_weight_diagnostic(
    panel: pd.DataFrame,
    controls: tuple[str, ...] = ("price_unit",),
) -> tuple[dict, pd.DataFrame]:
    """Decompose the TWFE coefficient into weights on treated group-time cells.

    For an unweighted regression, Frisch-Waugh-Lovell gives each treated cell
    weight ``residualized_D / sum(D * residualized_D)``, where treatment is
    residualized on the group effects, time effects, and supplied controls.
    The returned reconstruction is an algebraic parity check against the fitted
    TWFE coefficient; it does not add causal identification by itself.
    """
    required = {"group_id", "time_id", "promotion_flag", "log_units", *controls}
    missing = required.difference(panel.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    if panel.duplicated(["group_id", "time_id"]).any():
        raise ValueError("duplicate (group_id, time_id) rows")
    if not pd.api.types.is_numeric_dtype(panel["group_id"]):
        raise ValueError("group_id must be numeric")

    indexed = panel.set_index(["group_id", "time_id"]).sort_index()
    treatment = indexed["promotion_flag"].astype(float)
    if controls:
        exog = indexed[list(controls)].astype(float)
    else:
        exog = pd.DataFrame({"constant": 1.0}, index=indexed.index)

    treatment_model = PanelOLS(
        treatment,
        exog,
        entity_effects=True,
        time_effects=True,
        drop_absorbed=True,
    ).fit()
    residualized_treatment = treatment_model.resids.rename("residualized_treatment")
    denominator = float((residualized_treatment * treatment).sum())
    if np.isclose(denominator, 0.0):
        raise ValueError("treatment has no residual variation after fixed effects and controls")

    detail = indexed[["promotion_flag", "log_units"]].copy()
    detail["residualized_treatment"] = residualized_treatment
    detail["twfe_weight"] = np.where(
        treatment.eq(1), residualized_treatment / denominator, 0.0
    )
    detail = detail.reset_index()

    treated_weights = detail.loc[detail["promotion_flag"].eq(1), "twfe_weight"]
    negative = treated_weights.lt(0)
    reconstructed = float(
        (residualized_treatment * indexed["log_units"].astype(float)).sum() / denominator
    )
    summary = {
        "n_obs": int(len(detail)),
        "n_treated_cells": int(len(treated_weights)),
        "n_positive_weights": int(treated_weights.gt(0).sum()),
        "n_negative_weights": int(negative.sum()),
        "negative_weight_pct": float(negative.mean() * 100),
        "sum_positive_weights": float(treated_weights[treated_weights.gt(0)].sum()),
        "sum_negative_weights": float(treated_weights[negative].sum()),
        "sum_weights": float(treated_weights.sum()),
        "min_weight": float(treated_weights.min()),
        "max_weight": float(treated_weights.max()),
        "reconstructed_twfe_coef": reconstructed,
    }
    return summary, detail


def build_promotion_lead_lag_panel(
    panel: pd.DataFrame,
    lags: int = 4,
    leads: int = 2,
) -> tuple[pd.DataFrame, list[str]]:
    """Add within-group promotion leads/lags for a recurring-treatment audit.

    Boundary rows without the requested history/future are removed. Because
    shifts are calculated separately within each group, values can never leak
    across SKU-channel-region series.
    """
    if not isinstance(lags, int) or isinstance(lags, bool) or lags < 0:
        raise ValueError("lags must be a non-negative integer")
    if not isinstance(leads, int) or isinstance(leads, bool) or leads < 0:
        raise ValueError("leads must be a non-negative integer")
    required = {"group_id", "time_id", "promotion_flag"}
    missing = required.difference(panel.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    if panel.duplicated(["group_id", "time_id"]).any():
        raise ValueError("duplicate (group_id, time_id) rows")

    out = panel.sort_values(["group_id", "time_id"]).copy()
    treatment_by_group = out.groupby("group_id", sort=False)["promotion_flag"]
    out["promo_current"] = out["promotion_flag"].astype(float)
    terms = ["promo_current"]
    for offset in range(1, lags + 1):
        name = f"promo_lag_{offset}"
        out[name] = treatment_by_group.shift(offset)
        terms.append(name)
    for offset in range(1, leads + 1):
        name = f"promo_lead_{offset}"
        out[name] = treatment_by_group.shift(-offset)
        terms.append(name)
    return out.dropna(subset=terms).copy(), terms


def fit_promotion_distributed_lag(
    panel: pd.DataFrame,
    lags: int = 4,
    leads: int = 2,
    controls: tuple[str, ...] = (),
):
    """Fit a supplementary recurring-promotion lead/lag TWFE specification.

    Lag coefficients test for post-promotion displacement conditional on the
    current promotion state. Lead coefficients are placebo checks. Interpreting
    either causally requires strict/sequential exogeneity and stable additive
    lag effects, so this is a sensitivity analysis rather than a replacement
    for a fully validated dynamic DiD estimator.
    """
    design, treatment_terms = build_promotion_lead_lag_panel(panel, lags=lags, leads=leads)
    missing = set(controls).difference(design.columns)
    if missing:
        raise ValueError(f"missing control columns: {sorted(missing)}")
    indexed = design.set_index(["group_id", "time_id"])
    exog_cols = treatment_terms + list(controls)
    model = PanelOLS(
        indexed["log_units"].astype(float),
        indexed[exog_cols].astype(float),
        entity_effects=True,
        time_effects=True,
    )
    return model.fit(cov_type="clustered", clusters=indexed["sku"])


def export_group_panel_csv(panel: pd.DataFrame, path: str) -> None:
    """Writes the columns the R TwoWayFEWeights diagnostic (analysis/r/) needs."""
    cols = [
        "group_id", "time_id", "log_units", "promotion_flag", "price_unit",
        "cluster_id", "group_label", "sku",
    ]
    panel[cols].to_csv(path, index=False)
