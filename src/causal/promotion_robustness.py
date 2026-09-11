"""Robustness-audit helpers for the Phase 8 promotion TWFE estimate.

Phase 8 (src/causal/promotion_effect.py) fits SKU-level fixed effects. The
heterogeneity-robust diagnostics recommended in LITERATURE_GROUNDING.md Sec3a --
de Chaisemartin & D'Haultfoeuille (2020)'s negative-weights check and its
`DID_M`/WAS alternative estimator -- need a finer entity unit: group = (sku,
channel, region), since `promotion_flag` varies at that level, not just by sku.
This module builds that panel and refits a parity TWFE on it, so any difference
found by the heterogeneity-robust tools reflects the robustness check itself,
not a change in fixed-effects granularity.
"""

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS


def build_group_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Adds group_id (sku|channel|region), a sequential time_id, and log_units.

    Validates the preconditions the downstream heterogeneity-robust estimators
    assume: exactly one row per (group_id, time_id), a binary treatment, and no
    internal gaps in any group's observed time range (staggered launches are
    fine -- a group's window can start late -- but it can't have a hole once
    started).
    """
    out = df.copy()
    out["group_id"] = out["sku"] + "|" + out["channel"] + "|" + out["region"]
    all_weeks = sorted(out["week"].unique())
    week_to_id = {w: i for i, w in enumerate(all_weeks)}
    out["time_id"] = out["week"].map(week_to_id)
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


def export_group_panel_csv(panel: pd.DataFrame, path: str) -> None:
    """Writes the columns the R TwoWayFEWeights diagnostic (analysis/r/) needs."""
    cols = ["group_id", "time_id", "log_units", "promotion_flag", "price_unit", "sku"]
    panel[cols].to_csv(path, index=False)
