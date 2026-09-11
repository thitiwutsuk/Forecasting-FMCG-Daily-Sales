"""Sanity check that Phase 7's Global pooled LightGBM never predicts negative units_sold.

units_sold can't be negative in reality, and the model has no mechanism that forces
predictions to stay >= 0 (no output clipping, no non-negative-friendly objective) --
LightGBM's plain regression objective can in principle predict below zero. This was
previously an informal, unrecorded observation ("hasn't been seen to happen") rather
than a checked guarantee; this test makes it a real, repeatable check against the
project's actual walk-forward CV folds so a future model/feature/data change that
starts producing negative predictions gets caught instead of silently shipping.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models.forecast import ALL_FEATURE_COLS, fit_predict_lgb, prepare_categoricals
from splits.walk_forward import WalkForwardSplitter

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "weekly_features.csv"


def test_lgb_predictions_are_never_negative_across_all_cv_folds():
    df = pd.read_csv(DATA_PATH, parse_dates=["week"])
    dfc = prepare_categoricals(df)
    splitter = WalkForwardSplitter(initial_train_end="2023-09-30", val_fold_weeks=8, final_test_weeks=10)
    folds, _ = splitter.get_folds(dfc)

    for fold in folds:
        train = dfc[dfc.week.isin(fold.train_weeks)]
        val = dfc[dfc.week.isin(fold.val_weeks)]
        pred = fit_predict_lgb(train, val, feature_cols=ALL_FEATURE_COLS)
        assert (pred >= 0).all(), f"fold with val weeks {fold.val_weeks.min()}..{fold.val_weeks.max()} predicted a negative value (min={pred.min()})"
