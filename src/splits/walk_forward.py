"""Panel-aware, time-respecting walk-forward CV (Phase 5).

The data is a panel (sku x channel x region x week): splitting must happen on `week`,
never on row index, so every sku-channel-region cell for a given week lands in the
same side of the split. Splitting by row index would let some cells from week W train
while others from the same week validate, which isn't a real forecasting scenario and
would let the model implicitly see same-week information across the split.

Scheme:
- Initial training window: the first weeks up to `initial_train_end`.
- Expanding walk-forward validation folds: each fold's training window grows to include
  everything before it (never shrinks or slides), and validates on the next
  `val_fold_weeks` weeks — mirrors how a model would actually be retrained and
  re-validated as more weeks become available in production.
- Final holdout test: the last `final_test_weeks` weeks, never touched by CV at all —
  used exactly once, in Phase 13.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Fold:
    fold_id: int
    train_weeks: np.ndarray
    val_weeks: np.ndarray


def get_sorted_weeks(df: pd.DataFrame, week_col: str = "week") -> np.ndarray:
    return np.sort(pd.to_datetime(df[week_col]).unique())


def make_walk_forward_folds(
    weeks: np.ndarray,
    initial_train_end: str,
    val_fold_weeks: int = 8,
    final_test_weeks: int = 10,
) -> tuple:
    """Build expanding walk-forward CV folds plus a final untouched test window.

    Returns (folds, final_test_weeks_array). `weeks` must be sorted, unique week
    timestamps covering the whole dataset. Everything from `final_test_weeks` weeks
    before the end of `weeks` onward is excluded from every CV fold.
    """
    weeks = np.sort(np.unique(weeks))
    initial_train_end = pd.Timestamp(initial_train_end)

    holdout_start_idx = len(weeks) - final_test_weeks
    if holdout_start_idx <= 0:
        raise ValueError("final_test_weeks is larger than the available week range")
    cv_weeks = weeks[:holdout_start_idx]
    final_test = weeks[holdout_start_idx:]

    initial_train_mask = cv_weeks <= initial_train_end
    if not initial_train_mask.any():
        raise ValueError("initial_train_end is before the first available week")
    train_end_idx = int(np.sum(initial_train_mask))

    folds = []
    fold_id = 0
    cursor = train_end_idx
    while cursor < len(cv_weeks):
        val_end = min(cursor + val_fold_weeks, len(cv_weeks))
        folds.append(
            Fold(
                fold_id=fold_id,
                train_weeks=cv_weeks[:cursor],
                val_weeks=cv_weeks[cursor:val_end],
            )
        )
        fold_id += 1
        cursor = val_end

    return folds, final_test


class WalkForwardSplitter:
    """sklearn-style splitter: split(df) yields (train_idx, val_idx) row-index arrays.

    Row membership follows week membership only — a whole sku x channel x region x
    week panel slice moves together, never split across train/val.
    """

    def __init__(self, initial_train_end: str, val_fold_weeks: int = 8, final_test_weeks: int = 10):
        self.initial_train_end = initial_train_end
        self.val_fold_weeks = val_fold_weeks
        self.final_test_weeks = final_test_weeks

    def get_folds(self, df: pd.DataFrame, week_col: str = "week"):
        weeks = get_sorted_weeks(df, week_col)
        return make_walk_forward_folds(weeks, self.initial_train_end, self.val_fold_weeks, self.final_test_weeks)

    def split(self, df: pd.DataFrame, week_col: str = "week"):
        folds, _ = self.get_folds(df, week_col)
        week_series = pd.to_datetime(df[week_col])
        for fold in folds:
            train_idx = df.index[week_series.isin(fold.train_weeks)].to_numpy()
            val_idx = df.index[week_series.isin(fold.val_weeks)].to_numpy()
            yield train_idx, val_idx

    def final_test_split(self, df: pd.DataFrame, week_col: str = "week"):
        """Row-index split for the untouched final holdout: (cv_idx, test_idx)."""
        _, final_test_weeks = self.get_folds(df, week_col)
        week_series = pd.to_datetime(df[week_col])
        test_idx = df.index[week_series.isin(final_test_weeks)].to_numpy()
        cv_idx = df.index[~week_series.isin(final_test_weeks)].to_numpy()
        return cv_idx, test_idx
