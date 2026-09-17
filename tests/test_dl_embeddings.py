"""Correctness/leakage checks for the optional DL embedding experiment (Phase 11b).

dl_embeddings.py is never imported by production code (engineer.py, enrich.py,
forecast.py) -- it exists purely for notebooks/11b_dl_feature_experiment.ipynb.
These tests exist because the whole point of that notebook is a leakage-safe
comparison against the project's existing walk-forward CV: if fit_fold_embeddings
ever let val-period information leak into the learned embeddings, the notebook's
WAPE comparison would be meaningless without anyone noticing.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from features.dl_embeddings import (
    ENTITY_COLS,
    dl_embedding_cols,
    fit_fold_embeddings,
    transform_with_fold_embeddings,
)
from models.forecast import ALL_FEATURE_COLS, fit_predict_lgb, prepare_categoricals


def _toy_panel(n_weeks=10, skus=("A", "B", "C"), seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    weeks = pd.date_range("2023-01-01", periods=n_weeks, freq="W")
    for sku in skus:
        for channel in ["Retail", "Online"]:
            for region in ["North", "South"]:
                for w in weeks:
                    rows.append({
                        "sku": sku, "channel": channel, "region": region, "week": w,
                        "lag_1": rng.normal(50, 5), "rolling_mean_4": rng.normal(50, 5),
                        "week_number": w.isocalendar()[1], "promotion_flag": int(rng.random() < 0.3),
                        "target_next_week": rng.normal(50, 5),
                    })
    return pd.DataFrame(rows)


def test_fit_fold_embeddings_shapes_and_no_nan():
    df = _toy_panel()
    tables = fit_fold_embeddings(df, epochs=5)
    assert set(tables) == set(ENTITY_COLS)
    for col, table in tables.items():
        assert table[col].nunique() == df[col].nunique()
        emb_cols = [c for c in table.columns if c != col]
        assert not table[emb_cols].isna().any().any()


def test_transform_adds_expected_columns_without_mutating_input():
    df = _toy_panel()
    tables = fit_fold_embeddings(df, epochs=5)
    before_cols = list(df.columns)
    out = transform_with_fold_embeddings(df, tables)

    assert list(df.columns) == before_cols
    for col in dl_embedding_cols():
        assert col in out.columns
    assert not out[dl_embedding_cols()].isna().any().any()
    assert len(out) == len(df)


def test_embeddings_do_not_depend_on_val_period_rows():
    df = _toy_panel(n_weeks=10)
    weeks = sorted(df.week.unique())
    train_weeks, val_weeks = weeks[:6], weeks[6:]
    train_df = df[df.week.isin(train_weeks)]

    tables_a = fit_fold_embeddings(train_df, epochs=5, seed=1)

    # Mutate ONLY the val-period rows' target/features; train rows are untouched.
    mutated = df.copy()
    mutated.loc[mutated.week.isin(val_weeks), "target_next_week"] += 1000.0
    mutated.loc[mutated.week.isin(val_weeks), "lag_1"] += 1000.0
    train_df_mutated = mutated[mutated.week.isin(train_weeks)]

    tables_b = fit_fold_embeddings(train_df_mutated, epochs=5, seed=1)

    for col in ENTITY_COLS:
        pd.testing.assert_frame_equal(
            tables_a[col].sort_values(col).reset_index(drop=True),
            tables_b[col].sort_values(col).reset_index(drop=True),
        )


def test_extended_feature_cols_work_end_to_end_with_unmodified_fit_predict_lgb():
    df = _toy_panel(n_weeks=10)
    dfc = prepare_categoricals(df)
    weeks = sorted(dfc.week.unique())
    train, val = dfc[dfc.week.isin(weeks[:6])], dfc[dfc.week.isin(weeks[6:])]

    tables = fit_fold_embeddings(train, epochs=5)
    train_ext = transform_with_fold_embeddings(train, tables)
    val_ext = transform_with_fold_embeddings(val, tables)

    base_cols = [c for c in ALL_FEATURE_COLS if c in train_ext.columns]
    extended_cols = base_cols + dl_embedding_cols()

    preds = fit_predict_lgb(train_ext, val_ext, feature_cols=extended_cols)
    assert len(preds) == len(val_ext)
    assert not np.isnan(preds).any()
