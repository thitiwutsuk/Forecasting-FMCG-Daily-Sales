"""Optional, purely additive DL feature-engineering experiment (Phase 11b).

Learns small entity embeddings for sku/channel/region -- the project's own
GROUP_KEYS (see engineer.py) -- using a lightweight PyTorch feed-forward net
(Guo & Berkhahn-style entity embeddings), trained PER FOLD on that fold's
train_weeks ONLY, then exposes the learned vectors as plain numeric feature
columns that can be concatenated onto models.forecast.ALL_FEATURE_COLS and
passed, unmodified, into fit_predict_lgb.

Not imported by any production file (engineer.py, enrich.py, forecast.py).
Used only by notebooks/11b_dl_feature_experiment.ipynb and
tests/test_dl_embeddings.py.
"""

from typing import Dict, List

import pandas as pd
import torch
from torch import nn

# LightGBM (via forecast.py, n_jobs=4) and PyTorch each bundle their own OpenMP
# runtime; running both multi-threaded in the same process causes severe thread
# contention on macOS (observed: a fold's embedding fit going from <1s to a
# multi-minute stall right after a preceding LightGBM fit in the same process).
# The embedding net here is tiny (a few thousand parameters) -- single-threaded
# CPU execution costs nothing and removes the contention entirely.
torch.set_num_threads(1)

ENTITY_COLS: List[str] = ["sku", "channel", "region"]
EMBED_DIM: Dict[str, int] = {"sku": 4, "channel": 2, "region": 2}
# Deliberately narrow context: just enough signal for the net to shape useful
# embeddings, not a copy of ALL_FEATURE_COLS.
NUMERIC_CONTEXT_COLS: List[str] = ["lag_1", "rolling_mean_4", "week_number", "promotion_flag"]
TARGET = "target_next_week"


class _EntityEmbedNet(nn.Module):
    def __init__(self, vocab_sizes: Dict[str, int], embed_dim: Dict[str, int], n_numeric: int, hidden: int = 16):
        super().__init__()
        self.embeddings = nn.ModuleDict({
            col: nn.Embedding(vocab_sizes[col], embed_dim[col]) for col in vocab_sizes
        })
        total_embed = sum(embed_dim.values())
        self.mlp = nn.Sequential(
            nn.Linear(total_embed + n_numeric, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, cat_idx: Dict[str, torch.Tensor], numeric: torch.Tensor) -> torch.Tensor:
        parts = [self.embeddings[col](cat_idx[col]) for col in self.embeddings]
        x = torch.cat(parts + [numeric], dim=1)
        return self.mlp(x).squeeze(-1)


def _build_vocab(train_df: pd.DataFrame, col: str) -> Dict[str, int]:
    """value -> int index, built ONLY from values observed in train_df (fold-local vocabulary)."""
    values = sorted(train_df[col].astype(str).unique())
    return {v: i for i, v in enumerate(values)}


def fit_fold_embeddings(
    train_df: pd.DataFrame,
    entity_cols: List[str] = ENTITY_COLS,
    embed_dim: Dict[str, int] = EMBED_DIM,
    numeric_cols: List[str] = NUMERIC_CONTEXT_COLS,
    epochs: int = 30,
    hidden: int = 16,
    lr: float = 1e-2,
    weight_decay: float = 1e-4,
    seed: int = 42,
) -> Dict[str, pd.DataFrame]:
    """Train ONE small embedding network on train_df ONLY (a single fold's train_weeks).

    LEAKAGE GUARD: this function's signature takes only train_df -- there is no
    parameter through which val_weeks rows, or any full-dataset statistic, can
    enter fitting. Vocabularies (_build_vocab) and network weights are both
    derived exclusively from train_df.

    Returns {entity_col: DataFrame[entity_col, f"{entity_col}_emb_0", ...]} -- one
    small lookup table per entity, keyed by entity VALUE (e.g. sku id), not by week.
    """
    torch.manual_seed(seed)
    vocabs = {c: _build_vocab(train_df, c) for c in entity_cols}
    vocab_sizes = {c: len(v) for c, v in vocabs.items()}

    cat_idx = {
        c: torch.tensor(train_df[c].astype(str).map(vocabs[c]).values, dtype=torch.long)
        for c in entity_cols
    }
    numeric = train_df[numeric_cols].fillna(0.0).values.astype("float32")
    numeric_mean, numeric_std = numeric.mean(0), numeric.std(0) + 1e-6
    numeric_t = torch.tensor((numeric - numeric_mean) / numeric_std, dtype=torch.float32)

    y = train_df[TARGET].values.astype("float32")
    y_mean, y_std = y.mean(), y.std() + 1e-6
    y_t = torch.tensor((y - y_mean) / y_std, dtype=torch.float32)

    net = _EntityEmbedNet(vocab_sizes, embed_dim, n_numeric=len(numeric_cols), hidden=hidden)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    net.train()
    for _ in range(epochs):
        opt.zero_grad()
        pred = net(cat_idx, numeric_t)
        loss = loss_fn(pred, y_t)
        loss.backward()
        opt.step()

    tables = {}
    net.eval()
    with torch.no_grad():
        for c in entity_cols:
            vecs = net.embeddings[c].weight.detach().numpy()
            cols = [f"{c}_emb_{i}" for i in range(embed_dim[c])]
            table = pd.DataFrame(vecs, columns=cols)
            table[c] = list(vocabs[c].keys())
            tables[c] = table[[c] + cols]
    return tables


def transform_with_fold_embeddings(
    df: pd.DataFrame,
    embedding_tables: Dict[str, pd.DataFrame],
    entity_cols: List[str] = ENTITY_COLS,
) -> pd.DataFrame:
    """Left-merge each entity's fold-fitted embedding table onto df by entity VALUE.

    Safe to call on train_df OR val_df of the SAME fold: merge keys are entity
    identities (e.g. sku="SKU_014"), never weeks or the target -- a val row's own
    outcome or week never participates. Unseen entity values fall back to that
    column's training-mean embedding, never NaN. Returns a copy; never mutates
    df in place.
    """
    out = df.copy()
    for c in entity_cols:
        table = embedding_tables[c]
        emb_cols = [col for col in table.columns if col != c]
        out = out.merge(table, on=c, how="left")
        missing = out[emb_cols[0]].isna()
        if missing.any():
            fallback = table[emb_cols].mean()
            out.loc[missing, emb_cols] = fallback.values
    return out


def dl_embedding_cols(entity_cols: List[str] = ENTITY_COLS, embed_dim: Dict[str, int] = EMBED_DIM) -> List[str]:
    """Flat list of new column names, e.g. ['sku_emb_0', ..., 'region_emb_1'] --
    for building ALL_FEATURE_COLS + dl_embedding_cols() in the notebook."""
    return [f"{c}_emb_{i}" for c in entity_cols for i in range(embed_dim[c])]
