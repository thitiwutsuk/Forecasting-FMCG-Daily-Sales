"""Phase 12 rollup. Fresh CV and recorded experiments remain separate protocols.

One ledger contains only newly generated CV predictions. Reports and plots are
derived from it, without fitting a model again. LightGBM/XGBoost are the focus;
other core models, ablations and embeddings provide supporting comparisons.
"""

import ast
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import score_all

KEYS = ["sku", "channel", "region", "week"]
METRICS = ["MAE", "RMSE", "WAPE", "SMAPE"]
FOCUS = ["global_lgb", "global_xgb"]
CORE = {
    "naive": ("Naive", 6),
    "seasonal_naive": ("Seasonal Naive", 6),
    "moving_avg_4": ("Moving Average (4w)", 6),
    "local_lgb": ("Local per-SKU LightGBM", 7),
    "global_lgb": ("Global LightGBM", 7),
    "global_xgb": ("Global XGBoost", 7),
    "global_rf": ("Global Random Forest", 7),
}
SPLIT_CONFIG = dict(initial_train_end="2023-09-30", val_fold_weeks=8, final_test_weeks=10)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _text(value):
    return "".join(value) if isinstance(value, list) else value


def load_feature_groups(root, all_features):
    """Read the literal Phase 11 partition, never execute an old notebook."""
    path = Path(root) / "notebooks/11_feature_ablation.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        for node in ast.parse(_text(cell["source"])).body:
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "FEATURE_GROUPS" for t in node.targets
            ):
                groups = ast.literal_eval(node.value)
                flat = [c for cols in groups.values() for c in cols]
                ids = {"sku", "channel", "region", "category", "segment", "brand"}
                if len(groups) != 6 or len(flat) != len(set(flat)) or ids & set(flat):
                    raise ValueError("invalid Phase 11 feature partition")
                if set(flat) | ids != set(all_features):
                    raise ValueError("Phase 11 partition no longer covers ALL_FEATURE_COLS")
                return groups
    raise ValueError(f"FEATURE_GROUPS not found in {path.name}")


def prediction_rows(val, predictions, model_id, fold_id):
    """Pair actual/prediction by panel keys, including after a DL merge/reorder.

    Arrays must follow val order. A prediction frame must supply KEYS plus
    'prediction'; its index/order is irrelevant. Missing rows are not allowed;
    missing *values* (e.g. seasonal warm-up) are retained for coverage accounting.
    """
    gold = val[KEYS + ["target_next_week"]].rename(columns={"target_next_week": "actual"}).copy()
    if gold[KEYS].isna().any().any() or gold.duplicated(KEYS).any():
        raise ValueError("validation keys must be unique and non-null")
    if not np.isfinite(gold.actual.to_numpy(dtype=float)).all():
        raise ValueError("validation actuals must be finite")
    if isinstance(predictions, pd.DataFrame):
        pred = predictions[KEYS + ["prediction"]].copy()
    else:
        values = np.asarray(predictions, dtype=float)
        if values.shape != (len(gold),):
            raise ValueError("prediction shape must match validation rows")
        pred = gold[KEYS].assign(prediction=values)
    if pred[KEYS].isna().any().any() or pred.duplicated(KEYS).any():
        raise ValueError("prediction keys must be unique and non-null")
    joined = gold.merge(pred, on=KEYS, how="outer", validate="one_to_one", indicator=True)
    if not joined._merge.eq("both").all():
        raise ValueError("prediction keys differ from validation keys")
    if np.isinf(joined.prediction.to_numpy(dtype=float)).any():
        raise ValueError("infinite predictions are invalid")
    return joined.drop(columns="_merge").assign(model_id=model_id, fold_id=fold_id)


def validate_ledger(ledger, folds, holdout_weeks, expected_models):
    """Reject duplicates, missing model/fold panels and any holdout prediction."""
    if ledger.duplicated(["model_id", "fold_id"] + KEYS).any():
        raise ValueError("duplicate prediction keys")
    if ledger.week.isin(holdout_weeks).any():
        raise ValueError("holdout predictions are forbidden in Phase 12")
    if set(ledger.model_id) != set(expected_models):
        raise ValueError("model coverage differs from the evaluation registry")
    if set(ledger.fold_id) != set(range(1, len(folds) + 1)):
        raise ValueError("missing or unexpected validation fold")
    for i, fold in enumerate(folds, 1):
        train_weeks, val_weeks = set(fold.train_weeks), set(fold.val_weeks)
        if train_weeks & val_weeks or (train_weeks | val_weeks) & set(holdout_weeks):
            raise ValueError("train/validation/holdout overlap")
        if max(train_weeks) >= min(val_weeks):
            raise ValueError("training must precede validation")
        panel = ledger[ledger.fold_id.eq(i)]
        if not panel.week.isin(fold.val_weeks).all():
            raise ValueError("prediction week outside its validation fold")
        anchor = panel[panel.model_id.eq(expected_models[0])][KEYS + ["actual"]]
        if anchor.empty or set(anchor.week) != set(pd.to_datetime(fold.val_weeks)):
            raise ValueError("incomplete validation weeks")
        for model in expected_models:
            part = panel[panel.model_id.eq(model)][KEYS + ["actual"]]
            paired = anchor.merge(part, on=KEYS, how="outer", suffixes=("_a", "_b"), indicator=True)
            if not paired._merge.eq("both").all() or not paired.actual_a.eq(paired.actual_b).all():
                raise ValueError("models must have identical validation keys and actuals")


def _cohort_hash(frame):
    keys = frame[["fold_id"] + KEYS].sort_values(["fold_id"] + KEYS)
    return hashlib.sha256(keys.to_csv(index=False, date_format="%Y-%m-%d").encode()).hexdigest()


def summarize_predictions(ledger, registry):
    """Macro fold means, sample WAPE std, and ranks only for identical cohorts."""
    if ledger.duplicated(["model_id", "fold_id"] + KEYS).any():
        raise ValueError("duplicate prediction keys")
    if set(ledger.model_id) != set(registry):
        raise ValueError("model coverage differs from the evaluation registry")
    if ledger[["model_id", "fold_id"] + KEYS].isna().any().any():
        raise ValueError("ledger keys must be non-null")
    if not np.isfinite(ledger.actual.to_numpy(dtype=float)).all() or np.isinf(ledger.prediction).any():
        raise ValueError("invalid ledger actual/prediction values")
    core_ids = [m for m in CORE if m in registry]
    fold_ids = sorted(ledger.fold_id.unique())
    key_sets, common_sets = {}, {}
    for fold in fold_ids:
        for model in registry:
            part = ledger[ledger.fold_id.eq(fold) & ledger.model_id.eq(model)]
            if part.empty:
                raise ValueError(f"missing model/fold: {model}/{fold}")
            valid = part[np.isfinite(part.prediction)]
            if valid.empty:
                raise ValueError(f"no valid predictions: {model}/{fold}")
            key_sets[(fold, model)] = set(valid[KEYS].itertuples(index=False, name=None))
        common_sets[fold] = set.intersection(*(key_sets[(fold, m)] for m in core_ids))
        if not common_sets[fold]:
            raise ValueError(f"no common core coverage in fold {fold}")
    need_common = any(key_sets[(f, m)] != common_sets[f] for f in fold_ids for m in core_ids)

    summary_rows, fold_rows = [], []
    requests = [(m, registry[m]["comparison_group"], False) for m in registry]
    # Full LightGBM is reused as a reference, never fitted again.
    if any(v["comparison_group"] == "ablation_cv" for v in registry.values()):
        requests.append(("global_lgb", "ablation_cv", False))
    if "lgb_dl" in registry:
        requests.append(("global_lgb", "dl_cv", False))
    if need_common:
        requests.extend((m, "core_common", True) for m in core_ids)

    for model, group, common in requests:
        data = ledger[ledger.model_id.eq(model)]
        valid = data[np.isfinite(data.prediction)].copy()
        if common:
            keep = [tuple(row) in common_sets[f] for f, *row in valid[["fold_id"] + KEYS].itertuples(index=False, name=None)]
            valid = valid.loc[keep]
        scores = []
        for fold in fold_ids:
            part = valid[valid.fold_id.eq(fold)]
            if part.empty:
                raise ValueError(f"empty scored fold: {model}/{group}/{fold}")
            score = score_all(part.actual.values, part.prediction.values)
            scores.append(score)
            n_eligible = int(data.fold_id.eq(fold).sum())
            fold_rows.append(dict(model_id=model, comparison_group=group, fold_id=int(fold),
                                  n_eligible=n_eligible, n_scored=len(part),
                                  coverage=len(part) / n_eligible, **score))
        meta = registry[model]
        summary_rows.append(dict(
            model_id=model, model=meta["model"], phase=meta["phase"], comparison_group=group,
            source="rerun", source_notebook=meta["source_notebook"],
            n_folds=len(fold_ids), n_eligible=len(data), n_scored=len(valid),
            coverage=len(valid) / len(data), cohort_hash=_cohort_hash(valid),
            WAPE_std=float(np.std([s["WAPE"] for s in scores], ddof=1)),
            **{k: float(np.mean([s[k] for s in scores])) for k in METRICS},
            delta_WAPE=np.nan, rank=np.nan, rank_group=np.nan, notes=meta.get("notes", ""),
        ))
    summary = pd.DataFrame(summary_rows)
    summary["rank_group"] = summary["rank_group"].astype(object)
    for group in ("core_cv", "core_common"):
        subset = summary[summary.comparison_group.eq(group)]
        for cohort, rows in subset.groupby("cohort_hash"):
            if len(rows) > 1:
                summary.loc[rows.index, "rank"] = rows.WAPE.rank(method="min")
                summary.loc[rows.index, "rank_group"] = f"{group}:{cohort}"
    for group in ("ablation_cv", "dl_cv"):
        rows = summary[summary.comparison_group.eq(group)]
        if rows.empty:
            continue
        ref = rows[rows.model_id.eq("global_lgb")].iloc[0]
        if not rows.cohort_hash.eq(ref.cohort_hash).all():
            raise ValueError(f"{group} must share the full-model cohort")
        summary.loc[rows.index, "delta_WAPE"] = rows.WAPE - ref.WAPE
    return summary, pd.DataFrame(fold_rows)


class _TableReader(HTMLParser):
    """Read pandas' recorded HTML tables with only the standard library."""
    def __init__(self):
        super().__init__()
        self.rows, self.row, self.cell = [], [], None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.row = []
        elif tag in ("th", "td"):
            self.cell = []

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag):
        if tag in ("th", "td") and self.cell is not None:
            self.row.append("".join(self.cell).strip())
            self.cell = None
        elif tag == "tr" and self.row:
            self.rows.append(self.row)


def recorded_table(path, columns, expected_values=None):
    """Find exactly one executed output with the required schema, or fail."""
    path = Path(path)
    notebook = json.loads(path.read_text(encoding="utf-8"))
    matches = []
    for ci, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code" or cell.get("execution_count") is None:
            continue
        for oi, output in enumerate(cell.get("outputs", [])):
            html = output.get("data", {}).get("text/html")
            if not html:
                continue
            reader = _TableReader()
            reader.feed(_text(html))
            if not reader.rows or not set(columns).issubset(reader.rows[0]):
                continue
            header = reader.rows[0]
            if any(len(row) != len(header) for row in reader.rows[1:]):
                raise ValueError(f"malformed recorded table in {path.name}")
            table = [{k: row[header.index(k)] for k in columns} for row in reader.rows[1:]]
            if not table:
                raise ValueError(f"empty recorded table in {path.name}")
            if expected_values and any({row[k] for row in table} != set(values)
                                       for k, values in expected_values.items()):
                continue
            matches.append((table, dict(notebook=path.name, sha256=file_hash(path),
                                       cell_index=ci, output_index=oi, recorded_values=table)))
    if len(matches) != 1:
        raise ValueError(f"expected one recorded table in {path.name}, found {len(matches)}")
    return matches[0]


def load_recorded_results(root):
    """ETS and cold-start are historical references, not fresh comparable CV."""
    root = Path(root)
    ets_path = root / "notebooks/07_core_forecasting.ipynb"
    cold_path = root / "notebooks/10_cold_start.ipynb"
    ets, ets_source = recorded_table(ets_path, ["model"] + METRICS,
        expected_values={"model": {"ETS (Holt-Winters)", "Global LightGBM (same top-5 series)"}})
    cold_columns = ["window", "Analog WAPE", "Meta-learner WAPE", "Full model WAPE"]
    cold, cold_source = recorded_table(cold_path, cold_columns)
    if {r["model"] for r in ets} != {"ETS (Holt-Winters)", "Global LightGBM (same top-5 series)"}:
        raise ValueError("unexpected recorded ETS models")
    if {r["window"] for r in cold} != {"weeks 0-7 (launch window)", "weeks 8+"}:
        raise ValueError("unexpected recorded cold-start windows")
    records = []
    for row in ets:
        records.append(dict(
            model_id="recorded_ets" if row["model"].startswith("ETS") else "recorded_lgb_top5",
            model=row["model"], phase="7", comparison_group="recorded_ets_top5",
            source="recorded", source_notebook="notebooks/07_core_forecasting.ipynb",
            **{k: float(row[k]) for k in METRICS},
            notes="Recorded 3-decimal scores; 5 series selected on the full table. ETS uses a fixed-origin multi-step forecast, "
                  "with origin/target alignment different from ML; successful-fit coverage is not recorded. "
                  "No fresh-CV rank. Counts/std/coverage unavailable (N/A).",
        ))
    for row in cold:
        window = "0_7" if row["window"].startswith("weeks 0") else "8_plus"
        for col, label in [("Analog WAPE", "Analog"), ("Meta-learner WAPE", "Meta-learner"), ("Full model WAPE", "Full LightGBM")]:
            records.append(dict(
                model_id=f"recorded_cold_{window}_{col.split()[0].lower()}", model=label, phase="10",
                comparison_group=f"recorded_cold_{window}", source="recorded",
                source_notebook="notebooks/10_cold_start.ipynb", WAPE=float(row[col]),
                notes=f"Recorded 3-decimal WAPE, {row['window']}; 5 held-out SKUs, not time-respecting CV. "
                      "Age starts at first modeling-table appearance, not zero-history launch. "
                      "Analog may score fewer rows than ML. MAE/RMSE/SMAPE/counts/std/coverage unavailable (N/A); no rank.",
            ))
    return pd.DataFrame(records), [ets_source, cold_source]


def run_cv(root, progress=print, checkpoint_dir=None):
    """Fit each variant once per fold, sharing full LightGBM reference predictions."""
    from .baseline import add_baseline_predictions
    from .forecast import (ALL_FEATURE_COLS, prepare_categoricals, fit_predict_lgb,
                           fit_predict_xgb, fit_predict_rf, fit_predict_local_per_sku)
    from features.dl_embeddings import fit_fold_embeddings, transform_with_fold_embeddings, dl_embedding_cols
    from splits.walk_forward import WalkForwardSplitter

    root = Path(root)
    df = pd.read_csv(root / "data/processed/weekly_features.csv", parse_dates=["week"])
    if df.duplicated(KEYS).any():
        raise ValueError("input panel has duplicate keys")
    folds, holdout = WalkForwardSplitter(**SPLIT_CONFIG).get_folds(df)
    if len(folds) != 7 or len(holdout) != 10:
        raise ValueError("Phase 12 expects the existing 7-fold/10-week protocol")
    dfc = prepare_categoricals(df)
    # Preserve Phase 7 category bookkeeping, then exclude all holdout origins.
    cv = dfc[~dfc.week.isin(holdout)].copy()
    dfb = add_baseline_predictions(cv)
    groups = load_feature_groups(root, ALL_FEATURE_COLS)
    registry = {m: dict(model=label, phase=str(phase), comparison_group="core_cv",
                       source_notebook=f"notebooks/{'06_baseline_models' if phase == 6 else '07_core_forecasting'}.ipynb",
                       notes="Own finite-prediction coverage; macro mean of 7 folds; rank only within identical key cohorts.")
                for m, (label, phase) in CORE.items()}
    for group in groups:
        registry[f"without_{group}"] = dict(model=f"LightGBM without {group}", phase="11",
            comparison_group="ablation_cv", source_notebook="notebooks/11_feature_ablation.ipynb",
            notes="Delta vs the same-run full LightGBM; positive delta means worse WAPE. No new significance claim.")
    registry["lgb_dl"] = dict(model="LightGBM + DL embeddings", phase="11b", comparison_group="dl_cv",
        source_notebook="notebooks/11b_dl_feature_experiment.ipynb",
        notes="Embeddings fit on training fold only; delta vs same-run LightGBM. No new significance claim.")
    cache = None
    if checkpoint_dir is not None:
        inputs = ["data/processed/weekly_features.csv", "src/models/evaluation.py", "src/models/forecast.py",
                  "src/models/baseline.py", "src/models/metrics.py", "src/splits/walk_forward.py",
                  "src/features/dl_embeddings.py", "notebooks/11_feature_ablation.ipynb"]
        identity = dict(files={p: file_hash(root / p) for p in inputs}, split=SPLIT_CONFIG,
                        python=platform.python_version(), os=platform.platform(),
                        packages={p: importlib.metadata.version(p) for p in
                                  ["numpy", "pandas", "lightgbm", "xgboost", "scikit-learn", "torch", "joblib", "scipy"]})
        signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        cache = Path(checkpoint_dir) / signature
        cache.mkdir(parents=True, exist_ok=True)
        (cache / "identity.json").write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8")

    pieces = []
    # Finish the user's primary pair before spending time on supporting models.
    for stage in ("focus", "support"):
        progress(f"Stage: {stage}")
        for i, fold in enumerate(folds, 1):
            progress(f"Fold {i}/{len(folds)}: {pd.Timestamp(fold.val_weeks.min()).date()} to {pd.Timestamp(fold.val_weeks.max()).date()}")
            train = cv[cv.week.isin(fold.train_weeks)]
            val = cv[cv.week.isin(fold.val_weeks)]
            if train.week.isin(holdout).any() or val.week.isin(holdout).any() or train.week.max() >= val.week.min():
                raise ValueError("invalid fold boundaries")

            def obtain(model, predict):
                start = time.perf_counter()
                path = cache / f"{model}_{i}.csv" if cache is not None else None
                checksum = path.with_suffix(".sha256") if path is not None else None
                if path is not None and path.exists() and checksum.exists():
                    if file_hash(path) != checksum.read_text().strip():
                        raise ValueError(f"checkpoint checksum mismatch: {model}/{i}")
                    saved = pd.read_csv(path, parse_dates=["week"], float_precision="round_trip")
                    if not saved.model_id.eq(model).all() or not saved.fold_id.eq(i).all():
                        raise ValueError("checkpoint model/fold mismatch")
                    result = prediction_rows(val, saved[KEYS + ["prediction"]], model, i)
                    paired = result.merge(saved[KEYS + ["actual"]], on=KEYS, suffixes=("_new", "_old"), validate="one_to_one")
                    if not paired.actual_new.eq(paired.actual_old).all():
                        raise ValueError("checkpoint actuals differ from current data")
                    registry[model].setdefault("restored_folds", []).append(i)
                    progress(f"  {model}: restored verified checkpoint")
                    return result
                progress(f"  {model}: running")
                result = prediction_rows(val, predict(), model, i)
                if path is not None:
                    temporary = path.with_suffix(".tmp")
                    result.to_csv(temporary, index=False, date_format="%Y-%m-%d")
                    temporary.replace(path)
                    checksum.write_text(file_hash(path) + "\n")
                progress(f"  {model}: done in {time.perf_counter() - start:.1f}s")
                return result

            if stage == "focus":
                base = dfb[dfb.week.isin(fold.val_weeks)]
                for m, col in [("naive", "pred_naive"), ("seasonal_naive", "pred_seasonal_naive"), ("moving_avg_4", "pred_moving_avg_4")]:
                    pieces.append(obtain(m, lambda col=col: base[KEYS + [col]].rename(columns={col: "prediction"})))
                for m, fit in [("global_lgb", fit_predict_lgb), ("global_xgb", fit_predict_xgb)]:
                    pieces.append(obtain(m, lambda fit=fit: fit(train, val)))
            else:
                for m, fit in [("global_rf", fit_predict_rf), ("local_lgb", fit_predict_local_per_sku)]:
                    pieces.append(obtain(m, lambda fit=fit: fit(train, val)))
                for group, removed in groups.items():
                    cols = [c for c in ALL_FEATURE_COLS if c not in removed]
                    pieces.append(obtain(f"without_{group}", lambda cols=cols: fit_predict_lgb(train, val, feature_cols=cols)))

                def predict_dl():
                    tables = fit_fold_embeddings(train)
                    train_ext = transform_with_fold_embeddings(train, tables)
                    val_ext = transform_with_fold_embeddings(val, tables)
                    pred = fit_predict_lgb(train_ext, val_ext, feature_cols=ALL_FEATURE_COLS + dl_embedding_cols())
                    return val_ext[KEYS].assign(prediction=pred)

                pieces.append(obtain("lgb_dl", predict_dl))
        if stage == "focus":
            focus_registry = {m: registry[m] for m in registry if m in CORE and m not in ("local_lgb", "global_rf")}
            focus_summary, _ = summarize_predictions(pd.concat(pieces, ignore_index=True), focus_registry)
            focus_summary = focus_summary[focus_summary.comparison_group.eq("core_cv") & focus_summary.model_id.isin(FOCUS)]
            progress("Primary pair completed on all 7 folds:\n" + focus_summary[["model", "WAPE", "WAPE_std", "n_scored"]].to_string(index=False))
            if cache is not None:
                focus_summary.to_csv(cache / "focus_summary.csv", index=False)
    ledger = pd.concat(pieces, ignore_index=True).sort_values(["fold_id", "model_id"] + KEYS).reset_index(drop=True)
    validate_ledger(ledger, folds, holdout, list(registry))
    return ledger, registry, folds, holdout, groups


def make_manifest(root, folds, holdout, groups, recorded_sources):
    from .forecast import ALL_FEATURE_COLS, DEFAULT_LGB_PARAMS, DEFAULT_XGB_PARAMS, DEFAULT_RF_PARAMS
    root = Path(root)
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()
    packages = {p: importlib.metadata.version(p) for p in
                ["numpy", "pandas", "scikit-learn", "scipy", "lightgbm", "xgboost", "torch", "matplotlib", "statsmodels", "nbformat", "nbclient"]}
    files = ["data/processed/weekly_features.csv", "src/models/evaluation.py", "src/models/forecast.py",
             "src/models/baseline.py", "src/models/metrics.py", "src/splits/walk_forward.py",
             "src/features/dl_embeddings.py", "notebooks/11_feature_ablation.ipynb", "requirements-lock.txt"]
    reference, reference_source = recorded_table(root / "notebooks/07_core_forecasting.ipynb", ["model", "mean_WAPE", "std_WAPE"])
    return dict(
        created_at_utc=datetime.now(timezone.utc).isoformat(), source_commit=git("rev-parse", "HEAD"),
        upstream_commit=git("rev-parse", "origin/main"), git_status_at_run=git("status", "--porcelain"),
        python=platform.python_version(), os=platform.platform(), packages=packages,
        input_hashes={p: file_hash(root / p) for p in files}, split_config=SPLIT_CONFIG,
        folds=[dict(fold_id=i, train_weeks=[str(pd.Timestamp(w).date()) for w in f.train_weeks],
                    val_weeks=[str(pd.Timestamp(w).date()) for w in f.val_weeks]) for i, f in enumerate(folds, 1)],
        reserved_holdout_origin_weeks=[str(pd.Timestamp(w).date()) for w in holdout],
        january_2025_batches_read=False, holdout_fit_or_scored=False,
        origin_policy="Preserve Phase 5 origin-week split and existing target_next_week labels; no holdout-origin predictions.",
        focus_models=FOCUS, primary_model="global_lgb", feature_columns=ALL_FEATURE_COLS, feature_groups=groups,
        model_parameters=dict(lightgbm=DEFAULT_LGB_PARAMS, xgboost=DEFAULT_XGB_PARAMS, random_forest=DEFAULT_RF_PARAMS,
                              embeddings=dict(epochs=30, hidden=16, lr=0.01, weight_decay=0.0001, seed=42,
                                              dimensions=dict(sku=4, channel=2, region=2), cpu_threads=1)),
        metric_aggregation="Unweighted mean of per-fold metrics; WAPE_std uses ddof=1; undefined values propagate.",
        metric_units="WAPE and SMAPE are ratios (0.224 means 22.4%); MAE/RMSE are units sold.",
        recorded_sources=recorded_sources,
        historical_core_reference=dict(values=reference, source=reference_source),
        reproducibility_note="Model package versions follow the original lock where possible; Python/OS and transitive packages differ. Historical results are not overwritten.",
    )


def export_reports(root, ledger, registry, folds, holdout, groups):
    """Build every artifact from the ledger plus recorded notebook outputs."""
    import matplotlib.pyplot as plt
    root = Path(root)
    fresh, fold_scores = summarize_predictions(ledger, registry)
    recorded, sources = load_recorded_results(root)
    summary = pd.concat([fresh, recorded], ignore_index=True)
    out = root / "reports/phase12"
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in [("model_comparison.csv", summary), ("fold_scores.csv", fold_scores), ("predictions.csv", ledger)]:
        frame.to_csv(out / name, index=False, na_rep="N/A", date_format="%Y-%m-%d")
    manifest = make_manifest(root, folds, holdout, groups, sources)
    manifest["checkpoint_restored_folds"] = {m: meta.get("restored_folds", []) for m, meta in registry.items()}
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    # The two focus models use their own identical full-CV cohort. A seasonal
    # baseline's missing history must not shrink this principal comparison.
    plot_group = "core_cv"
    focus = fresh[fresh.comparison_group.eq(plot_group) & fresh.model_id.isin(FOCUS)].set_index("model_id").loc[FOCUS]
    if focus.cohort_hash.nunique() != 1:
        raise ValueError("focus models must score identical keys")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = ["#2874A6", "#D68910"]
    axes[0].bar(focus.model, focus.WAPE * 100, yerr=focus.WAPE_std * 100, capsize=5, color=colors)
    axes[0].set_ylabel("WAPE (%) — lower is better")
    axes[0].set_title("Mean and sample std across 7 folds")
    axes[0].set_ylim(0, float((focus.WAPE + focus.WAPE_std).max()) * 120)
    for x, value in enumerate(focus.WAPE):
        axes[0].text(x, value * 100 + 0.9, f"{value * 100:.3f}%", ha="center", fontsize=10)
    for model, color in zip(FOCUS, colors):
        part = fold_scores[fold_scores.comparison_group.eq(plot_group) & fold_scores.model_id.eq(model)].sort_values("fold_id")
        axes[1].plot(part.fold_id, part.WAPE * 100, marker="o", label=CORE[model][0], color=color)
    axes[1].set_xlabel("Walk-forward fold")
    axes[1].set_ylabel("WAPE (%)")
    axes[1].set_title("Accuracy over time on identical validation rows")
    axes[1].legend()
    fig.suptitle("Phase 12 — LightGBM vs XGBoost (CV only)")
    fig.tight_layout()
    fig.savefig(out / "model_comparison.png", dpi=150)
    plt.close(fig)
    return summary, fold_scores, manifest
