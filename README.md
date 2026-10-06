# FMCG Weekly Sales Forecasting

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-3499CD?style=for-the-badge&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-006ACC?style=for-the-badge&logoColor=white)
![CatBoost](https://img.shields.io/badge/CatBoost-FFCC00?style=for-the-badge&logoColor=black)
![statsmodels](https://img.shields.io/badge/statsmodels-8CAAE6?style=for-the-badge&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=matplotlib&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![License](https://img.shields.io/badge/MIT_License-4CAF50?style=for-the-badge&logoColor=white)

<div align="right">

![Progress](https://img.shields.io/badge/Progress_14%2F16-4CAF50?style=for-the-badge&logoColor=white)

</div>

Demand forecasting, promotion effectiveness, and cold-start forecasting for a simulated
multi-channel, multi-region FMCG business.

## Problem Statement

A demand planner needs answers to five recurring questions:

1. **Forecasting** — units expected next week, by SKU/channel/region
2. **Promotions** — do they pay for themselves, or just pull forward existing demand?
3. **Seasonality** — how much of a category's sales swing is seasonal vs. trend?
4. **Cold start** — how do we forecast a new SKU with no sales history?
5. **Feature value** — which engineered features actually move forecast accuracy?

Answered end-to-end: raw transactional data → validated, back-tested models → a written
analysis with business-interpretable results.

## Data

The weekly modeling table is built entirely from raw daily transactions, not received
pre-aggregated — `src/data/build_weekly_base.py` derives the roll-up, all lag/rolling/target
features, and every calendar/lifecycle feature from `FMCG_2022_2024.csv` alone.

| File | Grain | Rows | Description |
|---|---|---|---|
| `data/raw/FMCG_2022_2024.csv` | daily | 190,758 | Raw transactions, Jan 2022–Dec 2024 — the only true raw input |
| `data/raw/batch_MI-006_2025-01-*.parquet` | daily, MI-006 only | 4 files | Jan 2025 data, post-training-window — true future holdout (fails the 2022–2024 data contract, see Phase 13) |
| `data/interim/daily_validated.csv` | daily | 190,758 | Phase 3 output: 3 negative-value rows clipped to 0 |
| `data/processed/weekly_features.csv` | weekly | 31,027 | Self-built base + enrichment (30 SKUs) + hypothesis-driven features |
| `data/raw/given_reference/*.csv` | weekly | — | Originally-given tables, kept for reference only; not read by the pipeline |

**30 SKUs**, 5 categories (Yogurt 11, Milk 7, Snack 6, ReadyMeal 5, Juice 1), 3 channels, 3
regions. SKUs launch on staggered real dates (Feb 2022–Jun 2023), used as natural cold-start
cases rather than synthetic ones.

**Data quality issues found and fixed while deriving the base table** (rather than copied from
the given files):
- **`avg_temp` / `inflation_index`**: varied by sales channel — physically implausible
  - Regenerated deterministically at the correct grain (region×week / week)
  - Note: regenerated values are synthetic, seeded stand-ins, not real records — the "no
    incremental value" finding (Phase 11) is about these specific proxies only
- **`is_holiday_week` / `is_holiday_peak`**: no reconstructible rule in the given table (~95%
  best match against several Polish-holiday hypotheses, with an inconsistent mismatch pattern)
  - Redefined from a correct Polish public-holiday calendar instead
- **`sku_age`**: must anchor to a SKU's first sale across *all* channels/regions, not per group
  - 14/270 groups start selling after their SKU's true launch elsewhere, which silently
    miscomputes age (and the `lifecycle_stage` derived from it) if anchored per-group

## Methodology

Standard data science lifecycle, 16 phases mapped to numbered notebooks in `notebooks/`:

| Stage | Phase | Focus |
|---|---|---|
| Business Understanding | 0 | Business framing, success metrics |
| Project Setup | 1 | Repo & environment |
| Data Understanding | 2 | EDA, data-quality audit |
| Data Preparation | 3–5 | Validation · feature engineering · walk-forward split design |
| Modeling | 6–11 | Baselines · forecasting · promotion effect · seasonality · cold-start · ablation |
| Evaluation | 12–13 | Model rollup · future holdout backtest |
| Deployment & Communication | 14–15 | Final report · documentation |

## Approach Highlights

- **Splitting**: strictly time-based (by week), never random
  - No model sees a week it will later be tested on
- **Promotion effect**: two-way fixed-effects regression (SKU + week/season FE, price-controlled)
  - Not a naive promo-vs-non-promo comparison — promotions are confounded with price and season
- **Cold start**: validated against real staggered SKU launches
  - Analog-matching vs. a meta-learner restricted to launch-time-available features
- **Metrics**: WAPE/SMAPE alongside MAE/RMSE
  - MAPE is unstable at low SKU-week volumes
- **Base-table provenance**: built from raw data by this project's own code, not received pre-aggregated (see Data)
  - Re-running Phases 4–11 on it reproduced every headline result — a robustness check, not just a rebuild
- **Reproducibility**: a fixed `random_state` alone isn't enough for bit-reproducibility under multithreading
  - LightGBM and scikit-learn's RandomForest both have floating-point summation-order nondeterminism independent of the seed
  - LightGBM: `deterministic=True` + `force_row_wise=True`
  - RandomForest: single-threaded (`n_jobs=1`) — no deterministic-parallel mode in scikit-learn, dataset small enough that this costs little
  - `requirements-lock.txt` pins the exact package versions used for the numbers in this document
  - Exception: the Phase 7 numbers (incl. the CatBoost challenger) were last re-run in the current `.venv` (LightGBM 4.7.0, XGBoost 3.4.1, scikit-learn 1.9.0, CatBoost 1.2.10), which is newer than the lock file's LightGBM/XGBoost/scikit-learn pins; the lock file only had CatBoost and its dependencies added, so it has not been regenerated for that environment

## Tech Stack

pandas, numpy, scikit-learn, LightGBM, XGBoost, CatBoost, statsmodels / linearmodels, matplotlib, seaborn,
joblib, pytest (badges above). Exact pinned versions in `requirements-lock.txt`.

## Repository Structure

```
Forecasting FMCG Daily Sales/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/                  # true raw input, untouched
│   │   └── given_reference/  # originally-given weekly tables — reference only
│   ├── interim/               # validated/cleaned intermediate tables
│   └── processed/              # final modeling-ready feature tables
├── notebooks/                   # grouped by business question, numbered within each
│   ├── 00_foundation/            # Phases 2-5: EDA, validation, feature engineering, split strategy
│   ├── 01_forecasting/           # Phases 6,7,12,13 — Q1 Forecasting
│   ├── 02_promotions/            # Phases 8,8b — Q2 Promotions
│   ├── 03_seasonality/           # Phase 9 — Q3 Seasonality
│   ├── 04_cold_start/            # Phase 10 — Q4 Cold start
│   ├── 05_feature_value/         # Phases 11,11b — Q5 Feature value
│   └── weekly_base_exploration.ipynb
├── src/
│   ├── data/                   # loading, validation, weekly base table
│   ├── features/                # feature engineering
│   ├── splits/                   # walk-forward / panel-aware CV
│   ├── models/                    # baseline, LightGBM, statsmodels wrappers
│   ├── causal/                     # fixed-effects promotion-uplift estimator
│   └── viz/                         # shared plotting helpers
├── reports/
│   ├── figures/
│   └── final_report.md
└── tests/
```

## Status

### Business Understanding
- [x] **Phase 0 — Business framing**: 5 use cases framed as business questions with success metrics (see Problem Statement)

### Project Setup
- [x] **Phase 1 — Repo & environment**: folder structure, `requirements.txt`, git/GitHub, README

### Data Understanding
- [x] **Phase 2 — EDA**: profiled all data files and confirmed the data is fit for modeling
  - Found 3 rows with impossible negative values
  - Confirmed roll-up integrity and no target leakage
  - Confirmed staggered SKU launches are usable for cold-start

### Data Preparation
- [x] **Phase 3 — Data validation**: duplicate/leakage checks all pass; negative-value rows clipped to 0
- [x] **Phase 4 — Weekly base table & feature engineering**: rebuilt the entire weekly table from raw daily data (see Data)
  - Diagnosed and fixed the enrichment bug
  - Generalized enrichment to all 30 SKUs
  - Added 4 hypothesis-driven features
  - Added regression tests (`tests/test_build_weekly_base.py`)
- [x] **Phase 5 — Split strategy**: panel-aware walk-forward CV — 7 folds plus an untouched final holdout, zero train/validation overlap verified

### Modeling
- [x] **Phase 6 — Baselines**: Moving Average (4w) is the best simple baseline, WAPE 0.243
- [x] **Phase 7 — Core forecasting**: global pooled LightGBM carried forward, **WAPE 0.224**
  - Beats the baseline (0.243), local per-SKU LightGBM (0.257), and Holt-Winters ETS (0.301 vs. 0.214 for LightGBM on the same subset)
  - XGBoost, Random Forest and CatBoost challengers land within 0.0020 of LightGBM, and all four within 0.0034 of each other (mean WAPE: CatBoost 0.2215, LightGBM 0.2235, Random Forest 0.2241, XGBoost 0.2249)
    - Paired t-tests vs. LightGBM (Holm-corrected across 3 challengers): no significant difference — p = 0.297 (XGBoost), 0.606 (Random Forest), 0.072 (CatBoost)
    - CatBoost has the lowest mean WAPE (−0.0020, raw p = 0.024) but does not clear the 0.05 bar after Holm correction; a likely small edge, not a confirmed one — 7 folds is limited evidence
    - Consistent with robustness to library choice — not proof of equality
  - LightGBM carried forward as the primary model (Phases 10, 11, 11b are built on it); switching to CatBoost would mean re-running them
- [x] **Phase 8 — Promotion effect**: two-way fixed-effects regression estimates a **+28.4% uplift** [27.6%, 29.3%], p < 0.001
  - Consistent ~28–29% across all 5 categories
  - Estimate only — not proof of causal uplift or profitability
  - Repeating on/off treatment audited in `notebooks/02_promotions/08b_promotion_robustness.ipynb`:
    - R `TwoWayFEWeights` + independent Python decomposition: **0 / 19,032** negative treated-cell weights
    - Exact-match WAS estimate: **+28.9%** [28.0%, 29.8%], close to TWFE
    - Distributed-lag check: no evidence of pull-forward over weeks 1–4 (lag sum −0.0074, p = 0.421; joint lags p = 0.870)
    - Does not prove absence of pull-forward or establish causal identification (see `LITERATURE_GROUNDING.md` §3a)
- [x] **Phase 9 — Seasonality & trend**: STL decomposition per category
  - Seasonal variance share: ~70% (Milk, per-SKU robustness check) to 87% (SnackBar)
  - Milk's naive sum-of-SKUs figure (8%) was a SKU-count-growth artifact, not real trend (see Key Findings below)
- [x] **Phase 10 — Cold-start forecasting**: compared analog-matching vs. meta-learner vs. full model on 5 held-out new SKUs
  - Both ML approaches clearly beat analog-matching at every SKU age
- [x] **Phase 11 — Feature ablation**: measured the accuracy contribution of every engineered feature
  - Calendar/lifecycle features matter most, ahead of lag/rolling history
  - Price and external enrichment add ~0 incrementally
  - Supplementary check in `notebooks/05_feature_value/11b_dl_feature_experiment.ipynb`: DL entity embeddings for sku/channel/region (trained leakage-safe per fold) add no incremental accuracy either — mean WAPE 0.2235 → 0.2237, paired t-test p = 0.538

### Evaluation
- [x] **Phase 12 — Model evaluation rollup**: fresh 7-fold CV, focused on Global LightGBM vs Global XGBoost
  - Same validation keys for the primary pair (14,850 rows each); mean WAPE **0.223517 / 0.224009**, sample fold std **0.005992 / 0.005838**
  - Descriptive comparison only; LightGBM remains the primary model for Phase 13
  - Supporting core models, coverage-aware comparisons, ablations and DL embeddings are included; ETS/cold-start remain separately labeled recorded references
  - This Phase 12 run predates the CatBoost addition; CatBoost is recorded in Phase 7 and is not included in this comparison table
  - [Notebook](notebooks/01_forecasting/12_model_evaluation.ipynb) · [Comparison table](reports/phase12/model_comparison.csv) · [Primary comparison chart](reports/phase12/model_comparison.png) · [Run manifest](reports/phase12/run_manifest.json)
  - Run on Python 3.12.5 / Windows; environment and source hashes are recorded without overwriting historical Phase 7 results
  - No fit/scoring on the reserved 10 holdout origin weeks; no January 2025 batch scoring
- [x] **Phase 13 — Future holdout backtest**: final holdout + January 2025 batches, each model fitted once, nothing tuned on the test data
  - **Label audit first**: origin 2024-12-23's label week (Dec 30–Jan 5) has only 2/7 days because the raw file ends 2024-12-31
    - Excluded from the primary score; scoring it anyway would inflate LightGBM's WAPE from 0.224 to 0.299
  - **Part A — final holdout** (9 complete origin weeks, 30 SKUs, 2,430 forecasts): Global LightGBM **WAPE 0.2238** vs 0.2235 in Phase 12 CV, so no generalization gap
    - Beats Moving Average (4w) (0.2423) in 9/9 weeks; series-level bootstrap 95% CI for the gap [1.47, 2.21] pp
    - CatBoost 0.2231, XGBoost 0.2241, Random Forest 0.2265: CIs vs LightGBM include or touch 0, matching Phase 7's library-robustness result
    - LightGBM over-forecasts by 3.7% in aggregate (XGBoost, trained on L1, by 1.4%)
  - **Part B — January 2025** (MI-006, 35 one-step-ahead forecasts, model frozen through January): a data-contract audit fails 11/14 checks before any scoring
    - Batches carry 3 rows per series-day (history ≤1), `units_sold` is only 10–29 per row, and promotions no longer lift sales (0.97× vs 1.93×)
    - Weekly units per series jump ~5.5× (×3.6 rows per week × ×1.5 units per row)
    - Every ML model scores WAPE 0.71–0.73 with ~−72% bias; trees can't extrapolate past the training maximum (largest MI-006 series-week label 248 vs smallest January label 374)
    - Naive scores 0.255 only because it copies the already-shifted level; this measures a data break, not demand-forecast skill
    - Leakage checks: the rebuilt MI-006 history matches `weekly_features.csv` exactly, and a batch-by-batch arrival replay changes no feature
    - `category_trend` / `price_index` need other Milk SKUs, so their last complete-week values are carried forward
  - Recommendation: gate new batches on the contract audit; don't retrain on these batches until the data owner confirms whether the new grain is intended
  - Run on Python 3.9.6 / macOS with the exact `requirements-lock.txt` pins (LightGBM 4.6.0, XGBoost 2.1.4, scikit-learn 1.6.1, CatBoost 1.2.10); environment and input hashes in `reports/phase13/run_manifest.json`
    - Cross-check in the same environment: re-running Phase 12's CV reproduces XGBoost bit-for-bit and LightGBM's mean WAPE to 0.223513 vs 0.223517 (per-fold drift ≤ 0.0007, cross-OS; LightGBM output is identical at 1 vs 4 threads)
    - On macOS, LightGBM/XGBoost wheels need an OpenMP runtime (`brew install libomp`)
  - [Notebook](notebooks/01_forecasting/13_future_holdout_backtest.ipynb) · [Results](reports/phase13/) · code in `src/models/holdout.py`, tests in `tests/test_holdout.py`

### Deployment & Communication
- [ ] **Phase 14 — Communication deliverable**: `reports/final_report.md` with business-framed findings
- [ ] **Phase 15 — Documentation & polish**: finalize README, docstrings, tests

**Working note**: notebooks include detailed Thai-language explanations for non-technical
readers. `reports/final_report.md` stays in English for a hiring-manager audience.

## Key Findings

- **Forecasting**: global pooled LightGBM reaches **WAPE 0.224** on 7-fold walk-forward CV
  - Ahead of the best baseline (0.243), local per-SKU LightGBM (0.257), and Holt-Winters ETS (0.301 on the same top-5-series subset where LightGBM scores 0.214)
  - Global pooled XGBoost, Random Forest and CatBoost score 0.221–0.225 — a robustness check on library choice, not separate models carried forward
    - Paired t-tests against LightGBM (Holm-corrected across 3 challengers): no significant difference (p = 0.297 / 0.606 / 0.072 for XGBoost / Random Forest / CatBoost)
- **Final holdout**: LightGBM WAPE **0.224** on the 10 reserved weeks (9 with complete labels), the same as CV (0.2235); nothing tuned on it
  - The January 2025 batches fail the historical data contract (3× row density, different value ranges, no promo effect), so their ~0.73 WAPE reflects a data break, not model skill
- **Promotions**: two-way fixed-effects regression estimates a **+28.4% sales uplift** [27.6%, 29.3%], p < 0.001, consistent across all 5 categories
  - Negative-weight audit: **0 / 19,032** treated cells receive negative weights (independently reproduced in Python and official R `TwoWayFEWeights`)
  - Exact-match WAS: **+28.9%** [28.0%, 29.8%]
  - Recurring-treatment distributed-lag check: no statistical evidence of pull-forward over weeks 1–4 (lag sum −0.0074, p = 0.421; joint lags p = 0.870; placebo leads p = 0.222)
  - Addresses weighting and short-horizon displacement under the stated model only — does not prove causal identification, absence of pull-forward, or profitability (see `notebooks/02_promotions/08b_promotion_robustness.ipynb`)
- **Seasonality**: variance share ranges from ~70% (Milk) to 87% (SnackBar) — category-dependent, not a single business-wide factor
  - Milk's naive sum-of-SKUs STL initially looked trend-dominated (58% trend / 8% seasonal)
    - Artifact of Milk's active SKU count growing 2→7 over the window — summing across a growing SKU count inflates apparent trend
  - Per-SKU robustness check (STL on per-SKU mean instead of sum) flips it to ~70% seasonal / 22% trend, in line with other categories
    - Per-SKU mean is noisier when few SKUs are active early in a series — the more credible read for Milk specifically, not a universal replacement for the sum-based numbers
- **Cold start**: both ML approaches clearly beat naive analog-matching at every SKU age
  - The full model held up from the first available week
- **Feature value**: calendar/lifecycle features matter most, ahead of lag/rolling history; price and external enrichment add close to nothing incrementally
  - This is a predictive-value finding, distinct from promotion's causal effect above
  - Caveat: `avg_temp`/`inflation_index` are synthetic, deterministic stand-ins (see Data), largely redundant with `month`/`is_summer`/`is_winter` already in the model
    - Shows those *specific proxies* add nothing here, not that real external data wouldn't help actual FMCG demand planning

Every finding above was reproduced by re-running Phases 4–11 on an independently rebuilt weekly
base table (see Data), with every headline number matching the original run within rounding.
