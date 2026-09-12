# FMCG Weekly Sales Forecasting

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-3499CD?style=for-the-badge&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-006ACC?style=for-the-badge&logoColor=white)
![statsmodels](https://img.shields.io/badge/statsmodels-8CAAE6?style=for-the-badge&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=matplotlib&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![License](https://img.shields.io/badge/MIT_License-4CAF50?style=for-the-badge&logoColor=white)

<div align="right">

![Progress](https://img.shields.io/badge/Progress_12%2F16-4CAF50?style=for-the-badge&logoColor=white)

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
| `data/raw/batch_MI-006_2025-01-*.parquet` | daily, MI-006 only | 4 files | Jan 2025 data, post-training-window — true future holdout |
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

## Tech Stack

pandas, numpy, scikit-learn, LightGBM, XGBoost, statsmodels / linearmodels, matplotlib, seaborn,
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
├── notebooks/                  # numbered, one per phase
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
- [x] **Phase 7 — Core forecasting**: global pooled LightGBM wins, **WAPE 0.224**
  - Beats the baseline (0.243), local per-SKU LightGBM (0.257), and Holt-Winters ETS (0.301 vs. 0.214 for LightGBM on the same subset)
  - Random Forest ties at 0.224; XGBoost scores 0.225
    - Paired t-tests vs. LightGBM (Holm-corrected for both challengers): no significant difference (XGBoost p = 0.297, Random Forest p = 0.606)
    - Consistent with robustness to library choice — not proof of equality; 7 folds is limited evidence
  - LightGBM carried forward as the primary model
- [x] **Phase 8 — Promotion effect**: two-way fixed-effects regression estimates a **+28.4% uplift** [27.6%, 29.3%], p < 0.001
  - Consistent ~28–29% across all 5 categories
  - Estimate only — not proof of causal uplift or profitability
  - Repeating on/off treatment audited in `notebooks/08b_promotion_robustness.ipynb`:
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

### Evaluation
- [ ] **Phase 12 — Model evaluation rollup**: consolidate all models into one comparison table
- [ ] **Phase 13 — Future holdout backtest**: test against real, never-seen January 2025 data

### Deployment & Communication
- [ ] **Phase 14 — Communication deliverable**: `reports/final_report.md` with business-framed findings
- [ ] **Phase 15 — Documentation & polish**: finalize README, docstrings, tests

**Working note**: notebooks include detailed Thai-language explanations for non-technical
readers. `reports/final_report.md` stays in English for a hiring-manager audience.

## Key Findings

- **Forecasting**: global pooled LightGBM reaches **WAPE 0.224** on 7-fold walk-forward CV
  - Ahead of the best baseline (0.243), local per-SKU LightGBM (0.257), and Holt-Winters ETS (0.301 on the same top-5-series subset where LightGBM scores 0.214)
  - Global pooled Random Forest scores 0.224 too; XGBoost scores 0.225 — a robustness check on library choice, not separate models carried forward
    - Paired t-tests against LightGBM (Holm-corrected for both challengers): no significant difference (XGBoost p = 0.297, Random Forest p = 0.606)
- **Promotions**: two-way fixed-effects regression estimates a **+28.4% sales uplift** [27.6%, 29.3%], p < 0.001, consistent across all 5 categories
  - Negative-weight audit: **0 / 19,032** treated cells receive negative weights (independently reproduced in Python and official R `TwoWayFEWeights`)
  - Exact-match WAS: **+28.9%** [28.0%, 29.8%]
  - Recurring-treatment distributed-lag check: no statistical evidence of pull-forward over weeks 1–4 (lag sum −0.0074, p = 0.421; joint lags p = 0.870; placebo leads p = 0.222)
  - Addresses weighting and short-horizon displacement under the stated model only — does not prove causal identification, absence of pull-forward, or profitability (see `notebooks/08b_promotion_robustness.ipynb`)
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
