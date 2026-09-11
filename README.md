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

- **Splitting**: strictly time-based (by week), never random — no model sees a week it will
  later be tested on.
- **Promotion effect**: two-way fixed-effects regression (SKU + week/season FE, price-controlled),
  not a naive promo-vs-non-promo comparison — promotions are confounded with price and season.
- **Cold start**: validated against real staggered SKU launches; analog-matching vs. a
  meta-learner restricted to features available at launch time.
- **Metrics**: WAPE/SMAPE alongside MAE/RMSE — MAPE is unstable at low SKU-week volumes.
- **Base-table provenance**: built from raw data by this project's own code, not received pre-aggregated (see Data)
  - Re-running Phases 4–11 on it reproduced every headline result from the original run — a robustness check on the findings, not just a rebuild
- **Reproducibility**: a fixed `random_state` alone doesn't make LightGBM or scikit-learn's
  RandomForest bit-reproducible under multithreading — both have their own floating-point
  summation-order nondeterminism independent of the seed. LightGBM runs with
  `deterministic=True` + `force_row_wise=True`; RandomForest runs single-threaded
  (`n_jobs=1`) since scikit-learn has no equivalent deterministic-parallel mode and the
  dataset is small enough that this costs little. `requirements-lock.txt` pins the exact
  package versions used to produce the numbers in this document.

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
  - Random Forest challenger scores identically at 0.224; XGBoost scores 0.225 — paired t-tests of LightGBM against each challenger (Holm-corrected for testing both) found no significant difference either way: XGBoost p = 0.297, Random Forest p = 0.606 — results are consistent with the result being robust to library choice (failure to reject isn't proof of equality; 7 folds isn't enough data to make that claim)
  - LightGBM carried forward as the primary model
- [x] **Phase 8 — Promotion effect**: two-way fixed-effects regression finds **+28.4% uplift** [27.6%, 29.3%], p < 0.001
  - Consistent ~28–29% across all 5 categories
- [x] **Phase 9 — Seasonality & trend**: STL decomposition per category
  - Seasonal variance share ranges from 8% (Milk, trend-dominated) to 87% (SnackBar)
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
  - Global pooled Random Forest scores 0.224 too, XGBoost scores 0.225 — a robustness check on library choice, not separate models carried forward — paired t-tests against LightGBM (Holm-corrected for testing both challengers) found no significant difference either way (XGBoost p = 0.297, Random Forest p = 0.606)
- **Promotions**: **+28.4% sales uplift** [27.6%, 29.3%], p < 0.001, consistent across all 5 categories
- **Seasonality**: variance share ranges from 8% (Milk, trend-dominated) to 87% (SnackBar) — category-dependent, not a single business-wide factor
- **Cold start**: both ML approaches clearly beat naive analog-matching at every SKU age; the full model held up from the first available week
- **Feature value**: calendar/lifecycle features matter most, ahead of lag/rolling history; price and external enrichment add close to nothing incrementally
  - This is a predictive-value finding, distinct from promotion's causal effect above
  - Caveat: `avg_temp`/`inflation_index` are synthetic, deterministic stand-ins (see Data) largely redundant with `month`/`is_summer`/`is_winter` already in the model — this shows those *specific proxies* add nothing here, not that real external data wouldn't help actual FMCG demand planning

Every finding above was reproduced by re-running Phases 4–11 on an independently rebuilt weekly
base table (see Data), with every headline number matching the original run within rounding.
