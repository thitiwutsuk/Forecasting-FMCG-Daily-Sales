# FMCG Weekly Sales Forecasting

![Python](https://img.shields.io/badge/Python-3.9-3776AB?style=flat&logo=python&logoColor=white&labelColor=1a1a2e)
![pandas](https://img.shields.io/badge/Pandas-2.3-150458?style=flat&logo=pandas&logoColor=white&labelColor=1a1a2e)
![NumPy](https://img.shields.io/badge/NumPy-2.0-013243?style=flat&logo=numpy&logoColor=white&labelColor=1a1a2e)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6-F7931E?style=flat&logo=scikitlearn&logoColor=white&labelColor=1a1a2e)
![LightGBM](https://img.shields.io/badge/LightGBM-4.6-3499CD?style=flat&labelColor=1a1a2e)
![XGBoost](https://img.shields.io/badge/XGBoost-3.4-006ACC?style=flat&labelColor=1a1a2e)
![statsmodels](https://img.shields.io/badge/statsmodels-0.14-8CAAE6?style=flat&labelColor=1a1a2e)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.9-11557C?style=flat&logo=matplotlib&logoColor=white&labelColor=1a1a2e)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat&logo=jupyter&logoColor=white&labelColor=1a1a2e)
![License](https://img.shields.io/badge/License-MIT-4CAF50?style=flat&labelColor=1a1a2e)

![Progress](https://img.shields.io/badge/Progress-Complete_(12%2F16)-4CAF50?style=flat&labelColor=1a1a2e)

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
- `avg_temp`/`inflation_index` varied by sales channel — physically implausible, and noisier than
  the true 3-year signal. Regenerated deterministically at the correct grain (region×week / week).
  **Note:** the regenerated values are synthetic, seeded stand-ins (not real weather/inflation
  records) — appropriate for this simulated dataset, but a "no incremental value" finding on these
  columns (Phase 11) is about these specific proxies, not a claim that real external data wouldn't
  help actual demand planning.
- `is_holiday_week`/`is_holiday_peak` follow no reconstructible rule in the given table (best
  achievable match against several Polish-holiday hypotheses: ~95%, with an inconsistent
  mismatch pattern). Redefined from a correct Polish public-holiday calendar instead.
- `sku_age` must anchor to a SKU's first sale across *all* channels/regions, not per group — 14/270
  groups start selling after their SKU's true launch elsewhere, which silently miscomputes age
  (and the `lifecycle_stage` derived from it) if anchored per-group.

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
- **Base-table provenance**: built from raw data by this project's own code, not received
  pre-aggregated (see Data). Re-running Phases 4–11 on it reproduced every headline result from
  the original run — a robustness check on the findings, not just a rebuild.

## Tech Stack

pandas, numpy, scikit-learn, LightGBM, XGBoost, statsmodels / linearmodels, matplotlib, seaborn,
joblib, pytest (badges above).

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
- [x] **Phase 2 — EDA**: profiled all data files; found 3 rows with impossible negative values; confirmed roll-up integrity, no target leakage, and staggered SKU launches usable for cold-start

### Data Preparation
- [x] **Phase 3 — Data validation**: duplicate/leakage checks (all pass); negative-value rows clipped to 0
- [x] **Phase 4 — Weekly base table & feature engineering**: rebuilt the entire weekly table from raw daily data (see Data); diagnosed and fixed the enrichment bug; generalized enrichment to all 30 SKUs; added 4 hypothesis-driven features; added regression tests (`tests/test_build_weekly_base.py`)
- [x] **Phase 5 — Split strategy**: panel-aware walk-forward CV (7 folds + untouched final holdout), zero train/validation overlap verified

### Modeling
- [x] **Phase 6 — Baselines**: Moving Average (4w) best simple baseline, WAPE 0.243
- [x] **Phase 7 — Core forecasting**: global pooled LightGBM wins, **WAPE 0.224** — beats local per-SKU LightGBM (0.256), the baseline, and Holt-Winters ETS (0.301 vs. 0.212 for LightGBM on the same subset); a global pooled XGBoost challenger scores nearly identically (0.225, fold-level std dev checked), confirming the result is robust to boosting library choice, not an artifact of one implementation; LightGBM carried forward as the primary model
- [x] **Phase 8 — Promotion effect**: two-way fixed-effects regression, **+28.4% uplift [27.6%, 29.3%], p < 0.001**, consistent ~28–29% across all 5 categories
- [x] **Phase 9 — Seasonality & trend**: STL decomposition per category — seasonal variance share from 8% (Milk, trend-dominated) to 87% (SnackBar)
- [x] **Phase 10 — Cold-start forecasting**: analog-matching vs. meta-learner vs. full model on 5 held-out new SKUs — both ML approaches clearly beat analog matching at every age
- [x] **Phase 11 — Feature ablation**: calendar/lifecycle features matter most to accuracy, ahead of lag/rolling history; price and external enrichment add ~0 incrementally

### Evaluation
- [ ] **Phase 12 — Model evaluation rollup**: consolidate all models into one comparison table
- [ ] **Phase 13 — Future holdout backtest**: test against real, never-seen January 2025 data

### Deployment & Communication
- [ ] **Phase 14 — Communication deliverable**: `reports/final_report.md` with business-framed findings
- [ ] **Phase 15 — Documentation & polish**: finalize README, docstrings, tests

**Working note**: notebooks include detailed Thai-language explanations for non-technical
readers. `reports/final_report.md` stays in English for a hiring-manager audience.

## Key Findings

- **Forecasting**: Global pooled LightGBM reaches **WAPE 0.224** on 7-fold walk-forward CV, ahead of the best baseline (0.243), local per-SKU LightGBM (0.256), and Holt-Winters ETS (0.301 on the same top-5-series subset where LightGBM scores 0.212). A global pooled XGBoost run on identical folds/features scores 0.225 — a robustness check on the boosting-library choice, not a separate model carried forward
- **Promotions**: **+28.4% sales uplift [27.6%, 29.3%], p < 0.001**, consistent across all 5 categories
- **Seasonality**: variance share ranges from 8% (Milk, trend-dominated) to 87% (SnackBar) — category-dependent, not a single business-wide factor
- **Cold start**: both ML approaches clearly beat naive analog-matching at every SKU age; the full model held up from the first available week
- **Feature value**: calendar/lifecycle features matter most, ahead of lag/rolling history; price and external enrichment add close to nothing incrementally — a predictive-value finding, distinct from promotion's causal effect above. Caveat: `avg_temp`/`inflation_index` are synthetic, deterministic stand-ins (see Data), not real weather/inflation records, and are largely redundant with `month`/`is_summer`/`is_winter` already in the model — so this result shows those *specific proxies* add nothing here, not that real external data wouldn't help actual FMCG demand planning

Every finding above was reproduced by re-running Phases 4–11 on an independently rebuilt weekly
base table (see Data), with every headline number matching the original run within rounding.
