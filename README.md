# FMCG Weekly Sales Forecasting

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikitlearn&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-3499CD?style=flat)
![statsmodels](https://img.shields.io/badge/statsmodels-3776AB?style=flat)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat&logo=matplotlib&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)

A data science portfolio project on demand forecasting, promotion effectiveness, and cold-start
forecasting for a simulated FMCG (Fast-Moving Consumer Goods) business selling across multiple
channels and regions.

## Problem Statement

A demand planner at an FMCG company needs answers to five recurring questions:

1. **Forecasting** — How many units of each SKU should we expect to sell next week, by channel
   and region?
2. **Promotions** — Do promotions actually pay for themselves, or do they just pull forward sales
   that would have happened anyway?
3. **Seasonality** — How much of a category's sales swing is explained by seasonality and trend
   versus other factors, and does that pattern differ by category?
4. **Cold start** — When a new SKU launches with no sales history, what do we forecast for it,
   and how quickly does forecast accuracy catch up to a mature product?
5. **Feature engineering** — Which engineered features (lags, rolling stats, price, promotions,
   external enrichment) actually move forecast accuracy, and by how much?

This project answers each question end-to-end: from raw transactional data to validated,
back-tested models and a written analysis with business-interpretable results.

## Data

| File | Grain | Rows | Description |
|---|---|---|---|
| `data/raw/FMCG_2022_2024.csv` | sku × channel × region × date (daily) | 190,758 | Raw daily transactions, Jan 2022–Dec 2024 |
| `data/raw/weekly_df_final_for_modeling.csv` | sku × channel × region × week | 31,027 | Main weekly modeling table with lag/rolling/calendar features and the `target_next_week` label |
| `data/raw/df_weekly_MI-006_enriched.csv` | sku × channel × region × week (MI-006 only) | 1,349 | Prototype enrichment with weather, inflation, school calendar, category trend, and event score |
| `data/raw/batch_MI-006_2025-01-*.parquet` | daily, MI-006 only | 4 files, ~135 rows each | January 2025 data, after the training window — used as a true future holdout |
| `data/interim/daily_validated.csv` | sku × channel × region × date (daily) | 190,758 | Phase 3 output: raw daily data with the 3 negative-value rows clipped to 0 |
| `data/processed/weekly_features.csv` | sku × channel × region × week | 31,027 | Phase 4 output: final modeling table — validated core features + regenerated enrichment (all 30 SKUs) + hypothesis-driven features |

**30 SKUs** across 5 categories: Yogurt (11), Milk (7), Snack (6), ReadyMeal (5), Juice (1), sold
through 3 channels (Discount, E-commerce, Retail) and 3 regions (PL-Central, PL-North, PL-South).
SKUs launch on staggered real dates between Feb 2022 and Jun 2023, which this project uses as
naturally-occurring cold-start cases rather than synthetic ones.

**Known data quality issue (fixed in Phase 4):** in the MI-006 enrichment prototype, `avg_temp`
and `inflation_index` incorrectly varied by sales channel within the same week/region. The
channel-level noise turned out to be larger than the true 3-year trend/seasonal signal, so both
were regenerated deterministically at the correct grain (region×week / week) rather than patched,
and the enrichment pipeline was generalized to all 30 SKUs — see `notebooks/04_feature_engineering.ipynb`.

## Methodology

The project follows a standard data science lifecycle. Each of the 16 phases below maps to a
numbered notebook in `notebooks/` and belongs to one stage of that lifecycle:

### 1. Business Understanding
| Phase | Focus | Output |
|---|---|---|
| 0 | Business framing | Problem statement per use case, success metrics |

### 2. Project Setup
| Phase | Focus | Output |
|---|---|---|
| 1 | Repo & environment setup | Runnable project skeleton |

### 3. Data Understanding
| Phase | Focus | Output |
|---|---|---|
| 2 | Exploratory data analysis (EDA) | Data dictionary, distributions, data-quality notes |

### 4. Data Preparation
| Phase | Focus | Output |
|---|---|---|
| 3 | Data validation | Leakage checks, schema/range checks |
| 4 | Feature engineering | Enrichment bug fix, generalized to all 30 SKUs, hypothesis-driven features |
| 5 | Split strategy | Panel-aware, time-respecting walk-forward CV |

### 5. Modeling
| Phase | Focus | Output |
|---|---|---|
| 6 | Baseline models | Naive, seasonal naive, moving average |
| 7 | Core forecasting (use case 1) | Statistical model + global pooled LightGBM |
| 8 | Promotion effect (use case 2) | Two-way fixed-effects regression with CIs |
| 9 | Seasonality & trend (use case 3) | STL decomposition per category |
| 10 | Cold-start forecasting (use case 4) | Analog-based + meta-learner approaches |
| 11 | Feature ablation study (use case 5) | Quantified value of each feature group |

### 6. Evaluation
| Phase | Focus | Output |
|---|---|---|
| 12 | Model evaluation rollup | Final comparison table across all models |
| 13 | Future holdout backtest | Simulated live scoring against Jan 2025 data |

### 7. Deployment & Communication
| Phase | Focus | Output |
|---|---|---|
| 14 | Communication deliverable | `reports/final_report.md` with figures |
| 15 | Documentation & polish | Final README, docstrings, tests |

Full detail for each phase is in the project plan.

## Approach highlights

- **Splitting**: strictly time-based (by week), never random — every model is evaluated on
  weeks it has never seen.
- **Promotion effect**: estimated via two-way fixed-effects regression (SKU + week/season fixed
  effects, controlling for price), not a naive promo-vs-non-promo comparison, since promotions
  are confounded with price cuts, channel, and season.
- **Cold start**: validated against real staggered SKU launches, compared using a
  similarity/analog method and a meta-learner (global model restricted to features available at
  launch time).
- **Metrics**: WAPE and SMAPE are used alongside MAE/RMSE, since MAPE is unstable for
  low-volume SKU-weeks.

## Tech Stack

pandas, numpy, scikit-learn, LightGBM, statsmodels / linearmodels, matplotlib, seaborn, joblib
(badges at the top of this README).

## Repository Structure

```
Forecasting FMCG Daily Sales/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/            # original data, untouched
│   ├── interim/         # validated/cleaned intermediate tables
│   └── processed/       # final modeling-ready feature tables
├── notebooks/           # numbered, one per phase
├── src/
│   ├── data/            # loading + validation
│   ├── features/        # feature-engineering functions
│   ├── splits/           # walk-forward / panel-aware CV
│   ├── models/           # baseline, LightGBM, statsmodels wrappers
│   ├── causal/            # fixed-effects promotion-uplift estimator
│   └── viz/               # shared plotting helpers
├── reports/
│   ├── figures/
│   └── final_report.md
└── tests/
```

## Progress

### 1. Business Understanding
- [x] **Phase 0 — Business framing**
  - Framed each of the 5 use cases as a concrete business question (forecasting, promotions, seasonality, cold start, feature engineering)
  - Defined a success metric per thread
  - Documented in the Problem Statement section above

### 2. Project Setup
- [x] **Phase 1 — Repo & environment setup**
  - Created the project folder structure: `data/{raw,interim,processed}`, `notebooks/`, `src/{data,features,splits,models,causal,viz}`, `reports/figures`, `tests/`
  - Moved the original raw data files from `Data/` into `data/raw/` unchanged
  - Added `requirements.txt` with the project's tooling (pandas, scikit-learn, LightGBM, statsmodels, linearmodels, etc.)
  - Added `.vscode/settings.json` to hide `.venv/` from the file explorer and search
  - Added `.claude/settings.json` with a permission allowlist for activating the project's `.venv`
  - Initialized the git repo, made the initial commit, and connected it to the GitHub remote (`thitiwutsuk/Forecasting-FMCG-Daily-Sales`)
  - Wrote the initial README (problem statement, data inventory, methodology, repo structure) and later translated it fully to English

### 3. Data Understanding
- [x] **Phase 2 — EDA**
  - Created `.venv` and installed `requirements.txt` (note: LightGBM fails to import on this macOS setup until `libomp` is installed via Homebrew — not needed until Phase 7, tracked as a known setup issue)
  - Wrote `notebooks/02_eda.ipynb`: loaded and profiled all 4 data files (schema, dtypes, shape)
  - Confirmed zero missing values across all files, but found 3 rows in `FMCG_2022_2024.csv` with simultaneous negative `units_sold`/`stock_available`/`delivered_qty` (SKUs SN-028, SN-010, RE-007) — flagged for Phase 3
  - Plotted distributions of `units_sold`, `price_unit`, promotion rate by category, and channel sales split; saved figures to `reports/figures/`
  - Verified the panel is complete (270 = 30 SKU × 3 channel × 3 region combos) but unbalanced, since SKUs launch on staggered dates — confirmed usable for Phase 10 cold-start
  - Reverse-engineered and confirmed the daily→weekly roll-up convention (`W-MON`, `label='left'`, `closed='left'`) by recomputing it from `FMCG_2022_2024.csv` and diffing against `weekly_df_final_for_modeling.csv` across all 270 combos (31,027 weeks, 0 mismatches)
  - Sanity-checked `target_next_week` against a manual `groupby().shift(-1)` — matched on every comparable row, no leakage found at this pass (full feature-level leakage check deferred to Phase 3)
  - Confirmed the known `avg_temp`/`inflation_index` enrichment bug in `df_weekly_MI-006_enriched.csv` varies incorrectly by channel within the same week/region
  - Confirmed the future holdout batches (`batch_MI-006_2025-01-*.parquet`) fall entirely after the training window, usable as a true backtest in Phase 13

### 4. Data Preparation
- [x] **Phase 3 — Data validation**
  - Wrote `src/data/validate.py`: reusable checks for duplicate keys, schema/range violations, the daily→weekly roll-up convention, and leakage in `target_next_week`, `lag_1`, `lag_2`, `rolling_mean_4`, `rolling_std_4`, and `momentum`
  - Ran all checks in `notebooks/03_data_validation.ipynb`: confirmed zero duplicate rows, zero leakage across every derived feature (each verified against a `shift`-based recomputation over all 31,027 rows), and reconfirmed the roll-up convention from Phase 2
  - Found the only failures were the 3 negative-value rows in `FMCG_2022_2024.csv` identified in Phase 2 (`units_sold`, `stock_available`, `delivered_qty` all negative on the same rows); the weekly modeling table was unaffected since same-week positive days absorbed them
  - Decided to clip those values to 0 rather than drop the rows, to keep each SKU's daily time series contiguous ahead of the Phase 4 re-aggregation; re-ran all checks post-clip and confirmed 16/16 pass
  - Saved the cleaned file to `data/interim/daily_validated.csv` as the input for Phase 4
- [x] **Phase 4 — Feature engineering & enrichment generalization**
  - Diagnosed the `avg_temp`/`inflation_index` channel-leakage bug quantitatively in `notebooks/04_feature_engineering.ipynb`: the within-week-across-channel noise (std ≈ 7-9 for temp, ≈ 44-46 for inflation) is larger than the true 3-year trend/seasonal range in the prototype file, so the original signal can't be recovered — regenerated both deterministically instead of patching
  - Wrote `src/features/enrich.py`: regenerates `avg_temp` (region × week), `inflation_index` (week, national), `school_in_session` (week, calendar rule), `event_score` (week, Polish retail-holiday calendar), and `category_trend` (category × region × week, causal expanding index) — none depend on `channel`, since none plausibly should
  - Verified `price_avg`, `promo_rate`, `stock_avg`, `deliveries` formulas against the MI-006 prototype (0 mismatches over 1,349 rows) before generalizing them to all 30 SKUs
  - Confirmed the fix: std of `avg_temp`/`inflation_index` across channel within the same week/region is now exactly 0
  - Wrote `src/features/engineer.py` adding 4 hypothesis-driven features — `price_index`, `promo_recency`, `rolling_promo_rate`, `cross_channel_demand_share` — each designed to use only same-or-past-week information (no leakage into `target_next_week`)
  - Saved the final table to `data/processed/weekly_features.csv` (31,027 rows × 40 columns, all 30 SKUs, no duplicate keys)
- [x] **Phase 5 — Split strategy**
  - Wrote `src/splits/walk_forward.py`: a panel-aware `WalkForwardSplitter` that splits by `week` (never row index), so every sku-channel-region cell for a week stays on the same side of the split
  - Scheme (`notebooks/05_split_strategy.ipynb`): initial training window 2022-02-14→2023-09-25 (85 weeks), 7 expanding-window validation folds through 2024-10-14 (8 weeks each), final untouched holdout = last 10 weeks of 2024 (2024-10-21→2024-12-23) for Phase 13 only
  - Verified no train/validation row overlap, training always strictly precedes validation, and the final holdout is fully disjoint from CV — diagram saved to `reports/figures/phase5_walk_forward_scheme.png`

### 5. Modeling
- [ ] **Phase 6 — Baseline models**: naive, seasonal naive, moving average
- [ ] **Phase 7 — Core forecasting (use case 1)**: statistical model + global pooled LightGBM
- [ ] **Phase 8 — Promotion effect (use case 2)**: two-way fixed-effects regression with confidence intervals
- [ ] **Phase 9 — Seasonality & trend (use case 3)**: STL decomposition per category
- [ ] **Phase 10 — Cold-start forecasting (use case 4)**: analog-based + meta-learner forecasting for new SKUs
- [ ] **Phase 11 — Feature ablation study (use case 5)**: quantify the value each feature group adds to model accuracy

### 6. Evaluation
- [ ] **Phase 12 — Model evaluation rollup**: consolidate all models into a single comparison table
- [ ] **Phase 13 — Future holdout backtest**: test the model against real, never-seen January 2025 data

### 7. Deployment & Communication
- [ ] **Phase 14 — Communication deliverable**: produce `reports/final_report.md` with figures and business-framed findings
- [ ] **Phase 15 — Documentation & polish**: finalize the README, docstrings, and tests

**Working note**: notebooks in `notebooks/` include detailed Thai-language explanations of each step, so readers without a computer science background can follow along. The final report (`reports/final_report.md`) stays in English for a hiring-manager audience.

## Key Findings

_To be filled in as phases complete — headline results (forecast accuracy vs. baseline,
promotion uplift estimate with confidence interval, cold-start accuracy curve) will be
summarized here once available._
