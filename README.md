# FMCG Weekly Sales Forecasting

![Python](https://img.shields.io/badge/Python-3.9-3776AB?style=flat&logo=python&logoColor=white&labelColor=1a1a2e)
![pandas](https://img.shields.io/badge/Pandas-2.3-150458?style=flat&logo=pandas&logoColor=white&labelColor=1a1a2e)
![NumPy](https://img.shields.io/badge/NumPy-2.0-013243?style=flat&logo=numpy&logoColor=white&labelColor=1a1a2e)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6-F7931E?style=flat&logo=scikitlearn&logoColor=white&labelColor=1a1a2e)
![LightGBM](https://img.shields.io/badge/LightGBM-4.6-3499CD?style=flat&labelColor=1a1a2e)
![statsmodels](https://img.shields.io/badge/statsmodels-0.14-8CAAE6?style=flat&labelColor=1a1a2e)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.9-11557C?style=flat&logo=matplotlib&logoColor=white&labelColor=1a1a2e)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat&logo=jupyter&logoColor=white&labelColor=1a1a2e)
![License](https://img.shields.io/badge/License-MIT-4CAF50?style=flat&labelColor=1a1a2e)

![Progress](https://img.shields.io/badge/Progress-Complete_(12%2F16)-4CAF50?style=flat&labelColor=1a1a2e)

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

## Status

### 1. Business Understanding
- [x] Phase 0 — Business framing (framed each of the 5 use cases — forecasting, promotions, seasonality, cold start, feature engineering — as a concrete business question with a success metric per thread; documented in the Problem Statement section above)

### 2. Project Setup
- [x] Phase 1 — Repo & environment setup (created the project folder structure `data/{raw,interim,processed}`, `notebooks/`, `src/{data,features,splits,models,causal,viz}`, `reports/figures`, `tests/`; moved the original raw data from `Data/` into `data/raw/` unchanged; added `requirements.txt`, `.vscode/settings.json`, `.claude/settings.json`; initialized git and connected it to the GitHub remote `thitiwutsuk/Forecasting-FMCG-Daily-Sales`; wrote the README and later translated it fully to English)

### 3. Data Understanding
- [x] Phase 2 — EDA (wrote `notebooks/02_eda.ipynb` profiling all 4 data files; confirmed zero missing values but found 3 rows with simultaneous negative `units_sold`/`stock_available`/`delivered_qty` in `FMCG_2022_2024.csv` — flagged for Phase 3; plotted distributions and saved figures to `reports/figures/`; verified the panel is complete but unbalanced due to staggered SKU launches, usable for Phase 10 cold-start; confirmed the daily→weekly roll-up convention with 0 mismatches across all 31,027 weeks; sanity-checked `target_next_week` for leakage; confirmed the `avg_temp`/`inflation_index` channel bug and that the 2025 batch files are a valid future holdout for Phase 13)

### 4. Data Preparation
- [x] Phase 3 — Data validation (wrote `src/data/validate.py` for duplicate/schema/leakage checks; ran all checks in `notebooks/03_data_validation.ipynb` confirming zero duplicate rows and zero leakage across every derived feature; clipped the 3 negative-value rows found in Phase 2 to 0 rather than dropping them, then re-ran all 16 checks and confirmed all pass; saved the cleaned file to `data/interim/daily_validated.csv`)
- [x] Phase 4 — Feature engineering & enrichment generalization (diagnosed the `avg_temp`/`inflation_index` bug quantitatively — channel noise larger than the true 3-year signal — and regenerated both deterministically at the correct grain instead of patching; wrote `src/features/enrich.py` generalizing enrichment to all 30 SKUs; verified `price_avg`/`promo_rate`/`stock_avg`/`deliveries` against the MI-006 prototype with 0 mismatches; wrote `src/features/engineer.py` adding 4 hypothesis-driven features — `price_index`, `promo_recency`, `rolling_promo_rate`, `cross_channel_demand_share`; saved the final table to `data/processed/weekly_features.csv`, 31,027 rows × 40 columns)
- [x] Phase 5 — Split strategy (wrote `src/splits/walk_forward.py`, a panel-aware `WalkForwardSplitter` that splits by week, never row index; scheme is an 85-week initial training window, 7 expanding-window validation folds through Oct 2024, and a final untouched 10-week holdout reserved for Phase 13; verified no train/validation overlap and saved the scheme diagram to `reports/figures/`)

### 5. Modeling
- [x] Phase 6 — Baseline models (wrote `src/models/baseline.py` and `src/models/metrics.py`; evaluated on the Phase 5 walk-forward folds — Moving Average (4w) is the best baseline at **WAPE 0.243**, ahead of Naive (0.304) and Seasonal Naive (0.356); sets the bar Phase 7 must beat)
- [x] Phase 7 — Core forecasting, use case 1 (wrote `src/models/forecast.py`; compared 4 model families on identical CV folds — **global pooled LightGBM wins at WAPE 0.224**, beating local per-SKU LightGBM (0.256), the Moving Average baseline (0.243), and Holt-Winters ETS (0.301 vs. 0.216 for LightGBM on the same top-5-series subset); confirms pooling across SKUs beats per-SKU models; carried forward into Phase 10, 11, and 13)
- [x] Phase 8 — Promotion effect, use case 2 (wrote `src/causal/promotion_effect.py`, a two-way fixed-effects `PanelOLS`; overall uplift **+28.4% [27.6%, 29.3%], p < 0.001**, consistent ~28–29% across all 5 categories; found confounding is mild in this simulated dataset; Juice handled as a documented single-cluster edge case)
- [x] Phase 9 — Seasonality & trend, use case 3 (ran per-category STL decomposition; seasonality dominates SnackBar (87%), ReadyMeal (76%), and Juice (73%); trend dominates Milk (58% vs. 8% seasonal); Yogurt is mixed (53%/32%))
- [x] Phase 10 — Cold-start forecasting, use case 4 (wrote `src/models/cold_start.py` with analog matching and a meta-learner; held out the 5 genuinely latest-launching SKUs entirely; found the full model matched or beat the meta-learner at every age bucket since true zero-history rows aren't present in the table — documented as a scope caveat; both ML approaches clearly beat analog matching at every age)
- [x] Phase 11 — Feature ablation study, use case 5 (retrained the global LightGBM across every CV fold for each of 6 feature-group removals; calendar & lifecycle features matter most, followed by lag/rolling; promotion and operational features have a small effect; price and external enrichment show ~0 marginal effect; cross-validates the Phase 10 finding and is flagged as a predictive-value result, distinct from Phase 8's causal estimate)

### 6. Evaluation
- [ ] Phase 12 — Model evaluation rollup (consolidate all models into a single comparison table)
- [ ] Phase 13 — Future holdout backtest (test the model against real, never-seen January 2025 data)

### 7. Deployment & Communication
- [ ] Phase 14 — Communication deliverable (produce `reports/final_report.md` with figures and business-framed findings)
- [ ] Phase 15 — Documentation & polish (finalize the README, docstrings, and tests)

**Working note**: notebooks in `notebooks/` include detailed Thai-language explanations of each step, so readers without a computer science background can follow along. The final report (`reports/final_report.md`) stays in English for a hiring-manager audience.

## Key Findings

- **Forecasting**: Global pooled LightGBM reaches **WAPE 0.224** on 7-fold walk-forward CV, beating the best baseline (Moving Avg 4w, 0.243), a local per-SKU LightGBM (0.256), and Holt-Winters ETS (0.301 on the same top-5-series subset where LightGBM scores 0.216)
- **Promotions**: Two-way fixed-effects regression estimates a **+28.4% sales uplift [27.6%, 29.3%], p < 0.001**, consistent (~28–29%) across all 5 categories
- **Seasonality**: Seasonal variance share ranges from 8% (Milk, trend-dominated) to 87% (SnackBar) — category-dependent, not a single "seasonality factor" for the business
- **Cold start**: Both ML approaches (full model, meta-learner) clearly beat naive analog-matching at every SKU age; the full model held up from the first available week rather than needing a "catch-up" period, which the ablation study corroborates
- **Feature value**: Calendar/lifecycle features matter most to predictive accuracy, ahead of lag/rolling sales history; price and external enrichment add close to nothing incrementally once other features are present — this is a predictive-value finding, distinct from promotion's causal effect above
