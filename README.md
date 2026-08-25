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
- [x] **Phase 0 — Business framing**
  - Turned all 5 use cases into plain business questions: forecasting, promotions, seasonality, new products, and feature value
  - Set a clear way to measure success for each one
  - Written up in the Problem Statement section above

### 2. Project Setup
- [x] **Phase 1 — Repo & environment setup**
  - Set up the project folders: raw/interim/processed data, notebooks, reusable code, reports, tests
  - Moved the original data files into `data/raw/` without changing them
  - Added the tools list (`requirements.txt`) and editor/permission settings
  - Connected the project to GitHub
  - Wrote the README and later translated it fully into English

### 3. Data Understanding
- [x] **Phase 2 — EDA (exploring the data)**
  - Looked through all 4 data files to understand what's really in them
  - Found no missing values, but caught 3 rows with impossible negative numbers — flagged for cleanup
  - Made charts of sales, price, and promotions and saved them for later use
  - Confirmed products launch on different dates, which is useful for testing new-product forecasts later
  - Double-checked that weekly totals match the daily data exactly, no mismatches at all
  - Checked that "next week's sales" (what the model predicts) never accidentally leaks into today's data
  - Confirmed a known bug where weather and inflation numbers wrongly change depending on sales channel
  - Confirmed the January 2025 data is genuinely unseen and safe to use as a final real-world test later

### 4. Data Preparation
- [x] **Phase 3 — Data validation**
  - Wrote reusable checks for duplicate rows, bad values, and data leakage
  - Ran every check — found no duplicates and no leakage anywhere
  - Fixed the 3 negative-value rows found in Phase 2 by setting them to 0 instead of deleting them
  - Saved the cleaned data as a new file, ready for the next step
- [x] **Phase 4 — Feature engineering & fixing the enrichment bug**
  - Figured out exactly why the weather/inflation numbers were wrong, then rebuilt them correctly instead of just patching over the bug
  - Extended this fix to all 30 products (previously this data only existed for 1 product)
  - Double-checked the rebuilt numbers against the original working example — found zero mismatches
  - Added 4 new, purpose-built features: relative price vs. category average, time since last promotion, rolling promotion rate, and cross-channel sales share
  - Saved the final, model-ready data table
- [x] **Phase 5 — Split strategy**
  - Built a custom way to split the data by time (never randomly), so no model ever "sees the future" by accident
  - Set up a training period, 7 rounds of validation through 2024, and one final untouched test set saved for later
  - Verified there is zero overlap between what the model trains on and what it's tested on

### 5. Modeling
- [x] **Phase 6 — Baseline models**
  - Built simple "dumb" forecasts to set a fair bar: guess last week's number, or the last 4-week average
  - The 4-week average was the best simple guess, with a forecast error of about 24.3%
  - This number is the target every real model has to beat
- [x] **Phase 7 — Core forecasting (use case 1)**
  - Trained and compared 4 different forecasting approaches on the exact same test weeks
  - One shared model, trained on all 30 products together, won — cutting the forecast error to about 22.4%
  - It beat a model trained separately per product, the simple baseline, and a classical statistics-based model
  - Key takeaway: one model that learns from every product does better than giving each product its own model
  - This is now the main model carried forward into later phases
- [x] **Phase 8 — Promotion effect (use case 2)**
  - Used a statistical technique that isolates the true effect of promotions from other things that change at the same time, like price or the season
  - Found promotions genuinely lift sales by about 28.4%, with a reliable confidence range of 27.6%–29.3%
  - This effect size is similar (~28–29%) across every product category
  - One product category (Juice, with only 1 SKU) needed a slightly different statistical setup, handled and documented separately
- [x] **Phase 9 — Seasonality & trend (use case 3)**
  - Split each category's sales pattern into 3 parts: long-term trend, repeating seasonal pattern, and random noise
  - Some categories (like Snacks) are driven almost entirely by season (87%); others (like Milk) are driven mostly by long-term growth instead (58%)
  - Takeaway: different product categories need different planning approaches — season-driven vs. growth-driven
- [x] **Phase 10 — Cold-start forecasting (use case 4)**
  - Tested 2 ways to forecast a brand-new product with no sales history: find similar existing products, or use the main model without needing past sales data
  - Held out the 5 genuinely newest products completely and tested both methods against them
  - Both smarter methods clearly beat simple guessing at every stage of a product's early life
  - One early assumption turned out to be wrong once tested — documented that finding honestly instead of forcing the expected result
- [x] **Phase 11 — Feature ablation study (use case 5)**
  - Retrained the model repeatedly, each time removing one group of features, to see what actually matters
  - Calendar and product-lifecycle information mattered the most
  - Past sales history mattered less than expected once other context was already included
  - Price and the external weather/inflation data added almost nothing extra
  - This answers a different question from Phase 8: this measures prediction accuracy, Phase 8 measures true cause-and-effect

### 6. Evaluation
- [ ] **Phase 12 — Model evaluation rollup**
  - Bring every model's results from Phase 6-11 together into one clean comparison table
- [ ] **Phase 13 — Future holdout backtest**
  - Test the final model against real January 2025 data it has never seen before

### 7. Deployment & Communication
- [ ] **Phase 14 — Communication deliverable**
  - Write up all the findings in plain business language with supporting charts
- [ ] **Phase 15 — Documentation & polish**
  - Finish the README, code comments, and tests so the project is easy for anyone to pick up

**Working note**: notebooks in `notebooks/` include detailed Thai-language explanations of each step, so readers without a computer science background can follow along. The final report (`reports/final_report.md`) stays in English for a hiring-manager audience.

## Key Findings

- **Forecasting**: Global pooled LightGBM reaches **WAPE 0.224** on 7-fold walk-forward CV, beating the best baseline (Moving Avg 4w, 0.243), a local per-SKU LightGBM (0.256), and Holt-Winters ETS (0.301 on the same top-5-series subset where LightGBM scores 0.216)
- **Promotions**: Two-way fixed-effects regression estimates a **+28.4% sales uplift [27.6%, 29.3%], p < 0.001**, consistent (~28–29%) across all 5 categories
- **Seasonality**: Seasonal variance share ranges from 8% (Milk, trend-dominated) to 87% (SnackBar) — category-dependent, not a single "seasonality factor" for the business
- **Cold start**: Both ML approaches (full model, meta-learner) clearly beat naive analog-matching at every SKU age; the full model held up from the first available week rather than needing a "catch-up" period, which the ablation study corroborates
- **Feature value**: Calendar/lifecycle features matter most to predictive accuracy, ahead of lag/rolling sales history; price and external enrichment add close to nothing incrementally once other features are present — this is a predictive-value finding, distinct from promotion's causal effect above
