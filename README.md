# FMCG Weekly Sales Forecasting

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

**30 SKUs** across 5 categories: Yogurt (11), Milk (7), Snack (6), ReadyMeal (5), Juice (1), sold
through 3 channels (Discount, E-commerce, Retail) and 3 regions (PL-Central, PL-North, PL-South).
SKUs launch on staggered real dates between Feb 2022 and Jun 2023, which this project uses as
naturally-occurring cold-start cases rather than synthetic ones.

**Known data quality issue:** in the MI-006 enrichment prototype, `avg_temp` and
`inflation_index` incorrectly vary by sales channel within the same week/region — both are fixed
before the enrichment pipeline is generalized to all 30 SKUs (see Methodology, Phase 4).

## Methodology

The project follows a standard data science lifecycle, organized into 16 phases (see
`notebooks/`, numbered to match):

| Phase | Focus | Output |
|---|---|---|
| 0 | Business framing | Problem statement per use case, success metrics |
| 1 | Repo & environment setup | Runnable project skeleton |
| 2 | EDA | Data dictionary, distributions, data-quality notes |
| 3 | Data validation | Leakage checks, schema/range checks |
| 4 | Feature engineering | Enrichment bug fix, generalized to all 30 SKUs, hypothesis-driven features |
| 5 | Split strategy | Panel-aware, time-respecting walk-forward CV |
| 6 | Baseline models | Naive, seasonal naive, moving average |
| 7 | Core forecasting (use case 1) | Statistical model + global pooled LightGBM |
| 8 | Promotion effect (use case 2) | Two-way fixed-effects regression with CIs |
| 9 | Seasonality & trend (use case 3) | STL decomposition per category |
| 10 | Cold-start forecasting (use case 4) | Analog-based + meta-learner approaches |
| 11 | Feature ablation study (use case 5) | Quantified value of each feature group |
| 12 | Model evaluation rollup | Final comparison table across all models |
| 13 | Future holdout backtest | Simulated live scoring against Jan 2025 data |
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

pandas, numpy, scikit-learn, LightGBM, statsmodels / linearmodels, matplotlib, seaborn, joblib.

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

Project scoping, planning, and repo scaffolding complete (Phase 0–1). Phase 2 (EDA) is next —
see the phase table above for what's done and what's next.

## Key Findings

_To be filled in as phases complete — headline results (forecast accuracy vs. baseline,
promotion uplift estimate with confidence interval, cold-start accuracy curve) will be
summarized here once available._
