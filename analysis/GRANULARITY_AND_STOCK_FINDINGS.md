# Weekly vs. daily granularity, and stock/delivery enrichment semantics

**Status: preliminary sensitivity analysis.** Exploratory evidence produced to check a
hypothesis raised outside this project (a Kaggle discussion on the source dataset
claiming daily-level modeling can be "significantly more effective" than weekly). Not
a confirmatory result, not peer-reviewed methodology, and not a proposal to change the
Phase 7 model. Numbers below come from `analysis/preliminary_granularity_check.py`,
which is not part of the Phase 0-15 pipeline.

## What was tested

Three LightGBM specifications, run on the same 7 walk-forward folds Phase 5 already
defines (`WalkForwardSplitter`, `initial_train_end=2023-09-30`, 8-week validation
windows):

1. **Compact weekly** — one next-week total per SKU/channel/region, using a small
   hand-picked feature set (14 features: SKU/channel/region/category/segment/brand,
   `units_sold`, `lag_1`, `lag_2`, `rolling_mean_4`, `rolling_std_4`, target
   week-number/month/year). This script's own baseline, not the project's production
   model.
2. **Daily fixed-origin** — seven daily forecasts made using only information
   available at the end of the origin week (18 features: the same categoricals plus
   `daily_lag_7/14/28`, `horizon`, `target_dow`, target month/week-number), then
   summed to the same next-week total. Source-date assertions verify that each daily
   lag comes from no later than the end of the origin week. A separate assertion
   verifies outcome alignment: the seven daily targets sum to the project's own
   `target_next_week` before scoring.
3. **Daily rolling** — one-day-ahead forecasts that use actual sales from days
   already past that week. **Not leakage** — a system that reforecasts daily is
   legitimately entitled to see days that already happened — but it is a different
   service level (reforecast-every-day) than a plan frozen before the week starts, so
   it isn't a fair comparison to (1) or (2).

The project's actual Phase 7 production model (full feature set, `ALL_FEATURE_COLS`)
is also scored on the same weekly folds, for context — but comparing it directly
against the daily variants is confounded by feature-set differences on top of
granularity, so that comparison is reported separately from the granularity claim.

**Caveat that applies to every p-value below:** the 7 folds are expanding-window and
share overlapping training data — they are not independent samples. The paired
t-test assumes independence and therefore overstates confidence. Wilcoxon and a sign
test are reported alongside it, but neither one fixes the non-independence; they are
included to show the result isn't an artifact of the t-test's normality assumption,
not to manufacture statistical rigor the fold design doesn't support. Read every
p-value here as exploratory direction-and-magnitude evidence, not a confirmatory test.

## Result 1: direct weekly beats daily-then-sum, on the closest-to-fair pair

| | mean WAPE |
|---|---|
| Compact weekly (this script) | 0.2295 |
| Daily fixed-origin, summed | 0.2376 |
| Project's production weekly model | 0.2235 |
| Daily rolling | 0.2223 |

Compact weekly vs. daily fixed-origin is the pair to read for a granularity claim,
because both use the same code path and a comparable (in fact the daily side is
larger — 18 vs. 14 features) feature philosophy, so the difference isn't attributable
to one side being starved of information:

- Daily fixed-origin was worse in every one of 7/7 folds (mean +0.0081 WAPE, ≈3.5%
  relative).
- Paired t-test p = 0.0049; Wilcoxon p = 0.0156; two-sided sign test p = 0.0156.

**Claim actually supported:** under this benchmark's feature design and model
configuration, direct-weekly forecasting outperformed fixed-origin daily-then-aggregate
consistently across all 7 folds tested. **Claim NOT supported:** that this proves
weekly is fundamentally better than daily for this problem. The daily model was not
tuned separately from the weekly one (same hyperparameters, same `n_estimators`,
`learning_rate`, `num_leaves`), it optimizes daily squared error rather than the
weekly-aggregate WAPE actually being scored, and only 7 correlated folds back this up.

The daily-vs-project comparison (+0.0141 WAPE, p = 0.0006) tells a similar direction
but should not be quoted as a granularity result — the project's production model
uses a materially different, larger feature set, so this comparison mixes a
feature-set effect with a granularity effect and can't separate the two.

## Result 2: daily rolling is not proven better — it's answering a different question

Daily rolling scored 0.2223 vs. the project's 0.2235 (p = 0.53, not significant). This
is not evidence that finer granularity wins: daily rolling benefits from a later
forecast origin and daily information updates within the target week, so it is not
directly comparable to a forecast frozen before the week begins. The correct reading
is "no evidence either way from this comparison," not "daily ties or beats weekly."

## Candidate explanations for daily-then-sum's higher WAPE (not confirmed causes)

Framed as "consistent with," not "caused by" — daily errors don't have to accumulate
(positive and negative errors can cancel on aggregation); the following are measured
facts consistent with several possible mechanisms, not a demonstrated causal chain:

- **Daily target is noisier**: mean coefficient of variation of `units_sold` per
  (sku, channel, region) is 0.578 at daily grain vs. 0.344 at weekly grain — daily is
  ~1.68x noisier by this measure.
- **~15% of the daily panel is filled, not observed**: 190,757 of 224,403
  (sku, channel, region, date) combinations spanning each group's active life are
  actually present (85.01%); the remaining 14.99% are filled as zero sales because
  the project's own weekly sum already treats an absent row as contributing zero.
  That's an inherited assumption, not a proven fact about true zero-demand days, and
  it affects a non-trivial share of what the daily models are scored against.
  Note this cuts against the daily-rolling result being an unqualified positive too:
  the same filled-zero rows feed its lag/rolling features.
- **Feature/hyperparameter mismatch**: the daily model reuses the weekly problem's
  LightGBM configuration untouched, and optimizes a per-day L2 loss rather than the
  weekly-aggregate WAPE it's ultimately scored on.

## Result 3: stock_available — narrower conclusion than first drafted

A Kaggle Q&A thread with the dataset's creator described `stock_available` with an
illustrative (her word: "simplified") formula, `stock = stock + delivered_qty -
units_sold` clipped at 0, grouped by `(sku, region)` — i.e., not by channel.

`stock_audit()` checked two things against the Phase 3 validated daily file:

- Of 190,667 adjacent-row transitions within `(sku, region)`, only 203 (~0.1%)
  match that formula exactly. These comprise 116,168 same-date cross-channel
  transitions (126 matches), 74,256 transitions to the next calendar day (77
  matches, also ~0.1%), and 243 transitions across gaps longer than one day (0
  matches). “Adjacent row” is the literal check of the illustrative row-by-row
  snippet; “next calendar day” is the narrower check of its verbal beginning-of-day
  interpretation.
- Of 70,261 dates where a `(sku, region)` had ≥2 channels selling, only 139 (~0.2%)
  show the same `stock_available` value across those channels.

**What this rules out:** a single shared stock-available scalar copied across every
channel that day, and Beata Faron's shared-pool snippet as the literal generator of
this file. **What it does NOT establish:** that each channel's stock is actually
independent. The snippet itself updates a shared running total row-by-row in
whatever order the channels happen to sort in — under that exact code, two channels
on the same day would *also* show different stock values, just because of processing
order, not because the pool is split. So differing per-channel values are consistent
with both "independent per-channel stock" and "a shared pool processed sequentially
with some other allocation rule." This dataset alone can't distinguish the two —
"proves stock is tracked independently per channel" was an overclaim and has been
withdrawn.

## Result 4 (new): two enrichment-pipeline semantics findings

Found while building the audits above, in `src/features/enrich.py`,
`compute_internal_aggregates()`:

- **`deliveries=("delivery_days", "count")` literally counts observed rows.**
  `delivery_days` (a per-row lead-time figure) is never null across all 190,757
  validated rows, and `delivered_qty > 0` on 99.998% of rows — all but the same 3
  Phase-3-clipped rows. The resulting `deliveries` value therefore equals the number
  of observed daily rows in each group-week. This is not automatically a bug:
  because almost every observed row has a positive delivery, the count is also
  numerically almost identical to the number of positive-delivery days. Its business
  meaning depends on an unresolved source-data question: if an absent date means
  “no activity,” it can act as delivery frequency; if it means “unobserved/missing
  record,” it is a completeness count. The implementation is nevertheless
  semantically fragile because it counts a different column from the event it
  purports to measure. A clearer definition would count `delivered_qty > 0`
  explicitly and document the absent-date assumption.
- **`stock_avg=("stock_available", "mean")` is calculated correctly but is
  redundant.** A fresh recomputation over all 31,027 modeled group-weeks matches
  both stored `stock_avg` and stored weekly `stock_available` exactly. Thus
  `stock_avg` is not a calculation bug, but the two columns are duplicate model
  features (100% equality), because both originate from the same weekly mean of raw
  `stock_available`. Keep one canonical feature or explicitly document why both are
  retained.

## What this analysis does not establish

- That weekly is provably superior to daily for this forecasting problem in general
  (only that this particular untuned daily specification lost on this particular
  benchmark, across correlated folds).
- That `stock_available` is tracked independently per channel (ruled out one
  mechanism, not proved another).
- No direct source-pipeline bug is established here. `deliveries` has ambiguous
  semantics, and `stock_avg` is a correctly calculated but redundant feature; both
  warrant a documentation or design decision before any `src/` change.

## If pursued further

- Tune the daily model separately (hyperparameter search on the daily objective, or
  switch its loss to something closer to WAPE) before concluding weekly's advantage
  is about granularity rather than tuning effort.
- A block-bootstrap or fold-shuffling scheme to get a defensible confidence interval
  given the folds' non-independence, instead of the plain paired t-test.
- Decide whether `deliveries` should explicitly count positive `delivered_qty` days
  or be documented as an observed-row count, after clarifying what absent dates
  mean. Remove or justify the duplicate `stock_avg`/`stock_available` pair, then
  update `MODEL_SELECTION_METHODOLOGY.md` / Phase 11's feature list if the feature
  set changes.
