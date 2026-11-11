# 90-Day Sales Forecast: Features, Transformations, Selection, and Range Calibration

## Summary

- **Target.** `avg_sales` is mean daily net sales over a location's first 90 business days. There are n = 1,347 openings from January 2017 to July 2026.
- **Model.** An attribute-aware router of two LightGBM regressors: 15.60% WAPE on the confirmation cohort and 17.66% on the benchmark.
- **Features.** 152 engineered candidates for the attribute model (141 after within-fold filtering). Peer-store sales carry 43% of total split gain. Site attributes lower benchmark WAPE by 1.04 pp.
- **Range.** Forecast residuals are right-skewed and non-normal, so the planning range is set from empirical quantiles rather than a standard deviation. Moving the cap to 25% and the quantile to 70% raised walk-forward coverage from 55.0% to 65.3% (+10.2 pp, 95% CI [+7.9, +12.7], p < 10⁻¹⁸).
- **Key caveat.** At equal mean width, the dynamic (subtype × region) stretch is statistically indistinguishable from a fixed symmetric range (+0.8 pp, p = 0.47). Coverage is driven by width, not by the dynamic allocation.

## Data and validation design

| Item | Specification |
|---|---|
| Unit | One location; label = mean daily net sales over business days 1–90 after opening |
| Business day | Trading date with positive net sales; the label is undefined until all 90 exist |
| Label filter | $5,000 ≤ `avg_sales` ≤ $70,000 (removed 0 rows in the modeling table) |
| Maturity gap | Training rows must open ≥ 90 days before each validation origin |
| Confirmation | Openings 2025-01-01 to 2025-04-01 (n = 33); model trained on openings before 2024-10-03 |
| Benchmark | Openings from 2025-07-01 (n = 210); model trained on openings before 2025-04-02 |
| Walk-forward | 10 half-year origins, 2021-01 to 2025-07 (877 forecasts), used for range calibration |

**Leakage controls.** Predictors are restricted to information available at opening. The location's own sales, transactions, status and anniversary fields are excluded by an explicit allow-list. Peer-sales features use only nearby stores that opened before the focal location, measured over the 12 months before its opening.

**Caveat.** The benchmark was inspected during model selection, so its WAPE is optimistic. An unbiased estimate needs openings after the selection date.

## Feature engineering

Features come from four sources: location demographics (1-mile trade area), the boundary/peer table, hexbin access and competition, and site attributes. The model derives everything else from them (`advanced_sales_models.engineered_features`).

| Family | Construction | Rationale |
|---|---|---|
| Peer sales | 25th/50th/75th percentile and closest-store monthly sales of prior nearby stores; dispersion `(p75 − p25) / mean`; closest/mean ratio; growth relative to level | Local demand level and its uncertainty |
| Cannibalization | `peer_mean / (stores within r + 1)` and the closest-store equivalent, for r ∈ {5, 10, …, 30} miles; cumulative ring counts | Sales per existing unit, a proxy for saturation |
| Demographics | Worker share, daytime/resident ratio, persons per household, disposable/median income, home value/income, spend per household, budget per person, competitors per 1,000 residents | Ratios are scale-free across trade areas of different size |
| Access | Traffic peak ratio `max / median` in 3 rings; inner/outer traffic ratio; traffic per resident; highway and ramp proximity | Visibility and drive-by volume |
| Site attributes | 12 physical fields plus missingness flags; site-to-building ratio, occupancy per 1,000 sq ft, order points per lane | Throughput capacity |
| Time | Year, `sin`/`cos` of month and day of year, opening weekday | Trend and seasonality without a discontinuity at December |
| Name flags | Regex flags for no drive-thru, in-line, airport, campus, mall, highway | Format signals not coded elsewhere |

**Numerical safeguards.**
- **Ratios:** return missing when the denominator satisfies |b| ≤ 1, which prevents unbounded values.
- **Access sentinel:** 100,000 means "no access found". It is recoded to missing and given its own indicator rather than treated as a distance.
- **Infinities:** ±∞ values are set to missing.
- **Missing values:** LightGBM learns a default split direction for them, so no imputation is used.
- **Categoricals:** ordinal codes, with values unseen in training mapped to −1. LightGBM handles them natively (minimum 10 rows per category, smoothing 20, L2 penalty 10). Integer and float codes are normalized to the same text, so `1` and `1.0` cannot become different categories after a Spark round trip.

## Target transformations and loss

The two production components are fit differently, by design.

| | `v2_015_lgb_rich` (attribute model) | `v2_128_lgb_static` (fallback) |
|---|---|---|
| Scored rows | FSU with ≥ 70% of 7 core attributes present | DTO, MALL, and FSU with incomplete attributes |
| Target | Peer ratio `y / a`, with `a = max(peer_mean_monthly / 26, 100)` | Raw `y / 10,000` |
| Loss | Squared error | Pinball (quantile) loss, τ = 0.25 |
| Weights | Uniform | Inverse-level `median(y) / y` × tail weights (×2 below / ×3 above the subtype 20th/80th percentile) × half-life 730 days × subtype balancing; normalized to mean 1, clipped to [0.05, 20] |
| Capacity | 400 trees, learning rate 0.035, depth 5, 15 leaves, λ₂ = 10 | 800 trees, learning rate 0.07, depth 7, 15 leaves, λ₂ = 20 |

The peer-ratio target removes the dominant local level effect. The model learns a multiplicative deviation from nearby stores, which is closer to stationary across markets and years.

The fallback's τ = 0.25 loss and tail weights deliberately pull its forecasts down. It is used where information is weakest, and that is where the asymmetric cost of over-forecasting matters most.

## Feature selection

Selection was hierarchical and always chronological. Nothing was selected using the rows it was evaluated on.

1. **Group level.** A search over 336 base configurations and 34 ensembles compared feature groups (original, rich, peers, market, demographics, access, static, attributes and others) on six discovery folds. The final pair was then checked on the confirmation and benchmark cohorts.
2. **Within each fold.**
   - Columns with zero variance in the training fold are dropped.
   - Of any numeric pair with |Pearson ρ| > 0.995, the later column is dropped.
   - Rich model: 152 → 141 features (6 constant, 5 near-duplicate). Static model: 93 → 91.
3. **Model level.** Supervised top-k filters (LightGBM gain, Spearman |ρ|) were searched but not selected. The production models rely on regularized tree splitting: 122 of the 141 rich features receive non-zero gain.

**Gain by family (rich model).**

| Family | Features | Share of split gain |
|---|---:|---:|
| Peer sales | 30 | 43.0% |
| Access / competition | 45 | 26.1% |
| Site attributes | 21 | 11.7% |
| Demographics | 21 | 11.0% |
| Geography | 6 | 4.0% |
| Time | 7 | 3.8% |
| Core categorical | 5 | 0.5% |
| Name flags | 6 | 0.0% |

The top 10 features hold 40.8% of gain, led by the peer 25th-percentile monthly sales (13.7%), `is_relocated` (4.8%) and closest-store sales per unit within 5 miles (4.5%). Gain measures how much the model uses a feature, not its causal effect. Correlated features share credit.

**Attribute ablation.**

| Specification | Confirmation WAPE | Benchmark WAPE |
|---|---:|---:|
| Rich, no site attributes | 17.37% | 18.88% |
| Rich, with site attributes | 17.15% | 17.84% |
| Rich, attributes blanked at inference | — | 20.24% |
| **Router (rich if ≥ 70% core attributes, else static)** | **15.60%** | **17.66%** |

Blanking the attributes at inference costs 2.4 pp. That is why rows with sparse attributes are routed to a model that never relied on them.

## Planning range: residual analysis

Let `r = actual / forecast − 1` over the 877 walk-forward forecasts.

| Statistic | Value |
|---|---:|
| Median | −2.1% |
| Mean | +3.0% |
| Standard deviation | 31.7% |
| Skewness | 2.18 |
| 10th / 25th percentile | −27.1% / −17.0% |
| 75th / 90th percentile | +15.4% / +36.9% |
| Shapiro–Wilk p | < 10⁻²⁷ |

The residuals are strongly right-skewed and non-normal. A symmetric ±k·σ interval would be miscalibrated on both sides, so each side is estimated separately from empirical quantiles of the signed errors:

- **Lower side:** the q-th quantile of −r among rows where actual fell below the forecast (r < 0).
- **Upper side:** the q-th quantile of r among rows where actual exceeded the forecast (r > 0).

Each side is clipped to [15%, cap]. The calibration group is subtype × region when it has ≥ 12 rows, otherwise subtype, otherwise all rows.

## Planning range: impact of the dynamic stretch

Walk-forward evaluation: for each origin from 2023-01 (6 origins, n = 605), the stretch is calibrated only on forecasts from earlier origins. Intervals are Wilson 95% CIs. Paired comparisons use 5,000 bootstrap resamples of locations and the exact McNemar test.

| Rule | Coverage [95% CI] | Mean total width |
|---|---:|---:|
| Fixed ±15% | 46.0% [42.0, 49.9] | 30.0% |
| Fixed ±18.5% (width of previous rule) | 55.5% [51.6, 59.4] | 36.9% |
| **Previous: dynamic, q = 0.60, cap 20%** | 55.0% [51.1, 59.0] | 36.9% |
| Fixed ±20% | 58.5% [54.5, 62.4] | 40.0% |
| Fixed ±22.5% (width of production rule) | 64.5% [60.6, 68.2] | 44.9% |
| **Production: dynamic, q = 0.70, cap 25%** | **65.3% [61.4, 69.0]** | 44.9% |
| Fixed ±25% | 71.4% [67.7, 74.9] | 50.0% |

| Comparison | Δ coverage [95% CI] | Discordant pairs | McNemar p |
|---|---:|---:|---:|
| Production vs previous | +10.2 pp [+7.9, +12.7] | 62 / 0 | 4 × 10⁻¹⁹ |
| Previous dynamic vs fixed ±15% | +9.1 pp [+6.8, +11.4] | 55 / 0 | 6 × 10⁻¹⁷ |
| Previous dynamic vs width-matched fixed ±18.5% | −0.5 pp [−2.0, +1.2] | 10 / 13 | 0.68 |
| Production dynamic vs width-matched fixed ±22.5% | +0.8 pp [−1.0, +2.6] | 18 / 13 | 0.47 |
| Tier-specific stretch vs production | −2.0 pp [−4.1, +0.2] | — | 0.10 |

**Interpretation.**

- **Width explains the coverage gains.** Each comparison against a narrower rule is strongly significant, but the discordant pairs all run one way: 62/0 and 55/0. In other words, every location the wider rule gained, the narrower rule had missed, and no location was lost. That is the signature of a strictly wider interval. Across fixed widths from ±10% to ±30%, coverage rises by about 2.4 pp per +1 pp of half-width.
- **The dynamic allocation adds no measurable capture.** Against a fixed symmetric range of the same mean width, the subtype × region stretch differs by −0.5 pp and +0.8 pp, neither significant. With n = 605, the test can detect differences of roughly ±2 pp, so any real benefit is smaller than that.
- **Tier-specific ranges were rejected.** Low/mid/high tiers show the expected regression toward the mean: low forecasts are often beaten and high forecasts often missed. But tier-specific sides calibrated on earlier origins did not generalize (−2.0 pp, p = 0.10). Optimizing the lower/upper split per tier at a fixed width budget also did not beat a single overall split. Tier is therefore a reporting label only.
- **Why keep the dynamic rule anyway:** coverage is not worse, and the sides adapt automatically as residuals drift by subtype and region. It is a defensible default, not a source of accuracy.

**Production choice.** Quantile 0.70 with a 25% cap trades +8.0 pp of mean width for +10.2 pp of coverage. Benchmark coverage with in-sample calibration is 65.7%. The weakest segments are the West region (53.5%) and the low tier (56.1%). The range is an empirical planning interval, not a confidence interval with nominal coverage.

## Limitations

- The benchmark (n = 210) was reused during selection, so point accuracy is optimistic.
- All figures use the historical modeling table (`90_days_features_with_attributes.parquet`). The production pipeline now requires all 90 business days before a label exists. Results should be re-confirmed on the first feature run built that way.
- The walk-forward folds use the final configuration, which was selected with later data. Fold-level errors are therefore slightly optimistic, which biases calibrated range widths narrow.
- Sparse segments are descriptive only: MALL n = 4, inline n = 4 and SSAM n = 11 benchmark locations.
- Calibration assumes the residual distribution is stable over time. Monitoring tracks realized coverage by tier, region and subtype once 90-day labels mature.
