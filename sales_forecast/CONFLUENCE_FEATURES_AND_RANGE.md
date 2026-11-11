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

Features come from four sources: location demographics (1-mile trade area), the boundary/peer table, hexbin access and competition, and site attributes. The model derives everything else from them (`advanced_sales_models.engineered_features`). The appendix lists every feature with its source table.

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

## Transformations illustrated

### Data flow

```
Upstream sources                              Upstream notebook          Table
-------------------------------------------   ------------------------   -----------------------
rd_datamart.conf.location_attributes      ┐
conf.primary_market_area (1-mile)         ├─> location-attributes  ──>   location_features      ┐
sbx_gis.mom_location_pipeline_strategy    ┘                                                     │
sbx_gis.store_store_odcm_driving          ┐                                                     │
dx_prod.restfin_lh.daily_sales_extended   ├─> boundary-features    ──>   boundary_features      │
enterprise_sales_forecast_ensemble        ┘                                                     │
rd_datamart.competitors.competitors_all   ┐                                                     ├─> 00_preprocessing
sbx_gis.hexbin_traffic_stats              ├─> hexbin-level-feature ──>   hexbin_level_features  │     (inner joins on location_num)
sbx_gis.odcm_test_interstates             ┘                                                     │
rd_datamart.conf.location_attributes (site attributes, Redshift) ───────────────────────────────┤
dx_prod.restfin_lh.daily_sales_extended (label: first 90 business days) ────────────────────────┘
                                                                                                │
   preprocessed_inputs ──> 01_feature_engineering ──> features_by_run (typed contract, inline_flag, year)
                                                                                                │
   02_training, per fold:  engineered_features ──> drop constant / |ρ| > 0.995 ──> ordinal-encode categories ──> LightGBM
```

### Worked example: one benchmark FSU opening

The example is a typical benchmark opening: an FSU in the Atlantic Region that opened on 2025-11-06, with first-90-business-day average sales of $23,746. Every value below was computed by the production code.

| Transformation | Input values | Formula | Output |
|---|---|---|---:|
| Peer level ratio | closest 787,954; mean 791,060 | `closest / mean` | 0.996 |
| Peer dispersion | p25 741,301; p75 900,774; mean 791,060 | `(p75 − p25) / mean` | 0.202 |
| Cumulative ring | `boundary_5` = 0, `boundary_10` = 0 | `Σ boundary_k, k ≤ 10` | 0 |
| Cannibalization, guarded | mean 791,060; `boundary_5` = 0 | `mean / (0 + 1)`, but denominator ≤ 1 → missing | missing |
| Demographic share | workers 3,744; residents 1,405 | `workers / residents` | 2.66 |
| Competition density | competitors 11; residents 1,405 | `1000 · 11 / 1,405` | 7.83 |
| Traffic peak | ring-1 max 9,391; ring-1 median 4,707 | `max / median` | 2.00 |
| Traffic gradient | ring-1 median 4,707; ring-3 median 8,237 | `inner / outer` | 0.57 |
| Site ratio | site 130,680 sq ft; building 5,321 sq ft | `site / building` | 24.6 |
| Capacity | occupancy 117; building 5,321 sq ft | `1000 · 117 / 5,321` | 22.0 |
| Throughput | order points 2; lanes 2 | `2 / (2 + 1)` | 0.67 |
| Cyclical month | November (month 11) | `sin(2π·11/12)`, `cos(2π·11/12)` | −0.50, 0.87 |
| Cyclical day | 6 November (day 310) | `sin`/`cos(2π·310/365.25)` | −0.81, 0.58 |
| Name flag | "… FSU" | regex `in[\s\-]*line` | 0 |

The guarded ratio is deliberate: any denominator with |b| ≤ 1 returns missing. A side effect for the "per store" features is that a location with no stores in the ring gets a missing value rather than the full peer level. LightGBM then sends it down the branch it learned for missing values.

### Missing values and sentinels

| Raw value | Treatment | Model sees |
|---|---|---|
| `close_highway_dist` = 7.91 | Kept | 7.91; `_unavailable` = 0 |
| `close_highway_dist` = 100,000 (no access found) | Sentinel recoded | missing; `_unavailable` = 1 |
| `maximum_occupancy` missing | Kept missing, flagged | missing; `maximum_occupancy_missing` = 1 |
| Ratio with denominator 0 | Guarded | missing |
| ±∞ from any feature | Replaced | missing |

### Categorical encoding

| Raw value | Canonical text | Ordinal code |
|---|---|---:|
| `subtype` DTO / FSU / MALL | "DTO" / "FSU" / "MALL" | 0 / 1 / 2 |
| `ssam_flag` 1, 1.0 or True | "1" | same code for all three |
| `ssam_flag` missing | "missing" | its own code |
| Value not seen in training | — | −1 |

Codes are learned on each training fold. LightGBM splits on them as categories, not as ordered numbers.

### Target transformations

**Attribute model (peer ratio).** For the example opening:

| Step | Formula | Value |
|---|---|---:|
| Anchor | `a = max(peer_mean_monthly / 26, 100)` = 791,060 / 26 | 30,425 |
| Training target | `y / a` = 23,746 / 30,425 | 0.780 |
| Prediction | `ŷ = (predicted ratio) × a` | ratio × 30,425 |

The model learns that this store sells about 78% of what its nearby peers average per trading day. That relative deviation is comparable across markets of very different absolute size.

**Fallback model (raw scale, pinball loss with τ = 0.25).** The target is `y / 10,000`, here 2.37. The loss on residual `u = y − ŷ` is `max(τ·u, (τ − 1)·u)`:

| Case | Residual u | Loss |
|---|---:|---:|
| Over-forecast by $2,000 | −2,000 | 1,500 |
| Under-forecast by $2,000 | +2,000 | 500 |

Over-forecasting costs three times as much, so the fallback aims at the 25th percentile of the conditional distribution. This is a deliberately conservative point for rows with weak information.

### Sample weights (fallback model)

Each weight multiplies four factors, then is normalized to mean 1 and clipped to [0.05, 20]. The examples come from the benchmark-fold training set: 1,085 openings, median `y` = $20,401. FSU tails are below $16,033 and above $27,070.

| Example | `y` | Level `median / y` | Tail | Recency `2^(−age / 730)` | Subtype balance | Final weight |
|---|---:|---:|---:|---:|---:|---:|
| FSU, low tail | 15,727 | 1.30 | ×2 | 0.99 | 0.61 | 3.31 |
| FSU, middle | 19,137 | 1.07 | ×1 | 1.00 | 0.61 | 1.38 |
| FSU, high tail | 35,030 | 0.58 | ×3 | 0.99 | 0.61 | 2.23 |
| DTO, middle | 20,099 | 1.02 | ×1 | 1.00 | 2.59 | 5.55 |

Subtype balance is `sqrt(N / (K · n_subtype))`. It upweights each of the 54 DTO openings about 4× relative to each of the 976 FSU openings, so the fallback, which scores every DTO and MALL store, is not tuned only for FSU.

### Range and tier

The range scales with the point forecast. Here is an FSU Atlantic forecast of $20,000, using a calibration from forecasts before 2025-07 (subtype + region group, 70 rows):

| Quantity | Formula | Value |
|---|---|---:|
| Lower stretch | 70th percentile of `−r \| r < 0`, clipped to [15%, 25%] | 22.4% |
| Upper stretch | 70th percentile of `r \| r > 0`, clipped to [15%, 25%] | 23.9% |
| Range | `20,000 × (1 − 0.224)` to `20,000 × (1 + 0.239)` | $15,529 – $24,783 |
| Tier | FSU cutoffs $21,114 / $26,278 (tertiles of 335 recent FSU openings) | low |

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

## Appendix: feature inventory

All 152 candidate features, generated from the production code. ✓ = used by the model. "dropped" = in that model's feature group but removed by the within-fold filter in the final refit (constant, or |ρ| > 0.995 with an earlier column). – = not in that model's feature group.

Pipeline tables are written by the upstream notebooks to `source_schema`: `location_features` by location-attributes, `boundary_features` by boundary-features, `hexbin_level_features` by hexbin-level-feature. Site attributes are read directly from Redshift in `00_preprocessing`.

### Source columns (79)

| Feature | Pipeline table | Upstream source | Meaning | Attribute model | Fallback model |
|---|---|---|---|:---:|:---:|
| `total_population_1` | `location_features` | `conf.primary_market_area` (1-mile trade area) | Trade-area demographic | ✓ | ✓ |
| `worker_population_1` | `location_features` | `conf.primary_market_area` (1-mile trade area) | Trade-area demographic | ✓ | ✓ |
| `daytime_population_1` | `location_features` | `conf.primary_market_area` (1-mile trade area) | Trade-area demographic | ✓ | ✓ |
| `total_households_1` | `location_features` | `conf.primary_market_area` (1-mile trade area) | Trade-area demographic | ✓ | ✓ |
| `competitor_count_1` | `location_features` | `conf.primary_market_area` (1-mile trade area) | Trade-area demographic | ✓ | ✓ |
| `boundary_5` | `boundary_features` | `sbx_gis.store_store_odcm_driving` | Prior stores 0–5 driving miles away | ✓ | ✓ |
| `boundary_10` | `boundary_features` | `sbx_gis.store_store_odcm_driving` | Prior stores 5–10 driving miles away | ✓ | ✓ |
| `boundary_15` | `boundary_features` | `sbx_gis.store_store_odcm_driving` | Prior stores 10–15 driving miles away | ✓ | ✓ |
| `boundary_20` | `boundary_features` | `sbx_gis.store_store_odcm_driving` | Prior stores 15–20 driving miles away | ✓ | ✓ |
| `boundary_25` | `boundary_features` | `sbx_gis.store_store_odcm_driving` | Prior stores 20–25 driving miles away | ✓ | ✓ |
| `boundary_30` | `boundary_features` | `sbx_gis.store_store_odcm_driving` | Prior stores 25–30 driving miles away | ✓ | ✓ |
| `months_6m` | `boundary_features` | `sbx_gis.store_store_odcm_driving`, `location_features` | Nearby stores opened 0–6 months earlier | ✓ | ✓ |
| `months_12m` | `boundary_features` | `sbx_gis.store_store_odcm_driving`, `location_features` | Nearby stores opened 6–12 months earlier | ✓ | ✓ |
| `months_18m` | `boundary_features` | `sbx_gis.store_store_odcm_driving`, `location_features` | Nearby stores opened 12–18 months earlier | ✓ | ✓ |
| `months_24m` | `boundary_features` | `sbx_gis.store_store_odcm_driving`, `location_features` | Nearby stores opened 18–24 months earlier | ✓ | ✓ |
| `months_30m` | `boundary_features` | `sbx_gis.store_store_odcm_driving`, `location_features` | Nearby stores opened 24–30 months earlier | ✓ | ✓ |
| `months_36m` | `boundary_features` | `sbx_gis.store_store_odcm_driving`, `location_features` | Nearby stores opened 30–36 months earlier | ✓ | ✓ |
| `avg_population_growth_1` | `location_features` | `conf.primary_market_area` (1-mile trade area) | Trade-area demographic | ✓ | ✓ |
| `total_households_per_sq_mile_1` | `location_features` | `conf.primary_market_area` (1-mile trade area) | Trade-area demographic | dropped | dropped |
| `med_household_income_1` | `location_features` | `conf.primary_market_area` (1-mile trade area) | Trade-area demographic | ✓ | ✓ |
| `med_disposable_income_1` | `location_features` | `conf.primary_market_area` (1-mile trade area) | Trade-area demographic | ✓ | ✓ |
| `avg_disposable_income_1` | `location_features` | `conf.primary_market_area` (1-mile trade area) | Trade-area demographic | ✓ | ✓ |
| `annual_budget_expenditure_1` | `location_features` | `conf.primary_market_area` (1-mile trade area) | Trade-area demographic | ✓ | ✓ |
| `annual_budget_expenditure_index_1` | `location_features` | `conf.primary_market_area` (1-mile trade area) | Trade-area demographic | ✓ | ✓ |
| `med_home_value_1` | `location_features` | `conf.primary_market_area` (1-mile trade area) | Trade-area demographic | ✓ | ✓ |
| `avg_home_value_1` | `location_features` | `conf.primary_market_area` (1-mile trade area) | Trade-area demographic | ✓ | ✓ |
| `food_away_from_home_1` | `location_features` | `conf.primary_market_area` (1-mile trade area) | Trade-area demographic | dropped | – |
| `bigbox_1` | `hexbin_level_features` | `rd_datamart.competitors.competitors_all` | Big-box retailers in hex ring 1 | ✓ | ✓ |
| `bigbox_2` | `hexbin_level_features` | `rd_datamart.competitors.competitors_all` | Big-box retailers in hex ring 2 | ✓ | ✓ |
| `bigbox_3` | `hexbin_level_features` | `rd_datamart.competitors.competitors_all` | Big-box retailers in hex ring 3 | ✓ | ✓ |
| `competitor_1` | `hexbin_level_features` | `rd_datamart.competitors.competitors_all` | Restaurant competitors in hex ring 1 | ✓ | ✓ |
| `competitor_2` | `hexbin_level_features` | `rd_datamart.competitors.competitors_all` | Restaurant competitors in hex ring 2 | ✓ | ✓ |
| `competitor_3` | `hexbin_level_features` | `rd_datamart.competitors.competitors_all` | Restaurant competitors in hex ring 3 | ✓ | ✓ |
| `highway_5` | `hexbin_level_features` | `sbx_gis.odcm_test_interstates` | Interstate ramp proximity (distance or drive time) | ✓ | ✓ |
| `highway_10` | `hexbin_level_features` | `sbx_gis.odcm_test_interstates` | Interstate ramp proximity (distance or drive time) | ✓ | ✓ |
| `highway_5min` | `hexbin_level_features` | `sbx_gis.odcm_test_interstates` | Interstate ramp proximity (distance or drive time) | ✓ | ✓ |
| `highway_10min` | `hexbin_level_features` | `sbx_gis.odcm_test_interstates` | Interstate ramp proximity (distance or drive time) | ✓ | ✓ |
| `close_highway_dist` | `hexbin_level_features` | `sbx_gis.odcm_test_interstates` | Interstate ramp proximity (distance or drive time) | ✓ | ✓ |
| `avg_monthly_sales_closest_1year` | `boundary_features` | `dx_prod.restfin_lh.daily_sales_extended`, `rd_datamart.conf.enterprise_sales_forecast_ensemble` | Monthly sales, closest prior store, 12 months before opening | ✓ | – |
| `growth_rate_closest` | `boundary_features` | `dx_prod.restfin_lh.daily_sales_extended` | Linear monthly sales trend, closest prior store | ✓ | – |
| `avg_monthly_sales_mean_1year` | `boundary_features` | `dx_prod.restfin_lh.daily_sales_extended`, `rd_datamart.conf.enterprise_sales_forecast_ensemble` | Monthly sales, mean of prior stores, 12 months before opening | ✓ | – |
| `growth_rate_mean` | `boundary_features` | `dx_prod.restfin_lh.daily_sales_extended` | Linear monthly sales trend, mean of prior stores | ✓ | – |
| `avg_monthly_sales_25th_1year` | `boundary_features` | `dx_prod.restfin_lh.daily_sales_extended`, `rd_datamart.conf.enterprise_sales_forecast_ensemble` | Monthly sales, 25th percentile of prior stores, 12 months before opening | ✓ | – |
| `growth_rate_25th` | `boundary_features` | `dx_prod.restfin_lh.daily_sales_extended` | Linear monthly sales trend, 25th percentile of prior stores | ✓ | – |
| `avg_monthly_sales_75th_1year` | `boundary_features` | `dx_prod.restfin_lh.daily_sales_extended`, `rd_datamart.conf.enterprise_sales_forecast_ensemble` | Monthly sales, 75th percentile of prior stores, 12 months before opening | ✓ | – |
| `growth_rate_75th` | `boundary_features` | `dx_prod.restfin_lh.daily_sales_extended` | Linear monthly sales trend, 75th percentile of prior stores | ✓ | – |
| `median_average_daily_traffic_1` | `hexbin_level_features` | `sbx_gis.hexbin_traffic_stats` | Median daily traffic count in hex ring 1 | ✓ | ✓ |
| `max_average_daily_traffic_1` | `hexbin_level_features` | `sbx_gis.hexbin_traffic_stats` | Max daily traffic count in hex ring 1 | ✓ | ✓ |
| `median_average_daily_traffic_2` | `hexbin_level_features` | `sbx_gis.hexbin_traffic_stats` | Median daily traffic count in hex ring 2 | ✓ | ✓ |
| `max_average_daily_traffic_2` | `hexbin_level_features` | `sbx_gis.hexbin_traffic_stats` | Max daily traffic count in hex ring 2 | ✓ | ✓ |
| `median_average_daily_traffic_3` | `hexbin_level_features` | `sbx_gis.hexbin_traffic_stats` | Median daily traffic count in hex ring 3 | ✓ | ✓ |
| `max_average_daily_traffic_3` | `hexbin_level_features` | `sbx_gis.hexbin_traffic_stats` | Max daily traffic count in hex ring 3 | ✓ | ✓ |
| `subtype` | `location_features` | `rd_datamart.conf.location_attributes` | Location master record | ✓ | ✓ |
| `state` | `location_features` | `rd_datamart.conf.location_attributes` | Location master record | ✓ | ✓ |
| `region` | `location_features` | `rd_datamart.conf.location_attributes` | Location master record | ✓ | ✓ |
| `density_classification_1` | `location_features` | `conf.primary_market_area` (1-mile trade area) | Trade-area demographic | ✓ | ✓ |
| `ssam_flag` | `location_features` | `sbx_gis.mom_location_pipeline_strategy`, `tririga.design_construction_sites_all` | 1 if the site is SSAM | ✓ | ✓ |
| `no_dt_flag` | `location_features` | `sbx_gis.mom_location_pipeline_strategy` | Drive-thru indicator from the location name | ✓ | ✓ |
| `inline_flag` | `features_by_run` (derived in `01_feature_engineering`) | `location_name` in `location_features` | 1 if the name contains "In-Line" | ✓ | ✓ |
| `is_relocated` | Joined in `00_preprocessing` | `rd_datamart.conf.location_attributes` (Redshift, via `load_from_redshift`) | Physical site attribute | ✓ | – |
| `prototype` | Joined in `00_preprocessing` | `rd_datamart.conf.location_attributes` (Redshift, via `load_from_redshift`) | Physical site attribute | ✓ | – |
| `building_square_footage` | Joined in `00_preprocessing` | `rd_datamart.conf.location_attributes` (Redshift, via `load_from_redshift`) | Physical site attribute | ✓ | – |
| `total_site_square_footage` | Joined in `00_preprocessing` | `rd_datamart.conf.location_attributes` (Redshift, via `load_from_redshift`) | Physical site attribute | ✓ | – |
| `has_outside_seating` | Joined in `00_preprocessing` | `rd_datamart.conf.location_attributes` (Redshift, via `load_from_redshift`) | Physical site attribute | dropped | – |
| `maximum_occupancy` | Joined in `00_preprocessing` | `rd_datamart.conf.location_attributes` (Redshift, via `load_from_redshift`) | Physical site attribute | ✓ | – |
| `has_cross_parking` | Joined in `00_preprocessing` | `rd_datamart.conf.location_attributes` (Redshift, via `load_from_redshift`) | Physical site attribute | ✓ | – |
| `has_bypass_lane` | Joined in `00_preprocessing` | `rd_datamart.conf.location_attributes` (Redshift, via `load_from_redshift`) | Physical site attribute | ✓ | – |
| `has_drivethru_door` | Joined in `00_preprocessing` | `rd_datamart.conf.location_attributes` (Redshift, via `load_from_redshift`) | Physical site attribute | ✓ | – |
| `total_drivethru_lanes` | Joined in `00_preprocessing` | `rd_datamart.conf.location_attributes` (Redshift, via `load_from_redshift`) | Physical site attribute | ✓ | – |
| `total_order_points` | Joined in `00_preprocessing` | `rd_datamart.conf.location_attributes` (Redshift, via `load_from_redshift`) | Physical site attribute | ✓ | – |
| `playground` | Joined in `00_preprocessing` | `rd_datamart.conf.location_attributes` (Redshift, via `load_from_redshift`) | Physical site attribute | ✓ | – |
| `latitude` | `location_features` | `rd_datamart.conf.location_attributes` | Location master record | ✓ | ✓ |
| `longitude` | `location_features` | `rd_datamart.conf.location_attributes` | Location master record | ✓ | ✓ |
| `close_highway_min` | `hexbin_level_features` | `sbx_gis.odcm_test_interstates` | Interstate ramp proximity (distance or drive time) | ✓ | ✓ |
| `close_ramp_1` | `hexbin_level_features` | `sbx_gis.odcm_test_interstates` | Interstate ramp proximity (distance or drive time) | ✓ | ✓ |
| `close_ramp_2` | `hexbin_level_features` | `sbx_gis.odcm_test_interstates` | Interstate ramp proximity (distance or drive time) | ✓ | ✓ |
| `interstate_within_2min` | `hexbin_level_features` | `sbx_gis.odcm_test_interstates` | Interstate ramp proximity (distance or drive time) | ✓ | ✓ |
| `dma` | `location_features` | `rd_datamart.conf.location_attributes` | Location master record | ✓ | ✓ |
| `submarket` | `location_features` | `rd_datamart.conf.location_attributes` | Location master record | ✓ | ✓ |

### Derived features (73)

| Feature | Formula | Inputs from | Attribute model | Fallback model |
|---|---|---|:---:|:---:|
| `year` | Calendar year of `open_date` | `location_features` (`open_date`) | ✓ | ✓ |
| `month` | Calendar month of `open_date` | `location_features` (`open_date`) | ✓ | – |
| `month_sin` | `sin(2π · month / 12)` | `location_features` (`open_date`) | ✓ | ✓ |
| `month_cos` | `cos(2π · month / 12)` | `location_features` (`open_date`) | ✓ | ✓ |
| `day_of_year_sin` | `sin(2π · day_of_year / 365.25)` | `location_features` (`open_date`) | ✓ | ✓ |
| `day_of_year_cos` | `cos(2π · day_of_year / 365.25)` | `location_features` (`open_date`) | ✓ | ✓ |
| `opening_weekday` | Weekday of `open_date` (0 = Monday) | `location_features` (`open_date`) | ✓ | ✓ |
| `name_no_dt` | Case-insensitive regex on `location_name`: `no[\s\-]*d[tT] \| no[\s\-]*drive` | `location_features` (`location_name`) | ✓ | ✓ |
| `name_inline` | Case-insensitive regex on `location_name`: `in[\s\-]*line` | `location_features` (`location_name`) | ✓ | ✓ |
| `name_airport` | Case-insensitive regex on `location_name`: `airport \| terminal` | `location_features` (`location_name`) | ✓ | ✓ |
| `name_campus` | Case-insensitive regex on `location_name`: `university \| college \| campus` | `location_features` (`location_name`) | ✓ | ✓ |
| `name_mall` | Case-insensitive regex on `location_name`: `\bmall\b \| shopping center` | `location_features` (`location_name`) | ✓ | ✓ |
| `name_highway` | Case-insensitive regex on `location_name`: `\bhwy\b \| highway \| interstate` | `location_features` (`location_name`) | ✓ | ✓ |
| `peer_closest_vs_mean` | `closest / mean` peer monthly sales | `boundary_features` | ✓ | – |
| `peer_spread_relative` | `(p75 − p25) / mean` peer monthly sales | `boundary_features` | ✓ | – |
| `peer_lower_vs_mean` | `p25 / mean` | `boundary_features` | ✓ | – |
| `peer_upper_vs_mean` | `p75 / mean` | `boundary_features` | ✓ | – |
| `peer_mean_daily_26` | `peer mean monthly sales / 26` | `boundary_features` | dropped | – |
| `peer_closest_daily_26` | `closest monthly sales / 26` | `boundary_features` | dropped | – |
| `peer_growth_relative_closest` | `growth_rate_s / avg_monthly_sales_s` | `boundary_features` | ✓ | – |
| `peer_growth_relative_mean` | `growth_rate_s / avg_monthly_sales_s` | `boundary_features` | ✓ | – |
| `peer_growth_relative_25th` | `growth_rate_s / avg_monthly_sales_s` | `boundary_features` | ✓ | – |
| `peer_growth_relative_75th` | `growth_rate_s / avg_monthly_sales_s` | `boundary_features` | ✓ | – |
| `worker_share` | `worker_population / total_population` | `location_features` | ✓ | ✓ |
| `daytime_resident_ratio` | `daytime_population / total_population` | `location_features` | ✓ | ✓ |
| `people_per_household` | `total_population / total_households` | `location_features` | ✓ | ✓ |
| `disposable_income_share` | `med_disposable_income / med_household_income` | `location_features` | ✓ | ✓ |
| `home_income_ratio` | `med_home_value / med_household_income` | `location_features` | ✓ | ✓ |
| `spending_per_household` | `food_away_from_home / total_households` | `location_features` | ✓ | ✓ |
| `budget_per_person` | `annual_budget_expenditure / total_population` | `location_features` | ✓ | ✓ |
| `competitors_per_1000` | `1000 · competitor_count / total_population` | `location_features` | ✓ | ✓ |
| `nearby_ring_total_10` | Cumulative `Σ boundary_k` for k ≤ r | `boundary_features` | ✓ | ✓ |
| `nearby_ring_total_15` | Cumulative `Σ boundary_k` for k ≤ r | `boundary_features` | ✓ | ✓ |
| `nearby_ring_total_20` | Cumulative `Σ boundary_k` for k ≤ r | `boundary_features` | ✓ | ✓ |
| `nearby_ring_total_25` | Cumulative `Σ boundary_k` for k ≤ r | `boundary_features` | ✓ | ✓ |
| `nearby_ring_total_30` | Cumulative `Σ boundary_k` for k ≤ r | `boundary_features` | ✓ | ✓ |
| `traffic_peak_ratio_1` | `max / median` daily traffic in ring i | `hexbin_level_features` | ✓ | ✓ |
| `traffic_peak_ratio_2` | `max / median` daily traffic in ring i | `hexbin_level_features` | ✓ | ✓ |
| `traffic_peak_ratio_3` | `max / median` daily traffic in ring i | `hexbin_level_features` | ✓ | ✓ |
| `traffic_nearest_vs_outer` | `median_traffic_1 / median_traffic_3` | `hexbin_level_features` | ✓ | ✓ |
| `retail_competitor_ratio` | `(bigbox_1 + 1) / (competitor_1 + 1)` | `hexbin_level_features` | ✓ | ✓ |
| `traffic_per_population` | `median_traffic_1 / total_population` | `hexbin_level_features` | ✓ | ✓ |
| `peer_sales_per_population` | `peer mean / total_population` | `boundary_features`, `location_features` | ✓ | ✓ |
| `peer_income_interaction` | `peer mean · med_household_income / 10⁵` | `boundary_features`, `location_features` | ✓ | ✓ |
| `peer_sales_per_store_5` | `peer mean / (boundary_r + 1)` | `boundary_features` | ✓ | – |
| `closest_sales_per_store_5` | `closest / (boundary_r + 1)` | `boundary_features` | ✓ | – |
| `peer_sales_per_store_10` | `peer mean / (boundary_r + 1)` | `boundary_features` | ✓ | – |
| `closest_sales_per_store_10` | `closest / (boundary_r + 1)` | `boundary_features` | ✓ | – |
| `peer_sales_per_store_15` | `peer mean / (boundary_r + 1)` | `boundary_features` | ✓ | – |
| `closest_sales_per_store_15` | `closest / (boundary_r + 1)` | `boundary_features` | ✓ | – |
| `peer_sales_per_store_20` | `peer mean / (boundary_r + 1)` | `boundary_features` | ✓ | – |
| `closest_sales_per_store_20` | `closest / (boundary_r + 1)` | `boundary_features` | ✓ | – |
| `peer_sales_per_store_25` | `peer mean / (boundary_r + 1)` | `boundary_features` | ✓ | – |
| `closest_sales_per_store_25` | `closest / (boundary_r + 1)` | `boundary_features` | ✓ | – |
| `peer_sales_per_store_30` | `peer mean / (boundary_r + 1)` | `boundary_features` | ✓ | – |
| `closest_sales_per_store_30` | `closest / (boundary_r + 1)` | `boundary_features` | ✓ | – |
| `is_relocated_missing` | 1 if the site attribute is missing | `rd_datamart.conf.location_attributes` | dropped | – |
| `prototype_missing` | 1 if the site attribute is missing | `rd_datamart.conf.location_attributes` | ✓ | – |
| `building_square_footage_missing` | 1 if the site attribute is missing | `rd_datamart.conf.location_attributes` | ✓ | – |
| `total_site_square_footage_missing` | 1 if the site attribute is missing | `rd_datamart.conf.location_attributes` | dropped | – |
| `has_outside_seating_missing` | 1 if the site attribute is missing | `rd_datamart.conf.location_attributes` | dropped | – |
| `maximum_occupancy_missing` | 1 if the site attribute is missing | `rd_datamart.conf.location_attributes` | ✓ | – |
| `has_cross_parking_missing` | 1 if the site attribute is missing | `rd_datamart.conf.location_attributes` | ✓ | – |
| `has_bypass_lane_missing` | 1 if the site attribute is missing | `rd_datamart.conf.location_attributes` | ✓ | – |
| `has_drivethru_door_missing` | 1 if the site attribute is missing | `rd_datamart.conf.location_attributes` | ✓ | – |
| `total_drivethru_lanes_missing` | 1 if the site attribute is missing | `rd_datamart.conf.location_attributes` | dropped | – |
| `total_order_points_missing` | 1 if the site attribute is missing | `rd_datamart.conf.location_attributes` | dropped | – |
| `playground_missing` | 1 if the site attribute is missing | `rd_datamart.conf.location_attributes` | ✓ | – |
| `site_to_building_ratio` | `total_site_sqft / building_sqft` | `rd_datamart.conf.location_attributes` | ✓ | – |
| `occupancy_per_1000sqft` | `1000 · maximum_occupancy / building_sqft` | `rd_datamart.conf.location_attributes` | ✓ | – |
| `order_points_per_lane` | `total_order_points / (total_drivethru_lanes + 1)` | `rd_datamart.conf.location_attributes` | ✓ | – |
| `close_highway_dist_unavailable` | 1 if the access value is the 100,000 sentinel | `hexbin_level_features` | ✓ | ✓ |
| `close_highway_min_unavailable` | 1 if the access value is the 100,000 sentinel | `hexbin_level_features` | dropped | dropped |
