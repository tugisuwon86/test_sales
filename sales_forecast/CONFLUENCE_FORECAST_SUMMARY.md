# First-90-Day Average Sales Forecast — Current Approach and Results

## Executive summary

- **Target:** `avg_sales`, average daily sales during the first 90 days after opening.
- **Official point forecast:** unmodified LightGBM base model (`v2_128_lgb_static`).
- **Current benchmark:** 176 openings from July 2025 onward. The benchmark has been reused during iteration, so results are directional rather than a fresh validation claim.
- **Best aggregate result:** 18.01% WAPE, 17.84% MAPE, and $4,159 MAE.
- **Tail scenario:** the tail-only hybrid lowers lower-tail WAPE from 32.70% to 27.77% and upper-tail WAPE from 26.41% to 23.24%, but aggregate WAPE rises to 18.75%.
- **Range recommendation:** centered ±15% around the hybrid planning center; 49.4% observed benchmark coverage. This is a planning range, not a 78/95/98% confidence interval.

## Data and validation approach

1. Sorted locations by opening date.
2. Required a 90-day maturity gap between training openings and each validation origin so first-90-day labels would be available.
3. Used six chronological discovery periods during 2022–2024 for model and post-processing selection.
4. Used January–April 2025 as a confirmation cohort.
5. Used openings from July 2025 onward as the final reused benchmark.
6. Excluded the forecasted location’s own `avg_monthly_sales`, sales totals, transaction counts, IDs, status, and other fields that can overlap the target window.
7. Applied the requested 5,000–70,000 target filter to every partition. It removed zero rows because observed targets range from 5,874.59 to 61,552.36.

## Model selection

We evaluated 450 base configurations and 34 ensemble/calibration configurations. Families included LightGBM, CatBoost, XGBoost, regularized linear models, SVR, tree ensembles, geographic models, historical-summary models, neural regressors, and tail-gated mixtures. We also tested raw, log, and neighbor-offset targets; multiple loss functions; recency, subtype, inverse-sales and tail weighting; training-only feature selection; and several feature subsets.

The official base is a regularized LightGBM quantile model using the 25th-percentile loss, 800 trees, maximum depth 7, 15 leaves, recent-history weighting, subtype balancing, and additional weight on training tails. It was retained because it has the lowest aggregate benchmark WAPE/MAPE among the final candidates. Its filename contains `exploratory_best` because the benchmark was inspected during selection; a later opening cohort is required for unbiased validation.

| Forecast | Band | n | WAPE | MAPE | MAE |
|---|---|---:|---:|---:|---:|
| base | ALL | 176 | 18.01% | 17.84% | $4,159 |
| base | low | 18 | 32.70% | 33.78% | $4,617 |
| base | middle | 120 | 12.58% | 12.98% | $2,709 |
| base | high | 38 | 26.41% | 25.63% | $8,520 |
| tail_only_hybrid | ALL | 176 | 18.75% | 18.76% | $4,329 |
| tail_only_hybrid | low | 18 | 27.77% | 27.82% | $3,921 |
| tail_only_hybrid | middle | 120 | 15.73% | 16.12% | $3,387 |
| tail_only_hybrid | high | 38 | 23.24% | 22.78% | $7,497 |
| region_stretch | ALL | 176 | 18.30% | 18.29% | $4,225 |
| region_stretch | low | 18 | 32.70% | 33.78% | $4,617 |
| region_stretch | middle | 120 | 13.45% | 13.84% | $2,896 |
| region_stretch | high | 38 | 25.53% | 25.01% | $8,235 |

## Tail-stretch analysis

The model compresses extremes toward the middle. We tested output-only corrections that push low predictions lower and high predictions higher. All correction parameters and tail cutoffs were learned from earlier mature chronological predictions; actual tail membership is never available at inference.

### Subtype and region

- Subtype-specific, region-specific, and subtype-plus-region stretch factors were tested across 48 configurations per base.
- Sparse estimates shrink toward a pooled factor using `n / (n + k)`.
- Chronological discovery selected **region-only** stretching for every base. Subtype-region intersections were too sparse to support stable corrections.
- For the official base, region stretching improves upper-tail WAPE from 26.41% to 25.53%, leaves lower-tail WAPE at 32.70%, and worsens aggregate WAPE from 18.01% to 18.30%. It is therefore not the default.
- The tail-only hybrid produces the best two-sided tail tradeoff (27.77% lower, 23.24% upper), but middle-band WAPE rises from 12.58% to 15.73%. This middle regression is why aggregate WAPE worsens.

| Subtype | Train/Test n | Base train WAPE | Base test WAPE | Hybrid train WAPE | Hybrid test WAPE |
|---|---:|---:|---:|---:|---:|
| ALL | 1085/176 | 13.54% | 18.01% | 18.97% | 18.75% |
| FSU | 976/155 | 13.08% | 17.96% | 18.08% | 18.49% |
| DTO | 54/17 | 12.58% | 20.42% | 18.48% | 22.44% |
| MALL | 55/4 | 23.40% | 10.69% | 36.53% | 16.49% |

| Region | Train/Test n | Base train WAPE | Base test WAPE | Hybrid train WAPE | Hybrid test WAPE |
|---|---:|---:|---:|---:|---:|
| Atlantic Region | 134/22 | 12.96% | 15.72% | 17.86% | 16.16% |
| Midwest Region | 171/43 | 9.66% | 16.31% | 16.61% | 17.08% |
| Northeast Region | 181/26 | 16.31% | 17.15% | 21.98% | 17.40% |
| Southeast Region | 187/28 | 13.07% | 14.57% | 16.97% | 17.03% |
| Southwest Region | 225/22 | 11.90% | 24.86% | 19.64% | 25.48% |
| West Region | 187/35 | 16.43% | 20.52% | 19.92% | 20.55% |

## Range forecast

The range scales with the sales baseline rather than using a fixed dollar width. The recommended output contains the official base point, hybrid planning center, and a centered ±15% range around the planning center. Examples: 20,000 → 17,000–23,000; 60,000 → 51,000–69,000.

| Desired benchmark coverage | Required symmetric percentage per side | Observed coverage |
|---:|---:|---:|
| Practical planning range | ±15.0% | 49.4% |
| Approximately 78% | ±29.9% | 78.4% |
| Approximately 95% | ±57.5% | 95.5% |
| Approximately 98% | ±65.6% | 98.3% |

The 78/95/98 widths are descriptive quantiles of the reused benchmark, not future coverage guarantees. The 95% and 98% ranges are too wide for the current business requirement.

## Feature importance

Importance is measured on the saved official model. `Gain %` measures LightGBM training improvement attributed to a feature. `Mean |contribution|` uses LightGBM tree contributions on the benchmark, restored to approximate daily-sales dollars. Importance describes model use, not causal impact. Correlated features can share or substitute importance.

| Rank | Feature | Explanation | Gain % | Mean absolute contribution ($/day) |
|---:|---|---|---:|---:|
| 1 | `year` | Opening year | 3.7% | 3,242 |
| 2 | `boundary_5` | Nearest-ring boundary count (confirm source-system definition) | 2.5% | 1,067 |
| 3 | `peer_income_interaction` | Neighbor-sales × household-income interaction | 2.3% | 696 |
| 4 | `latitude` | Latitude | 1.9% | 578 |
| 5 | `nearby_ring_total_10` | Context or engineered opening-time predictor | 1.8% | 387 |
| 6 | `bigbox_2` | Context or engineered opening-time predictor | 1.0% | 379 |
| 7 | `subtype` | Store subtype | 15.4% | 374 |
| 8 | `traffic_nearest_vs_outer` | Context or engineered opening-time predictor | 2.2% | 356 |
| 9 | `longitude` | Longitude | 2.0% | 340 |
| 10 | `region` | Context or engineered opening-time predictor | 0.6% | 331 |

### Most significant features

1. **Opening year (`year`):** captures the strong secular change in nominal opening sales across 2017–2026, including price level, brand maturity, rollout strategy, and market regime. Its large contribution also means temporal extrapolation is a material risk; a regional menu-price/inflation feature could separate price from demand.
2. **Nearest-ring boundary count (`boundary_5`):** represents the supplied immediate local-area count/density signal and is the strongest local-context contribution. Confirm its source-system definition before business interpretation; depending on its lineage it may proxy nearby development, store density, or cannibalization.
3. **Neighbor-sales × household-income (`peer_income_interaction`):** combines `avg_monthly_sales_mean_1year` for nearby stores with median household income. It captures locations where both comparable-store demand and spending capacity are strong. It is useful only if the neighbor-sales snapshot was available before the forecasted opening.

`subtype` has the largest training gain (15.4%) even though its average standalone benchmark contribution ranks below the top three. It is repeatedly useful for tree splits and interactions, while opening year and local context produce larger average prediction movements.

### Event features

`num_events` and `total_predicthq_attendance` are **not used by the official model** and were not included in the 450 production candidate configurations, so they have zero production-model importance. They remain available only in exploratory feature engineering because event exposure may be useful for airport, university, tourism, and entertainment locations. Current diagnostics are:

| Split | Event feature | Nonzero | Missing | Spearman correlation with avg_sales |
|---|---|---:|---:|---:|
| train | `num_events` | 4.8% | 0.0% | 0.130 |
| train | `total_predicthq_attendance` | 4.8% | 0.0% | 0.131 |
| test | `num_events` | 7.4% | 0.0% | -0.121 |
| test | `total_predicthq_attendance` | 7.4% | 0.0% | -0.127 |

Before including events, confirm that the schedule and expected attendance were known before opening, use an explicit `as_of_date`, and engineer exposure by distance, timing within the first 90 days, event type, and expected attendance. Raw totals can leak revised or realized attendance. Test event features as a chronological ablation rather than adding them directly to production.

## Recommended feature backlog

1. Physical format and capacity: drive-through lanes, square footage, seats, parking, access, visibility and co-tenancy.
2. Opening operating plan: hours, staffing, throughput, delivery/mobile availability, promotion, soft opening and temporary restrictions.
3. Opening-time demand: mobile visits, commuters, workplace density, tourism, university enrollment and airport passengers.
4. Competition/cannibalization: competitor traffic, same-brand drive-time rings, nearby-store trends, openings, closures and remodels.
5. Price normalization: regional menu price, promotions, inflation and local wage/income changes.
6. Opening execution: delays, roadworks, permitting, access restrictions and actual operating days.

## Production files

- `Sales_Forecast_Latest.ipynb`: latest executed analysis.
- `filtered_sales_results/final_recommendation.md`: final decision record.
- `generate_future_forecast.py`: future-location inference.
- `FUTURE_INFERENCE.md`: inference contract and example.
- `run_filtered_sales.py`: retraining/evaluation entry point.

## Limitations

The final benchmark has only 18 lower-tail, 38 upper-tail, 17 DTO, and 4 MALL locations. Benchmark reuse means all reported benchmark comparisons are exploratory. Training WAPE is fitted/in-sample. Historical neighbor-sales and event inputs require an opening-time provenance audit. Achieving 15% overall WAPE and the requested tail goals likely requires better opening-time features rather than additional output stretching.
