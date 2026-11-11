# Final model and range summary

## Model

The production model is an attribute-aware subtype router built from two LightGBM regressors. The attribute model (`v2_015_lgb_rich`) uses the rich feature set, peer-ratio transformation, squared-error objective, and L2 regularization of 10. It is applied to FSU rows with at least 70% completeness across seven core site attributes. The static fallback (`v2_128_lgb_static`) is applied to DTO, MALL, and incomplete FSU rows.

The design was selected from 336 base configurations and 34 chronological ensembles. The router achieved 15.60% confirmation WAPE and 17.66% benchmark WAPE. Benchmark subtype WAPE is 23.03% DTO, 17.35% FSU, and 13.02% MALL.

| Benchmark segment | Locations | WAPE |
|---|---:|---:|
| Overall | 210 | 17.66% |
| DTO | 19 | 23.03% |
| FSU | 187 | 17.35% |
| MALL | 4 | 13.02% |

### SSAM and inline performance

| Segment | Locations | WAPE |
|---|---:|---:|
| SSAM: No | 199 | 18.00% |
| SSAM: Yes | 11 | 12.36% |
| Inline: No | 206 | 17.73% |
| Inline: Yes | 4 | 13.02% |

SSAM and inline results should be read with their sample sizes. The inline cohort contains only four benchmark locations, and the SSAM cohort contains eleven.

## Range forecast

The final range is proportional to the point forecast and does not use standard deviation. Lower and upper stretches are estimated separately from historical signed percentage errors. Calibration first uses subtype + region when at least 12 observations are available, then falls back to subtype or overall. Each side has a 15% floor and 20% cap.

On the current 210-location benchmark, fixed ±15% covers 49.5% of actuals. The capped subtype-and-region rule covers 56.7%. Median applied stretch is 16.9% on the lower side and 20.0% on the upper side. This is an empirical planning range rather than a formal confidence interval.

| Segment | Dynamic-range coverage |
|---|---:|
| Overall | 56.7% |
| DTO | 63.2% |
| FSU | 55.6% |
| MALL | 75.0% |
| SSAM: No | 55.8% |
| SSAM: Yes | 72.7% |
| Inline: No | 56.3% |
| Inline: Yes | 75.0% |
| Atlantic Region | 62.5% |
| Midwest Region | 56.8% |
| Northeast Region | 58.6% |
| Southeast Region | 64.1% |
| Southwest Region | 54.8% |
| West Region | 46.5% |

Coverage should be interpreted together with interval width. The cap prevents coverage from being increased simply by producing commercially unusable ranges.

### Fixed proportional ranges by subtype

| Range | Overall | DTO | FSU | MALL |
|---|---:|---:|---:|---:|
| ±15% | 49.5% | 47.4% | 49.7% | 50.0% |
| ±20% | 61.9% | 63.2% | 61.5% | 75.0% |
| ±25% | 73.8% | 73.7% | 73.8% | 75.0% |

The final production range remains the capped subtype-and-region rule rather than fixed ±25%. The wider fixed interval has higher coverage because it is mechanically larger.
