# 90-Day Sales Forecast — Final Model Performance

## Model selected

The production design is an attribute-aware subtype router with two LightGBM components. `v2_015_lgb_rich` is used for FSU locations with at least 70% coverage across seven core site attributes. `v2_128_lgb_static` is the fallback for DTO, MALL, and incomplete FSU records.

The design was selected after evaluating 336 base configurations and 34 chronological ensembles. The router achieved **15.60% confirmation WAPE** and **17.66% benchmark WAPE**. The benchmark is the current performance reference.

## Benchmark WAPE by subtype

| Subtype | Locations | WAPE |
|---|---:|---:|
| Overall | 210 | **17.66%** |
| DTO | 19 | 23.03% |
| FSU | 187 | 17.35% |
| MALL | 4 | 13.02% |

## Benchmark WAPE by SSAM and inline status

| Segment | Locations | WAPE |
|---|---:|---:|
| SSAM: No | 199 | 18.00% |
| SSAM: Yes | 11 | 12.36% |
| Inline: No | 206 | 17.73% |
| Inline: Yes | 4 | 13.02% |

The SSAM and inline cohorts are small. Their results are descriptive and should not be used alone to select a separate model.

## Range methodology

The production range does not use standard deviation. Each side begins at 15%, adjusts independently from historical signed percentage errors by subtype and region, and is capped at 20%. Groups with fewer than 12 observations fall back to subtype and then overall calibration. This keeps the interval proportional to the point forecast and prevents width from increasing without limit.

Fixed ±15% coverage is 49.5%. The capped subtype-and-region rule increases observed benchmark coverage to **56.7%**.

## Dynamic-range coverage by subtype

| Subtype | Locations | Coverage |
|---|---:|---:|
| Overall | 210 | **56.7%** |
| DTO | 19 | 63.2% |
| FSU | 187 | 55.6% |
| MALL | 4 | 75.0% |

## Fixed-range coverage by subtype

| Range | Overall | DTO | FSU | MALL |
|---|---:|---:|---:|---:|
| ±15% | 49.5% | 47.4% | 49.7% | 50.0% |
| ±20% | 61.9% | 63.2% | 61.5% | 75.0% |
| ±25% | 73.8% | 73.7% | 73.8% | 75.0% |

The fixed-range table shows the direct width-versus-coverage tradeoff. Coverage increases as the interval expands, so the 25% result should not be interpreted as a better point model. The final capped dynamic rule remains the production recommendation because it adjusts by subtype and region while limiting either side to 20%.

## Dynamic-range coverage by SSAM and inline status

| Segment | Locations | Coverage |
|---|---:|---:|
| SSAM: No | 199 | 55.8% |
| SSAM: Yes | 11 | 72.7% |
| Inline: No | 206 | 56.3% |
| Inline: Yes | 4 | 75.0% |

## Dynamic-range coverage by region

| Region | Locations | Coverage |
|---|---:|---:|
| Atlantic | 24 | 62.5% |
| Midwest | 44 | 56.8% |
| Northeast | 29 | 58.6% |
| Southeast | 39 | 64.1% |
| Southwest | 31 | 54.8% |
| West | 43 | 46.5% |

The reported ranges are empirically calibrated planning intervals. They are not standard-deviation confidence intervals. Coverage should always be reviewed with interval width and segment sample size.
