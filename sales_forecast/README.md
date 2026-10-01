# Annual new-store sales forecast (FSU / DTO)

Forecasts first-year annual sales for new FSU and DTO stores. A gradient-boosted ensemble produces
a base forecast. A residual corrector then learns, from earlier cohorts' out-of-fold errors, the
biases that repeat year after year, and removes them.

| Walk-forward backtest, cohorts 2020-2026 | Avg WAPE | Avg MAPE | 2026 WAPE |
|---|---|---|---|
| Base ensemble | 18.1% | 19.3% | 19.5% |
| Base + residual correction (production) | **17.6%** | **18.1%** | **18.8%** |

Each cohort is scored by a model and a corrector that only saw earlier cohorts.

## Files

| Path | Purpose |
|---|---|
| `sales-forecast-walkforward.ipynb` | Main notebook: transform, feature selection, backtest, residual analysis, train, save, inference |
| `execute_sales_forecast_notebook.py` | Runs the notebook without Jupyter and saves outputs into it |
| `sales_forecast/constants.py` | Every tunable value: paths, as-of date, feature lists, model parameters, correction settings |
| `sales_forecast/transform.py` | Loading, cohort definition, feature engineering, relocation/predecessor matching |
| `sales_forecast/features.py` | Feature selection and cleaning; time-safe outcome features (trend, spatial, target encodings) |
| `sales_forecast/model.py` | Base ensemble, walk-forward backtest, residual corrector, `SalesForecaster` (fit / predict / save / load) |
| `sales_forecast/residuals.py` | Residual diagnostics: cohort bias, group lag, spatial lag, feature-bin persistence |
| `sales_forecast/metrics.py` | WAPE, MAPE, median APE, bias |
| `sales_forecast_results/` | Outputs: OOF predictions, backtest metrics, saved model, inference forecasts |

## How to run

```bash
python execute_sales_forecast_notebook.py
```

Or open the notebook and run all cells. Python packages are read from `.annual_packages`. Set
`SALES_PACKAGE_DIR` to a local copy of that folder for much faster imports than from OneDrive. The full
run takes about 5 minutes. Set `RUN_BACKTEST = False` in section 3 to reuse cached OOF predictions.

To score new stores with the saved model:

```python
from sales_forecast.model import SalesForecaster
forecaster = SalesForecaster.load()
forecast = forecaster.predict(frame)  # frame built with transform.engineer_features
```

## Data and definitions

- **Inputs:** `sales_features.csv` (location, demographic, traffic, competition, nearby-store sales)
  and `loc_attributes.parquet` (store attributes, relocation flag).
- **Scope:** FSU and DTO only. MALL is excluded.
- **Target:** `12 × avg_monthly_sales`, in $M, modelled in log space.
- **Cohort:** calendar year of the first anniversary, `open_date + 1 year`.
- **Training stores:** existing, opened more than one year before `AS_OF_DATE`, positive sales (1,238 stores, cohorts 2016-2026).
- **Inference stores:** pipeline stores, plus existing stores opened within the last year (721 stores).
  Older existing stores without a sales record are neither trained on nor forecast.

## Method

### 1. Features
- Log transforms of population, income, spend, traffic, and nearby-store sales. Ratios: worker/resident, daytime/resident, closest vs mean neighbour sales.
- Composite scores: cannibalization (`3·boundary_5 + 2·boundary_10 + boundary_15`) and recent nearby openings.
- **Relocation:** `is_relocated` from the attributes table, plus a predecessor match. The predecessor is a
  closed "(OLD)" location with the same normalized name; only its age is used, since its sales are not available.
- **Not used:** building size, drive-thru lanes, occupancy, and bypass lane. They are mostly blank for pipeline
  stores (5% filled for building size), and 525 of 552 pipeline stores show 0 lanes, which almost
  certainly means "not entered yet".
- **Missing geography:** region is filled from the same DMA, then state (about 13% of pipeline stores lack it).
  Missing DMA or submarket fall back to neutral target encodings.
- **Price level:** there is no price column. Nearby-store sales absorb most geographic price and level differences.
  Region, state, and DMA errors do not repeat from year to year (see residual analysis).

### 2. Time-safe outcome features
These features use other stores' sales, so they are built with strict time ordering:
- **Robust trend:** a Huber regression of log sales on opening date. It is never extrapolated more than
  `MAX_TREND_EXTRAPOLATION_YEARS` past the latest training store.
- **Spatial smoothing:** Gaussian-kernel averages of detrended sales of nearby stores, at 25, 75, and 200 km bandwidths.
- **Shrunk target encodings:** state, DMA, submarket, and region averages of detrended sales.

For a training row, these use only stores from earlier cohorts. For a scored row, they use all training stores.

### 3. Base ensemble
The base forecast is the geometric mean of three models:
- LightGBM on log sales
- LightGBM (Huber loss) on log sales minus trend
- CatBoost (depth 6) on log sales minus trend

### 4. Walk-forward backtest
Cohort Y is predicted by a model trained on cohorts < Y, for every Y from 2018 onward. These out-of-fold
residuals are the only residuals used for learning corrections. Training-set residuals are overfit and near zero.

### 5. Residual analysis: what repeats from one cohort to the next

| Pattern in out-of-fold error | Repeats? | Effect |
|---|---|---|
| Relocated stores | Yes, lag correlation 0.7-0.85, same sign every year | Under-forecast about 12% |
| DTO stores | Yes | Over-forecast about 9% |
| SSAM stores | Yes | Over-forecast about 6% |
| Oct-Dec openings | Yes | Over-forecast about 6% |
| Overall cohort level | No (lag-1 autocorrelation -0.2) | Swings between -10% and +5% |
| Region / state / DMA | No (lag correlations about 0 or negative) | Correcting adds noise |
| Nearby earlier stores (spatial lag) | Weak, only within 10 km at lag 1 | No accuracy gain |

### 6. Residual corrector (`ResidualCorrector`)
1. **Factor offsets:** ridge regression (`RESIDUAL_RIDGE_ALPHA`) of the cohort-centred out-of-fold residual on
   `RESIDUAL_FACTORS` (relocated, SSAM, DTO, opening quarter). The final fitted offsets in log terms:
   relocated +10%, DTO -7%, SSAM -4%, Q4 opening -5%.
2. **Level anchor:** moves each cohort's median forecast `LEVEL_ANCHOR_STRENGTH` (0.5) of the way toward the latest
   observed cohort's actual median, plus `LEVEL_ANNUAL_GROWTH` per year (default 0). Inference cohorts with fewer
   than 20 stores use the store-weighted shift of the larger cohorts.

The production model fits the base ensemble on all training stores and the corrector on all out-of-fold residuals (2018-2026).

## Output columns (`sales_forecast_results/inference_forecasts.csv`)

| Column | Meaning |
|---|---|
| `forecast_base_annual_sales` | Base ensemble forecast, $/year |
| `forecast_annual_sales` | Corrected forecast, $/year (use this) |
| `forecast_avg_monthly_sales` | `forecast_annual_sales / 12` |
| `cohort` | First-anniversary year |
| `open_date_placeholder` | True if the open date is more than `PLACEHOLDER_OPEN_YEARS` past the as-of date. Check before use. |

## Known limitations
- Store-to-store variation within a market dominates the remaining error. Even exact hindsight DMA offsets would only reach about 10% WAPE, and that is memorization rather than skill.
- The level anchor assumes the next cohort's median resembles the latest one. If pricing or the market mix changes, set `LEVEL_ANNUAL_GROWTH`.
- Predecessor sales for relocations, dated nearby competitor openings, a price index, and pre-opening store attributes would be the most valuable additions.
