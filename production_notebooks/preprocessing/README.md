# 90-day sales forecast: Databricks notebooks

These are the `notebooks/` templates with the production_ready model work added. Run them in order as job tasks. Each task passes its run ID to the next one.

| Notebook | Status |
|---|---|
| `00_preprocessing.py` | Joins location, boundary, hexbin, and site-attribute sources; labels the first 90 business days |
| `01_feature_engineering.py` | Derives `inline_flag`, `year`, `month_of_opening`; casts to the training data's types |
| `02_training.py` | Held-out phases, range calibration, router refit, pyfunc registration |
| `03_validation.py` | WAPE and range coverage by segment, accept/reject decision |
| `04_deployment.py` | Unchanged; still stopped until ML engineering reviews promotion and rollback |
| `05_batch_inference.py` | Scores pending locations and adds the capped planning range |
| `06_monitoring.py` | Integrity alerts, forecast distribution, realized WAPE and coverage |

## Model code

Keep these files in the same workspace folder as the notebooks. `02_training.py` logs them with the model via `code_paths`.

- `sales_forecast_config.py`, `forecast_experiments.py`, `advanced_sales_models.py`, `attribute_router.py`: copied unchanged from `production_ready`.
- `sales_forecast_pipeline.py`: new. It holds the MLflow wrapper, the Spark input normalization, and the range calibration helpers.

The job environment needs `lightgbm`, `catboost`, and `scikit-learn`. `forecast_experiments.py` imports `catboost` at module load. Use the versions pinned in `production_ready/requirements-forecast.txt`.

## Sources and label

`00_preprocessing.py` reads the tables that the upstream notebooks write to `source_schema` (`location_features`, `boundary_features`, `hexbin_level_features`), plus `dx_prod.restfin_lh.daily_sales_extended`. It reads the site attributes from Redshift `rd_datamart.conf.location_attributes` through `load_from_redshift`.

`avg_sales` is the average daily net sales over the first 90 business days from `open_date`. A business day is a trading date with positive net sales. The label stays null until all 90 business days exist, so recent openings are scored rather than trained on. `total_sales`, `trans_count`, `total_trans`, `business_days`, and `label_end_date` are kept for review and are never model inputs.

## Feature contract

`features_by_run` is keyed by `feature_run_id` and `location_num`, with one row per location. `FEATURE_SCHEMA` in `01_feature_engineering.py` lists every column and its type. It matches `90_days_features_with_attributes.parquet`, except that the 365-day `avg_monthly_sales`, `avg_monthly_sales_log`, and `anniversary` columns and the unused event columns are left out. Labelled rows outside $5,000 to $70,000 are removed, as in training.

Training uses rows that have a label and opened at least 90 days before `as_of_date`. Batch inference scores the remaining rows.

## New tables

| Table | Written by |
|---|---|
| `range_calibration` | 02: side-specific stretch by subtype + region, subtype, and overall |
| `training_predictions` | 02: `confirmation` and `test` held-out rows |
| `batch_output` | 05: `forecast_point`, `forecast_lower`, `forecast_upper`, applied stretch, and calibration level |
