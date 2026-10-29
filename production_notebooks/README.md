# 90-day sales forecast: Databricks notebooks

These are the `notebooks/` templates with the production_ready model work added. Run them in order as job tasks. Each task passes its run ID to the next one.

| Notebook | Status |
|---|---|
| `00_preprocessing.py` | Unchanged template (owned by Su) |
| `01_feature_engineering.py` | Unchanged template (owned by Su) |
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

## What `01_feature_engineering.py` must produce

`features_by_run` is keyed by `feature_run_id` and `location_num` (`ENTITY_KEYS = ["location_num"]`). It has one row per location and the same columns as `90_days_features_with_attributes.parquet`:

- `open_date` and `avg_sales`. `avg_sales` is the first-90-day label. Leave it null for locations that have not opened yet.
- Every column in `FEATURE_COLUMNS`. The location attribute columns, `location_name`, `latitude`, `longitude`, `dma`, `submarket`, and the access columns are optional but used when present.
- Store numeric inputs as numeric types, not strings.

Training uses rows that have a label and opened at least 90 days before `as_of_date`. Batch inference scores the remaining rows.

## New tables

| Table | Written by |
|---|---|
| `range_calibration` | 02: side-specific stretch by subtype + region, subtype, and overall |
| `training_predictions` | 02: `confirmation` and `test` held-out rows |
| `batch_output` | 05: `forecast_point`, `forecast_lower`, `forecast_upper`, applied stretch, and calibration level |
