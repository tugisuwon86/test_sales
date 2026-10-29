# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "4"
# dependencies = []
# ///
# MAGIC %md
# MAGIC # 90 days sales forecast: monitoring
# MAGIC Checks batch integrity and forecast distribution, and realized WAPE and coverage once actuals mature.
# MAGIC Validate dependency pins in the job example before enabling the lifecycle.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Installs

# COMMAND ----------

# Roost intentionally follows the latest release. Pin model/platform packages in the job environment.
# MAGIC %pip install --upgrade roost

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Imports

# COMMAND ----------

import mlflow
import pandas as pd
from pyspark.sql import functions as F
from roost.context import RoostContext
from roost.connector import RoostConnector
from roost.connector.core.types import WriteMode
from roost.connector.formats import PySparkFormatHandler
from roost.connector.systems.database import (
    DatabricksConnectionConfig,
    DatabricksSystem,
)
from sales_forecast_pipeline import (
    ENTITY_KEYS,
    TARGET,
    metric_name,
    range_coverage,
    segment_wape,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Context and run selection

# COMMAND ----------

ctx = RoostContext(team="fit-ds", defer_to_prod=True)
PROJECT = "sales_forecast_90_days"
SCHEMA = "sales_forecast_90_days"
READ_CATALOG = ctx.read_catalog
SCHEMA_FQN = f"{ctx.write_catalog}.{SCHEMA}"
RAW_SCHEMA_FQN = f"{ctx.write_catalog_raw}.{SCHEMA}"
MODEL_NAME = f"{RAW_SCHEMA_FQN}.sales_forecast_90_days_model"

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")
connector = RoostConnector(
    context=ctx,
    format_handler=PySparkFormatHandler(),
    system=DatabricksSystem(DatabricksConnectionConfig(managed_ownership=False)),
)

dbutils.widgets.text("inference_run_id", "")
INFERENCE_RUN_ID = dbutils.widgets.get("inference_run_id").strip()
if not INFERENCE_RUN_ID:
    raise ValueError("Provide a successful inference_run_id")
if mlflow.get_run(INFERENCE_RUN_ID).info.status != "FINISHED":
    raise ValueError("Choose a successful upstream run")

# Optional: a later feature run whose labels cover this batch's locations.
dbutils.widgets.text("actuals_feature_run_id", "")
ACTUALS_FEATURE_RUN_ID = dbutils.widgets.get("actuals_feature_run_id").strip()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Review the published batch
# MAGIC Measure this batch's coverage, prediction quality, and useful model metrics.
# MAGIC Compare with actuals only when labels are available. Read shared actuals
# MAGIC from the read catalog and project results from the write catalog.
# MAGIC Agree on alert limits and owners; informational changes need not fail a job.
# MAGIC
# MAGIC Integrity alerts (row-count mismatch, duplicate locations, missing or
# MAGIC non-positive forecasts, inverted ranges) fail the run after the metrics are
# MAGIC saved. Distribution, routing, and realized-accuracy metrics are informational.

# COMMAND ----------

with mlflow.start_run(tags={"project": PROJECT, "stage": "monitoring"}) as run:
    MONITORING_RUN_ID = run.info.run_id
    mlflow.log_param("inference_run_id", INFERENCE_RUN_ID)
    batch = spark.sql(
        """
        select *
        from identifier(:table)
        where inference_run_id = :run_id
        """,
        args={"table": f"{SCHEMA_FQN}.batch_output", "run_id": INFERENCE_RUN_ID},
    )

    # One row per scored location, so the batch fits in pandas.
    frame = batch.toPandas()
    point = frame.forecast_point
    expected_rows = mlflow.get_run(INFERENCE_RUN_ID).data.metrics.get("scoring_rows")
    metric_values = {
        "output_rows": len(frame),
        "expected_rows": float("nan") if expected_rows is None else expected_rows,
        "duplicate_locations": int(frame.duplicated(ENTITY_KEYS).sum()),
        "null_forecasts": int(point.isna().sum()),
        "nonpositive_forecasts": int(point.le(0).sum()),
        "invalid_ranges": int(
            (~(frame.forecast_lower.le(point) & point.le(frame.forecast_upper))).sum()
        ),
        "forecast_point_mean": point.mean(),
        "forecast_point_median": point.median(),
        "forecast_point_p05": point.quantile(0.05),
        "forecast_point_p95": point.quantile(0.95),
        "range_left_pct_median": frame.range_left_pct.median(),
        "range_right_pct_median": frame.range_right_pct.median(),
        "attribute_model_share": frame.used_attribute_model.astype(float).mean(),
    }
    for subtype, g in frame.groupby("subtype"):
        metric_values[metric_name("rows_subtype", subtype)] = len(g)
    for level, g in frame.groupby("range_calibration_level"):
        metric_values[metric_name("range_level_share", level)] = len(g) / len(frame)

    if ACTUALS_FEATURE_RUN_ID:
        # Only labels whose 90-day window has closed are compared.
        actuals = spark.sql(
            """
            select *
            from identifier(:table)
            where feature_run_id = :run_id
              and avg_sales is not null
              and open_date <= date_sub(current_date(), 90)
            """,
            args={
                "table": f"{SCHEMA_FQN}.features_by_run",
                "run_id": ACTUALS_FEATURE_RUN_ID,
            },
        ).select(*ENTITY_KEYS, F.col(TARGET).cast("double").alias(TARGET))
        realized = frame.merge(actuals.toPandas(), on=ENTITY_KEYS, how="inner")
        mlflow.log_param("actuals_feature_run_id", ACTUALS_FEATURE_RUN_ID)
        metric_values["actuals_matched_rows"] = len(realized)
        if len(realized):
            accuracy = segment_wape(
                realized.rename(columns={"forecast_point": "prediction"}), "actuals"
            )
            coverage = range_coverage(realized)
            mlflow.log_table(accuracy, "monitoring/realized_wape_by_segment.json")
            mlflow.log_table(coverage, "monitoring/realized_range_coverage.json")
            for r in accuracy.itertuples():
                metric_values[metric_name("actuals_wape", r.dimension, r.segment)] = r.WAPE
            for r in coverage.itertuples():
                metric_values[metric_name("actuals_range_coverage", r.dimension, r.segment)] = r.coverage_pct

    integrity_alerts = [
        name
        for name in [
            "duplicate_locations",
            "null_forecasts",
            "nonpositive_forecasts",
            "invalid_ranges",
        ]
        if metric_values[name] > 0
    ]
    if expected_rows is not None and len(frame) != expected_rows:
        integrity_alerts.append("row_count_mismatch")
    metric_values["integrity_alerts"] = len(integrity_alerts)
    metric_values = {k: float(v) for k, v in metric_values.items()}
    mlflow.log_metrics(metric_values)
    metrics_df = spark.createDataFrame(
        list(metric_values.items()), "metric_name string, metric_value double"
    )

    output = metrics_df.withColumn(
        "monitoring_run_id", F.lit(MONITORING_RUN_ID)
    ).withColumn("inference_run_id", F.lit(INFERENCE_RUN_ID))
    connector.write(
        output, destination=f"{SCHEMA_FQN}.monitoring_metrics", mode=WriteMode.APPEND
    )
    # Integrity failures stop before the completion marker; metrics stay saved.
    if integrity_alerts:
        mlflow.set_tag("monitoring_status", "alert")
        raise ValueError(f"Batch integrity alerts: {integrity_alerts}")
    mlflow.set_tag("monitoring_status", "passed")
    # Publish the completion marker last, after all monitoring results are saved.
    completed = spark.createDataFrame(
        [(MONITORING_RUN_ID, INFERENCE_RUN_ID, "completed")],
        "monitoring_run_id string, inference_run_id string, status string",
    )
    connector.write(
        completed, destination=f"{SCHEMA_FQN}.monitoring_runs", mode=WriteMode.APPEND
    )

# COMMAND ----------

dbutils.jobs.taskValues.set(key="monitoring_run_id", value=MONITORING_RUN_ID)
