# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "4"
# dependencies = []
# ///
# MAGIC %md
# MAGIC # 90 days sales forecast: monitoring
# MAGIC Template for Su. Fill the TODO section before running this stage.
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
from pyspark.sql import functions as F
from roost.context import RoostContext
from roost.connector import RoostConnector
from roost.connector.core.types import WriteMode
from roost.connector.formats import PySparkFormatHandler
from roost.connector.systems.database import (
    DatabricksConnectionConfig,
    DatabricksSystem,
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

# COMMAND ----------

# MAGIC %md
# MAGIC ### Review the published batch
# MAGIC Measure this batch's coverage, prediction quality, and useful model metrics.
# MAGIC Compare with actuals only when labels are available. Read shared actuals
# MAGIC from the read catalog and project results from the write catalog.
# MAGIC Agree on alert limits and owners; informational changes need not fail a job.

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

    # TODO: produce metrics_df (metric_name, metric_value) and explain any alerts.
    # Compare with the intended scoring population as well as saved row counts.
    raise NotImplementedError(
        "Define batch coverage, monitoring metrics, and alert ownership"
    )

    output = metrics_df.withColumn(
        "monitoring_run_id", F.lit(MONITORING_RUN_ID)
    ).withColumn("inference_run_id", F.lit(INFERENCE_RUN_ID))
    connector.write(
        output, destination=f"{SCHEMA_FQN}.monitoring_metrics", mode=WriteMode.APPEND
    )
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
