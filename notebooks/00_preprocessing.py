# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "4"
# dependencies = []
# ///
# MAGIC %md
# MAGIC # 90 days sales forecast: preprocessing
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

# COMMAND ----------

# MAGIC %md
# MAGIC ### Choose the source and prepare the population
# MAGIC Read shared sources from the read catalog. Put project input, key, and join
# MAGIC checks here; shared contract checks belong primarily in rdda_common.

# COMMAND ----------

with mlflow.start_run(tags={"project": PROJECT, "stage": "preprocessing"}) as run:
    PREPROCESSING_RUN_ID = run.info.run_id

    # TODO: choose a governed source in ctx.read_catalog or ctx.read_catalog_for(team).
    # Build preprocessed_df with the approved eligibility rules and input checks.
    raise NotImplementedError("Choose the source, row grain, and preprocessing rules")

    # Keep earlier runs. The business key plus preprocessing_run_id identifies a row.
    output = preprocessed_df.withColumn(
        "preprocessing_run_id", F.lit(PREPROCESSING_RUN_ID)
    )
    spark.sql(
        "create schema if not exists identifier(:schema)", args={"schema": SCHEMA_FQN}
    )
    connector.write(
        output, destination=f"{SCHEMA_FQN}.preprocessed_inputs", mode=WriteMode.APPEND
    )
    mlflow.log_param("output_table", f"{SCHEMA_FQN}.preprocessed_inputs")
    mlflow.log_metric("output_rows", output.count())

# COMMAND ----------

dbutils.jobs.taskValues.set(key="preprocessing_run_id", value=PREPROCESSING_RUN_ID)
