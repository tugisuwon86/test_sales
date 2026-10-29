# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "4"
# dependencies = []
# ///
# MAGIC %md
# MAGIC # 90 days sales forecast: validation
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
from mlflow.tracking import MlflowClient
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
client = MlflowClient(registry_uri="databricks-uc")
connector = RoostConnector(
    context=ctx,
    format_handler=PySparkFormatHandler(),
    system=DatabricksSystem(DatabricksConnectionConfig(managed_ownership=False)),
)

dbutils.widgets.text("training_run_id", "")
TRAINING_RUN_ID = dbutils.widgets.get("training_run_id").strip()
if not TRAINING_RUN_ID:
    raise ValueError("Provide a successful training_run_id")
if mlflow.get_run(TRAINING_RUN_ID).info.status != "FINISHED":
    raise ValueError("Choose a successful upstream run")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Evaluate the saved candidate
# MAGIC Use held-out rows from this training run. Agree on metrics and limits with
# MAGIC Su; no thresholds are supplied by this template. Save the evidence and
# MAGIC decision before assigning a passing candidate as challenger.

# COMMAND ----------

with mlflow.start_run(tags={"project": PROJECT, "stage": "validation"}) as run:
    VALIDATION_RUN_ID = run.info.run_id
    mlflow.log_param("training_run_id", TRAINING_RUN_ID)
    candidates = spark.sql(
        """
        select *
        from identifier(:table)
        where training_run_id = :run_id
        """,
        args={"table": f"{SCHEMA_FQN}.training_candidates", "run_id": TRAINING_RUN_ID},
    ).collect()
    if len(candidates) != 1:
        raise ValueError("Choose a training run with one registered candidate")
    candidate = candidates[0]
    predictions = spark.sql(
        """
        select *
        from identifier(:table)
        where training_run_id = :run_id and dataset_split = 'test'
        """,
        args={"table": f"{SCHEMA_FQN}.training_predictions", "run_id": TRAINING_RUN_ID},
    )

    # TODO: calculate metrics_df (metric_name, metric_value) and accepted (bool).
    # Evaluate approved limits against held-out predictions, not the final refit's training rows.
    raise NotImplementedError(
        "Agree on the acceptance metrics and limits, then evaluate the holdouts"
    )

    metrics = metrics_df.withColumn(
        "validation_run_id", F.lit(VALIDATION_RUN_ID)
    ).withColumn("training_run_id", F.lit(TRAINING_RUN_ID))
    connector.write(
        metrics, destination=f"{SCHEMA_FQN}.validation_metrics", mode=WriteMode.APPEND
    )
    decision = spark.createDataFrame(
        [
            (
                VALIDATION_RUN_ID,
                TRAINING_RUN_ID,
                candidate.feature_run_id,
                candidate.model_run_id,
                candidate.model_name,
                candidate.model_version,
                "passed" if accepted else "rejected",
            )
        ],
        "validation_run_id string, training_run_id string, feature_run_id string, model_run_id string, model_name string, model_version string, status string",
    )
    connector.write(
        decision, destination=f"{SCHEMA_FQN}.validation_results", mode=WriteMode.APPEND
    )
    if accepted:
        client.set_registered_model_alias(
            candidate.model_name, "challenger", candidate.model_version
        )
    mlflow.set_tag("validation_status", "passed" if accepted else "rejected")

# COMMAND ----------

# A rejection is a saved decision; deployment can retain an existing champion.
dbutils.jobs.taskValues.set(key="validation_run_id", value=VALIDATION_RUN_ID)
