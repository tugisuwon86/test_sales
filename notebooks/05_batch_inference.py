# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "4"
# dependencies = []
# ///
# MAGIC %md
# MAGIC # 90 days sales forecast: batch inference
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
from databricks.feature_engineering import FeatureEngineeringClient
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
fe = FeatureEngineeringClient()
connector = RoostConnector(
    context=ctx,
    format_handler=PySparkFormatHandler(),
    system=DatabricksSystem(DatabricksConnectionConfig(managed_ownership=False)),
)

dbutils.widgets.text("deployment_run_id", "")
DEPLOYMENT_RUN_ID = dbutils.widgets.get("deployment_run_id").strip()
if not DEPLOYMENT_RUN_ID:
    raise ValueError("Provide a successful deployment_run_id")
if mlflow.get_run(DEPLOYMENT_RUN_ID).info.status != "FINISHED":
    raise ValueError("Choose a successful upstream run")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Score the intended population
# MAGIC Use the exact model version recorded by deployment, including a retained
# MAGIC champion when the new candidate was rejected. Supply the selected feature
# MAGIC snapshot's predictor values when scoring an older model. Keep its registered
# MAGIC feature tables available; explicit inputs do not remove metadata dependencies.

# COMMAND ----------

with mlflow.start_run(tags={"project": PROJECT, "stage": "batch_inference"}) as run:
    INFERENCE_RUN_ID = run.info.run_id
    mlflow.log_param("deployment_run_id", DEPLOYMENT_RUN_ID)
    deployments = spark.sql(
        """
        select *
        from identifier(:table)
        where deployment_run_id = :run_id
        """,
        args={"table": f"{SCHEMA_FQN}.deployment_models", "run_id": DEPLOYMENT_RUN_ID},
    ).collect()
    if len(deployments) != 1:
        raise ValueError("Choose a deployment run with one selected model")
    selected = deployments[0]
    features = spark.sql(
        """
        select *
        from identifier(:table)
        where feature_run_id = :run_id
        """,
        args={
            "table": f"{SCHEMA_FQN}.features_by_run",
            "run_id": selected.feature_run_id,
        },
    )

    # TODO: build scoring_df with the approved population, entity keys, and predictors.
    # Include request-time features and the model's lookup keys. Omit label columns.
    raise NotImplementedError("Define the scoring population and required model inputs")

    predictions = fe.score_batch(model_uri=selected.model_uri, df=scoring_df)
    output = (
        predictions.withColumn("inference_run_id", F.lit(INFERENCE_RUN_ID))
        .withColumn("deployment_run_id", F.lit(DEPLOYMENT_RUN_ID))
        .withColumn("feature_run_id", F.lit(selected.feature_run_id))
        .withColumn("model_version", F.lit(selected.model_version))
    )
    connector.write(
        output, destination=f"{SCHEMA_FQN}.batch_output", mode=WriteMode.APPEND
    )
    # Review the saved batch so metrics do not rerun prediction lineage.
    saved = spark.table(f"{SCHEMA_FQN}.batch_output").where(
        F.col("inference_run_id") == INFERENCE_RUN_ID
    )
    mlflow.log_metric("output_rows", saved.count())
    mlflow.log_param("model_uri", selected.model_uri)

# COMMAND ----------

dbutils.jobs.taskValues.set(key="inference_run_id", value=INFERENCE_RUN_ID)
