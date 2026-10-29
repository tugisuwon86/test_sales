# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "4"
# dependencies = []
# ///
# MAGIC %md
# MAGIC # 90 days sales forecast: training
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

dbutils.widgets.text("feature_run_id", "")
FEATURE_RUN_ID = dbutils.widgets.get("feature_run_id").strip()
if not FEATURE_RUN_ID:
    raise ValueError("Provide a successful feature_run_id")
if mlflow.get_run(FEATURE_RUN_ID).info.status != "FINISHED":
    raise ValueError("Choose a successful upstream run")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Choose features, split, and fit
# MAGIC Keep the feature list, split, model parameters, fit, and prediction
# MAGIC adjustments here so another scientist can follow them. Preserve approved
# MAGIC model logic when migrating. Choose the split before fitting.

# COMMAND ----------

with mlflow.start_run(tags={"project": PROJECT, "stage": "training"}) as run:
    TRAINING_RUN_ID = run.info.run_id
    mlflow.log_param("feature_run_id", FEATURE_RUN_ID)
    features = spark.sql(
        """
        select *
        from identifier(:table)
        where feature_run_id = :run_id
        """,
        args={"table": f"{SCHEMA_FQN}.features_by_run", "run_id": FEATURE_RUN_ID},
    )

    # TODO: define training_set with fe.create_training_set and run-keyed lookups.
    # Fit model using the approved algorithm and MLflow MODEL_FLAVOR.
    # Produce predictions_df with entity keys, dataset_split, target, prediction,
    # and feature_run_id. Keep held-out membership before any full-data refit.
    # Bound the population before collecting to pandas, if the model needs it.
    raise NotImplementedError(
        "Define the training population, split, model, and held-out predictions"
    )

    # Materialize predictions once; validation reads these exact saved rows.
    output = predictions_df.withColumn("training_run_id", F.lit(TRAINING_RUN_ID))
    connector.write(
        output, destination=f"{SCHEMA_FQN}.training_predictions", mode=WriteMode.APPEND
    )

    # The registry-aware template uses Feature Engineering to retain lookup metadata.
    fe.log_model(
        model=model,
        artifact_path="model",
        flavor=MODEL_FLAVOR,
        training_set=training_set,
        infer_input_example=True,
    )
    registered = mlflow.register_model(f"runs:/{TRAINING_RUN_ID}/model", MODEL_NAME)
    candidate = spark.createDataFrame(
        [
            (
                TRAINING_RUN_ID,
                FEATURE_RUN_ID,
                TRAINING_RUN_ID,
                MODEL_NAME,
                str(registered.version),
            )
        ],
        "training_run_id string, feature_run_id string, model_run_id string, model_name string, model_version string",
    )
    connector.write(
        candidate,
        destination=f"{SCHEMA_FQN}.training_candidates",
        mode=WriteMode.APPEND,
    )
    mlflow.log_param("model_version", str(registered.version))

# COMMAND ----------

dbutils.jobs.taskValues.set(key="training_run_id", value=TRAINING_RUN_ID)
