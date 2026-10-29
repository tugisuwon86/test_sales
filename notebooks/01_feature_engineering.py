# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "4"
# dependencies = []
# ///
# MAGIC %md
# MAGIC # 90 days sales forecast: feature engineering
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

dbutils.widgets.text("preprocessing_run_id", "")
PREPROCESSING_RUN_ID = dbutils.widgets.get("preprocessing_run_id").strip()
if not PREPROCESSING_RUN_ID:
    raise ValueError("Provide a successful preprocessing_run_id")
if mlflow.get_run(PREPROCESSING_RUN_ID).info.status != "FINISHED":
    raise ValueError("Choose a successful upstream run")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Build the features
# MAGIC Read this project's selected preprocessing run from the write catalog.
# MAGIC Derive features and labels, review model eligibility, and check any new
# MAGIC feature keys. Trust the input checks already owned by preprocessing.

# COMMAND ----------

with mlflow.start_run(tags={"project": PROJECT, "stage": "feature_engineering"}) as run:
    FEATURE_RUN_ID = run.info.run_id
    mlflow.log_param("preprocessing_run_id", PREPROCESSING_RUN_ID)
    inputs = spark.sql(
        """
        select *
        from identifier(:table)
        where preprocessing_run_id = :run_id
        """,
        args={
            "table": f"{SCHEMA_FQN}.preprocessed_inputs",
            "run_id": PREPROCESSING_RUN_ID,
        },
    )

    # TODO: produce features_df and define ENTITY_KEYS at the actual feature grain.
    # Include the population needed for training and batch scoring.
    raise NotImplementedError("Define features, labels, eligibility, and feature keys")

    output = features_df.withColumn("feature_run_id", F.lit(FEATURE_RUN_ID))
    feature_table = f"{RAW_SCHEMA_FQN}.features_by_run"
    fe.create_table(
        name=feature_table,
        primary_keys=["feature_run_id", *ENTITY_KEYS],
        schema=output.schema,
        description="Feature snapshots retained by feature run.",
    )
    fe.write_table(name=feature_table, df=output, mode="merge")
    mlflow.log_param("features_table", feature_table)
    mlflow.log_metric("feature_rows", output.count())

# COMMAND ----------

dbutils.jobs.taskValues.set(key="feature_run_id", value=FEATURE_RUN_ID)
