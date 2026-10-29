# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "4"
# dependencies = []
# ///
# MAGIC %md
# MAGIC # 90 days sales forecast: deployment
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

dbutils.widgets.text("validation_run_id", "")
VALIDATION_RUN_ID = dbutils.widgets.get("validation_run_id").strip()
if not VALIDATION_RUN_ID:
    raise ValueError("Provide a successful validation_run_id")
if mlflow.get_run(VALIDATION_RUN_ID).info.status != "FINISHED":
    raise ValueError("Choose a successful upstream run")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Promote the accepted version or retain the champion
# MAGIC This registered-model example does no model work. Review alias ownership
# MAGIC and recovery with ML engineering before removing the template stop.
# MAGIC Serialize validation, deployment, and manual alias edits.

# COMMAND ----------

raise NotImplementedError(
    "Review promotion and rollback for this project before enabling deployment"
)

# COMMAND ----------

with mlflow.start_run(tags={"project": PROJECT, "stage": "deployment"}) as run:
    DEPLOYMENT_RUN_ID = run.info.run_id
    mlflow.log_param("validation_run_id", VALIDATION_RUN_ID)
    decisions = spark.sql(
        """
        select *
        from identifier(:table)
        where validation_run_id = :run_id
        """,
        args={"table": f"{SCHEMA_FQN}.validation_results", "run_id": VALIDATION_RUN_ID},
    ).collect()
    if len(decisions) != 1:
        raise ValueError("Choose a validation run with one saved decision")
    decision = decisions[0]
    if decision.model_name != MODEL_NAME or decision.status not in {
        "passed",
        "rejected",
    }:
        raise ValueError("Review the saved model and acceptance decision")
    candidate = client.get_model_version(MODEL_NAME, decision.model_version)
    if candidate.run_id != decision.model_run_id or candidate.status != "READY":
        raise ValueError("The registered candidate does not match the saved decision")

    aliases = client.get_registered_model(MODEL_NAME).aliases
    previous = aliases.get("champion")
    target = decision.model_version if decision.status == "passed" else previous
    if target is None:
        raise ValueError("No accepted candidate or existing champion is available")
    if decision.status == "passed":
        matching = aliases.get("challenger") == target
        already_promoted = previous == target and aliases.get("challenger") is None
        if not (matching or already_promoted):
            raise ValueError("The challenger does not match this validation decision")

    # Retain the previous version before any alias change, so a retry can be reviewed.
    mlflow.log_dict(
        {
            "model_name": MODEL_NAME,
            "previous_champion_version": previous,
            "selected_model_version": target,
        },
        "deployment/plan.json",
    )
    if decision.status == "passed":
        if previous != target:
            client.set_registered_model_alias(MODEL_NAME, "champion", target)
        current = client.get_registered_model(MODEL_NAME).aliases
        if current.get("champion") != target or current.get("challenger") not in {
            None,
            target,
        }:
            raise ValueError(
                "Aliases changed during deployment; inspect the saved plan"
            )
        if current.get("challenger") == target:
            client.delete_registered_model_alias(MODEL_NAME, "challenger")

    selected = spark.createDataFrame(
        [
            (
                DEPLOYMENT_RUN_ID,
                VALIDATION_RUN_ID,
                decision.feature_run_id,
                MODEL_NAME,
                str(target),
                f"models:/{MODEL_NAME}/{target}",
                previous,
            )
        ],
        "deployment_run_id string, validation_run_id string, feature_run_id string, model_name string, model_version string, model_uri string, previous_champion_version string",
    )
    connector.write(
        selected, destination=f"{SCHEMA_FQN}.deployment_models", mode=WriteMode.APPEND
    )

# COMMAND ----------

dbutils.jobs.taskValues.set(key="deployment_run_id", value=DEPLOYMENT_RUN_ID)
