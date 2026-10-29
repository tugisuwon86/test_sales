# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "4"
# dependencies = []
# ///
# MAGIC %md
# MAGIC # 90 days sales forecast: batch inference
# MAGIC Scores locations without a mature 90-day label and adds the capped planning range.
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
from databricks.feature_engineering import FeatureEngineeringClient
from mlflow.tracking import MlflowClient
from roost.connector import RoostConnector
from roost.connector.core.types import WriteMode
from roost.connector.formats import PySparkFormatHandler
from roost.connector.systems.database import (
    DatabricksConnectionConfig,
    DatabricksSystem,
)
from sales_forecast_pipeline import (
    CORE_ATTRIBUTES,
    DATE,
    ENTITY_KEYS,
    HORIZON_DAYS,
    TARGET,
    apply_ranges,
    nulls_to_none,
    uses_attribute_model,
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
client = MlflowClient(registry_uri="databricks-uc")
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

# Locations whose label is missing or not yet mature on this date are scored.
# Blank as_of_date means today.
dbutils.widgets.text("as_of_date", "")
AS_OF_DATE = pd.Timestamp(
    dbutils.widgets.get("as_of_date").strip() or pd.Timestamp.today().normalize()
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Score the intended population
# MAGIC Use the exact model version recorded by deployment, including a retained
# MAGIC champion when the new candidate was rejected. Supply the selected feature
# MAGIC snapshot's predictor values when scoring an older model. Keep its registered
# MAGIC feature tables available; explicit inputs do not remove metadata dependencies.
# MAGIC
# MAGIC The planning range comes from the calibration saved by the selected model's
# MAGIC training run: subtype + region, then subtype, then overall, each side 15% to 20%.

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

    # The model looks up its predictors by feature_run_id and location, so the
    # scoring frame carries only keys. Labels are never passed to the model.
    mature_cutoff = (AS_OF_DATE - pd.Timedelta(days=HORIZON_DAYS)).date()
    pending = F.col(DATE) > F.lit(mature_cutoff)
    if TARGET in features.columns:
        pending = pending | F.col(TARGET).isNull()
    scoring_df = features.where(pending).select("feature_run_id", *ENTITY_KEYS)
    scoring_rows = scoring_df.count()
    if not scoring_rows:
        raise ValueError("No locations need a forecast; check the feature run and as_of_date")
    mlflow.log_param("as_of_date", str(AS_OF_DATE.date()))
    mlflow.log_metric("scoring_rows", scoring_rows)

    predictions = fe.score_batch(model_uri=selected.model_uri, df=scoring_df)

    # The model version's run is the training run that saved its calibration.
    model_run_id = client.get_model_version(
        selected.model_name, selected.model_version
    ).run_id
    min_core_fraction = float(mlflow.get_run(model_run_id).data.params["min_core_fraction"])
    calibration = spark.sql(
        """
        select *
        from identifier(:table)
        where training_run_id = :run_id
        """,
        args={"table": f"{SCHEMA_FQN}.range_calibration", "run_id": model_run_id},
    ).toPandas()
    descriptors = features.select(
        *ENTITY_KEYS,
        *[c for c in ["location_name", DATE, "subtype", "region"] if c in features.columns],
        F.col("ssam_flag").cast("double").alias("ssam_flag"),
        F.col("inline_flag").cast("double").alias("inline_flag"),
        *[c for c in CORE_ATTRIBUTES if c in features.columns],
    )
    # One row per pending location; bounded by the scoring population above.
    scored = (
        predictions.select(*ENTITY_KEYS, F.col("prediction").alias("forecast_point"))
        .join(descriptors, on=ENTITY_KEYS, how="left")
        .toPandas()
    )
    scored["used_attribute_model"] = uses_attribute_model(scored, min_core_fraction)
    scored = apply_ranges(scored, calibration).drop(
        columns=[c for c in CORE_ATTRIBUTES if c in scored]
    )
    predictions = spark.createDataFrame(nulls_to_none(scored))
    mlflow.log_param("model_run_id", model_run_id)

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
