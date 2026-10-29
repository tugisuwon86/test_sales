# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "4"
# dependencies = []
# ///
# MAGIC %md
# MAGIC # 90 days sales forecast: validation
# MAGIC Evaluates held-out WAPE and planning-range coverage, then accepts or rejects the candidate.
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
from mlflow.tracking import MlflowClient
from roost.connector import RoostConnector
from roost.connector.core.types import WriteMode
from roost.connector.formats import PySparkFormatHandler
from roost.connector.systems.database import (
    DatabricksConnectionConfig,
    DatabricksSystem,
)
from sales_forecast_pipeline import (
    apply_ranges,
    fixed_range_coverage,
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

# Acceptance limits. The approved design scored 17.66% test WAPE and 56.7%
# range coverage on 210 locations; confirm these limits with Su.
dbutils.widgets.text("max_test_wape", "20.0")
dbutils.widgets.text("min_range_coverage", "50.0")
dbutils.widgets.text("min_test_rows", "30")
MAX_TEST_WAPE = float(dbutils.widgets.get("max_test_wape"))
MIN_RANGE_COVERAGE = float(dbutils.widgets.get("min_range_coverage"))
MIN_TEST_ROWS = int(dbutils.widgets.get("min_test_rows"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Evaluate the saved candidate
# MAGIC Use held-out rows from this training run. Agree on metrics and limits with
# MAGIC Su; no thresholds are supplied by this template. Save the evidence and
# MAGIC decision before assigning a passing candidate as challenger.
# MAGIC
# MAGIC The candidate passes when the test split has enough rows, overall test WAPE
# MAGIC is within `max_test_wape`, and the capped dynamic range covers at least
# MAGIC `min_range_coverage` percent of test actuals. Range coverage is measured on
# MAGIC the same rows that calibrated it, so read it with the fixed-range table.

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
    # Confirmation rows are reported for context but do not decide acceptance.
    confirmation = spark.sql(
        """
        select *
        from identifier(:table)
        where training_run_id = :run_id and dataset_split = 'confirmation'
        """,
        args={"table": f"{SCHEMA_FQN}.training_predictions", "run_id": TRAINING_RUN_ID},
    ).toPandas()
    calibration = spark.sql(
        """
        select *
        from identifier(:table)
        where training_run_id = :run_id
        """,
        args={"table": f"{SCHEMA_FQN}.range_calibration", "run_id": TRAINING_RUN_ID},
    ).toPandas()
    test = predictions.toPandas()

    wape_table = pd.concat(
        [segment_wape(test, "test")]
        + ([segment_wape(confirmation, "confirmation")] if len(confirmation) else []),
        ignore_index=True,
    )
    check = apply_ranges(test.rename(columns={"prediction": "forecast_point"}), calibration)
    coverage = range_coverage(check)
    fixed = fixed_range_coverage(check)
    mlflow.log_table(wape_table, "validation/wape_by_segment.json")
    mlflow.log_table(coverage, "validation/range_coverage_by_segment.json")
    mlflow.log_table(fixed, "validation/fixed_range_coverage_by_subtype.json")
    display(wape_table.round(2))
    display(coverage.round(1))
    display(fixed.round(1))

    metric_values = {}
    for r in wape_table.itertuples():
        metric_values[metric_name(r.split, "wape", r.dimension, r.segment)] = float(r.WAPE)
        metric_values[metric_name(r.split, "locations", r.dimension, r.segment)] = float(r.locations)
    for r in coverage.itertuples():
        metric_values[metric_name("test", "range_coverage", r.dimension, r.segment)] = float(r.coverage_pct)
    for r in fixed.itertuples():
        metric_values[metric_name("test", "fixed_coverage", r.range, r.subtype)] = float(r.coverage_pct)
    test_wape = metric_values["test_wape_overall_all"]
    test_coverage = metric_values["test_range_coverage_overall_all"]
    accepted = bool(
        len(test) >= MIN_TEST_ROWS
        and test_wape <= MAX_TEST_WAPE
        and test_coverage >= MIN_RANGE_COVERAGE
    )
    mlflow.log_metrics(metric_values)
    mlflow.log_params(
        {
            "max_test_wape": MAX_TEST_WAPE,
            "min_range_coverage": MIN_RANGE_COVERAGE,
            "min_test_rows": MIN_TEST_ROWS,
        }
    )
    metrics_df = spark.createDataFrame(
        list(metric_values.items()), "metric_name string, metric_value double"
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
