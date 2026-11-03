# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "4"
# dependencies = []
# ///
# MAGIC %md
# MAGIC # 90 days sales forecast: feature engineering
# MAGIC Derives the opening-time flags and casts every column to the type the approved model was trained on.
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
from sales_forecast_config import TARGET_MAX, TARGET_MIN
from sales_forecast_pipeline import (
    DATE,
    ENTITY_KEYS,
    MODEL_INPUT_COLUMNS,
    REQUIRED_INPUT_COLUMNS,
    TARGET,
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

dbutils.widgets.text("preprocessing_run_id", "")
PREPROCESSING_RUN_ID = dbutils.widgets.get("preprocessing_run_id").strip()
if not PREPROCESSING_RUN_ID:
    raise ValueError("Provide a successful preprocessing_run_id")
if mlflow.get_run(PREPROCESSING_RUN_ID).info.status != "FINISHED":
    raise ValueError("Choose a successful upstream run")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Feature contract
# MAGIC Column names and types follow `90_days_features_with_attributes.parquet`,
# MAGIC the data the approved model was trained on. The 365-day `avg_monthly_sales`,
# MAGIC `avg_monthly_sales_log`, and `anniversary` columns are replaced by the
# MAGIC first-90-business-day label columns. Label columns are never model inputs.

# COMMAND ----------

FEATURE_SCHEMA = [
    ("location_num", "string"),
    ("location_name", "string"),
    ("latitude", "double"),
    ("longitude", "double"),
    ("subtype", "string"),
    ("location_type", "string"),
    ("location_status", "string"),
    ("state", "string"),
    ("dma", "string"),
    ("submarket", "string"),
    ("region", "string"),
    ("open_date", "date"),
    ("hex_string", "string"),
    ("ssam_flag", "int"),
    ("no_dt_flag", "int"),
    ("inline_flag", "string"),
    ("total_population_1", "double"),
    ("worker_population_1", "double"),
    ("daytime_population_1", "double"),
    ("avg_population_growth_1", "double"),
    ("total_households_1", "double"),
    ("total_households_per_sq_mile_1", "double"),
    ("density_classification_1", "string"),
    ("med_household_income_1", "double"),
    ("med_disposable_income_1", "double"),
    ("avg_disposable_income_1", "double"),
    ("annual_budget_expenditure_1", "double"),
    ("annual_budget_expenditure_index_1", "double"),
    ("med_home_value_1", "double"),
    ("avg_home_value_1", "double"),
    ("food_away_from_home_1", "double"),
    ("competitor_count_1", "double"),
    ("boundary_5", "int"),
    ("boundary_10", "int"),
    ("boundary_15", "int"),
    ("boundary_20", "int"),
    ("boundary_25", "int"),
    ("boundary_30", "int"),
    ("months_6m", "int"),
    ("months_12m", "int"),
    ("months_18m", "int"),
    ("months_24m", "int"),
    ("months_30m", "int"),
    ("months_36m", "int"),
    ("avg_monthly_sales_closest_1year", "double"),
    ("avg_monthly_sales_mean_1year", "double"),
    ("avg_monthly_sales_25th_1year", "double"),
    ("avg_monthly_sales_75th_1year", "double"),
    ("growth_rate_closest", "double"),
    ("growth_rate_mean", "double"),
    ("growth_rate_25th", "double"),
    ("growth_rate_75th", "double"),
    ("bigbox_1", "int"),
    ("bigbox_2", "int"),
    ("bigbox_3", "int"),
    ("competitor_1", "int"),
    ("competitor_2", "int"),
    ("competitor_3", "int"),
    ("highway_5", "int"),
    ("highway_10", "int"),
    ("highway_5min", "int"),
    ("highway_10min", "int"),
    ("close_highway_dist", "double"),
    ("close_highway_min", "double"),
    ("close_ramp_1", "int"),
    ("close_ramp_2", "int"),
    ("interstate_within_2min", "int"),
    ("median_average_daily_traffic_1", "double"),
    ("max_average_daily_traffic_1", "double"),
    ("median_average_daily_traffic_2", "double"),
    ("max_average_daily_traffic_2", "double"),
    ("median_average_daily_traffic_3", "double"),
    ("max_average_daily_traffic_3", "double"),
    ("month_of_opening", "int"),
    ("year", "int"),
    ("is_relocated", "boolean"),
    ("prototype", "string"),
    ("building_square_footage", "double"),
    ("total_site_square_footage", "double"),
    ("has_outside_seating", "boolean"),
    ("maximum_occupancy", "double"),
    ("has_cross_parking", "boolean"),
    ("has_bypass_lane", "boolean"),
    ("has_drivethru_door", "boolean"),
    ("total_drivethru_lanes", "int"),
    ("total_order_points", "double"),
    ("playground", "string"),
    # First-90-business-day label columns.
    ("business_days", "int"),
    ("label_end_date", "date"),
    ("avg_sales", "double"),
    ("total_sales", "double"),
    ("trans_count", "double"),
    ("total_trans", "double"),
]
LABEL_COLUMNS = {"business_days", "label_end_date", TARGET, "total_sales", "trans_count", "total_trans"}

missing = sorted(
    set([*ENTITY_KEYS, *REQUIRED_INPUT_COLUMNS, *MODEL_INPUT_COLUMNS])
    - {name for name, _ in FEATURE_SCHEMA}
)
if missing:
    raise ValueError(f"The feature contract is missing model inputs: {missing}")
if LABEL_COLUMNS & set(MODEL_INPUT_COLUMNS):
    raise ValueError("A label column is listed as a model input")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Build the features
# MAGIC Read this project's selected preprocessing run from the write catalog.
# MAGIC Derive features and labels, review model eligibility, and check any new
# MAGIC feature keys. Trust the input checks already owned by preprocessing.
# MAGIC
# MAGIC `inline_flag` is 1 when the location name contains "In-Line"; this matches
# MAGIC 1,346 of the 1,347 training rows. Labelled rows outside the approved
# MAGIC $5,000 to $70,000 daily range are removed, as they were in training.

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

    derived = (
        inputs.withColumn(
            "inline_flag",
            F.when(
                F.coalesce(F.col("location_name"), F.lit("")).rlike("(?i)in[\\s\\-]*line"),
                F.lit("1"),
            ).otherwise(F.lit("0")),
        )
        .withColumn("year", F.year(DATE))
        .withColumn("month_of_opening", F.month(DATE))
    )
    missing = sorted({name for name, _ in FEATURE_SCHEMA} - set(derived.columns))
    if missing:
        raise ValueError(f"Preprocessed inputs are missing columns: {missing}")

    def typed(name, spark_type):
        column = F.col(name)
        # Whole-number classes, so the category is "6" rather than "6.0".
        if name == "density_classification_1":
            column = column.cast("int")
        return column.cast(spark_type).alias(name)

    features_df = derived.select(*[typed(name, t) for name, t in FEATURE_SCHEMA])
    out_of_range = F.col(TARGET).isNotNull() & ~F.col(TARGET).between(TARGET_MIN, TARGET_MAX)
    mlflow.log_metric("out_of_range_labels", features_df.where(out_of_range).count())
    features_df = features_df.where(~out_of_range)

    # ENTITY_KEYS is location_num, shared with training and batch inference.
    if features_df.groupBy(*ENTITY_KEYS).count().where("count > 1").limit(1).count():
        raise ValueError("Expected one row per location_num")
    mlflow.log_metric("labelled_rows", features_df.where(F.col(TARGET).isNotNull()).count())

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
