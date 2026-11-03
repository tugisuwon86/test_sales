# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "4"
# dependencies = []
# ///
# MAGIC %md
# MAGIC # 90 days sales forecast: preprocessing
# MAGIC Joins the upstream location, boundary, hexbin, and site-attribute sources and labels the first 90 business days of sales.
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
from pyspark.sql.types import DecimalType
from pyspark.sql.window import Window
from roost.context import RoostContext
from roost.connector import RoostConnector
from roost.connector.core.types import WriteMode
from roost.connector.formats import PySparkFormatHandler
from roost.connector.systems.database import (
    DatabricksConnectionConfig,
    DatabricksSystem,
)
from redshift import load_from_redshift

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

# Schema written by the location-attributes, boundary-features, and
# hexbin-level-feature notebooks.
dbutils.widgets.text("source_schema", f"{READ_CATALOG}.mom_capacity_ml_cc")
SOURCE_SCHEMA = dbutils.widgets.get("source_schema").strip()
DAILY_SALES_TABLE = "dx_prod.restfin_lh.daily_sales_extended"
# Redshift table behind loc_attributes.parquet.
SITE_ATTRIBUTES_TABLE = "rd_datamart.conf.location_attributes"
# The label is average daily net sales over the first 90 business days.
BUSINESS_DAYS = 90

# COMMAND ----------

# MAGIC %md
# MAGIC ### Choose the source and prepare the population
# MAGIC Read shared sources from the read catalog. Put project input, key, and join
# MAGIC checks here; shared contract checks belong primarily in rdda_common.
# MAGIC
# MAGIC The population matches the sales-forecast notebook: existing and pipeline
# MAGIC locations with a five-digit location number that have boundary and hexbin
# MAGIC features. A business day is a trading date with positive net sales on or
# MAGIC after the opening date. The label is set only once all 90 business days
# MAGIC exist; newer openings keep a null label and are scored in batch inference.

# COMMAND ----------

with mlflow.start_run(tags={"project": PROJECT, "stage": "preprocessing"}) as run:
    PREPROCESSING_RUN_ID = run.info.run_id
    mlflow.log_params(
        {
            "source_schema": SOURCE_SCHEMA,
            "daily_sales_table": DAILY_SALES_TABLE,
            "site_attributes_table": SITE_ATTRIBUTES_TABLE,
            "business_days": BUSINESS_DAYS,
        }
    )

    location_features = spark.sql(
        """
        select
            location_num,
            location_name,
            latitude,
            longitude,
            subtype,
            location_type,
            location_status,
            state,
            dma,
            submarket,
            region,
            opened_dt as open_date,
            hex_string,
            ssam_flag,
            no_dt_flag,
            total_population_1,
            worker_population_1,
            daytime_population_1,
            avg_population_growth_1,
            total_households_1,
            total_households_per_sq_mile_1,
            density_classification_1,
            med_household_income_1,
            med_disposable_income_1,
            avg_disposable_income_1,
            annual_budget_expenditure_1,
            annual_budget_expenditure_index_1,
            med_home_value_1,
            avg_home_value_1,
            food_away_from_home_1,
            competitor_count_1
        from identifier(:table)
        where location_type in ('existing', 'pipeline')
          and startswith(cast(location_num as string), '0')
        """,
        args={"table": f"{SOURCE_SCHEMA}.location_features"},
    )
    # Peer features describe nearby stores before this location opened.
    boundary_features = spark.sql(
        """
        select
            location_num,
            boundary_5,
            boundary_10,
            boundary_15,
            boundary_20,
            boundary_25,
            boundary_30,
            months_6m,
            months_12m,
            months_18m,
            months_24m,
            months_30m,
            months_36m,
            avg_monthly_sales_closest_1year,
            avg_monthly_sales_mean_1year,
            avg_monthly_sales_25th_1year,
            avg_monthly_sales_75th_1year,
            growth_rate_closest,
            growth_rate_mean,
            growth_rate_25th,
            growth_rate_75th
        from identifier(:table)
        """,
        args={"table": f"{SOURCE_SCHEMA}.boundary_features"},
    )
    hexbin_features = spark.sql(
        """
        select
            location_num,
            bigbox_1,
            bigbox_2,
            bigbox_3,
            competitor_1,
            competitor_2,
            competitor_3,
            highway_5,
            highway_10,
            highway_5min,
            highway_10min,
            close_highway_dist,
            close_highway_min,
            close_ramp_1,
            close_ramp_2,
            interstate_within_2min,
            median_average_daily_traffic_1,
            max_average_daily_traffic_1,
            median_average_daily_traffic_2,
            max_average_daily_traffic_2,
            median_average_daily_traffic_3,
            max_average_daily_traffic_3
        from identifier(:table)
        """,
        args={"table": f"{SOURCE_SCHEMA}.hexbin_level_features"},
    )
    site_attributes = load_from_redshift(
        f"""
        SELECT
            location_number AS location_num,
            is_relocated,
            prototype,
            building_square_footage,
            total_site_square_footage,
            has_outside_seating,
            maximum_occupancy,
            has_cross_parking,
            has_bypass_lane,
            has_drivethru_door,
            total_drivethru_lanes,
            total_order_points,
            playground
        FROM
            {SITE_ATTRIBUTES_TABLE}
        """
    ).withColumn(
        "location_num", F.lpad(F.col("location_num").cast("string"), 5, "0")
    ).dropDuplicates(["location_num"])

    for name, frame in [
        ("location_features", location_features),
        ("boundary_features", boundary_features),
        ("hexbin_level_features", hexbin_features),
    ]:
        if frame.groupBy("location_num").count().where("count > 1").limit(1).count():
            raise ValueError(f"{name} has more than one row per location_num")

    # Inner joins keep the sales-forecast notebook's population.
    population = (
        location_features.join(boundary_features, "location_num", "inner")
        .join(hexbin_features, "location_num", "inner")
        .join(site_attributes, "location_num", "left")
    )

    # First 90 business days from each location's opening date.
    daily_sales = (
        spark.table(DAILY_SALES_TABLE)
        .groupBy("location_num", "trans_date")
        .agg(
            F.sum("sls_net").cast("double").alias("sls_net"),
            F.sum("trans_cnt").cast("double").alias("trans_cnt"),
        )
        .join(population.select("location_num", "open_date"), "location_num", "inner")
        .where((F.col("trans_date") >= F.col("open_date")) & (F.col("sls_net") > 0))
        .withColumn(
            "business_day_num",
            F.row_number().over(
                Window.partitionBy("location_num").orderBy("trans_date")
            ),
        )
        .where(F.col("business_day_num") <= BUSINESS_DAYS)
    )
    labels = daily_sales.groupBy("location_num").agg(
        F.count("*").alias("business_days"),
        F.max("trans_date").alias("label_end_date"),
        F.avg("sls_net").alias("avg_sales"),
        F.sum("sls_net").alias("total_sales"),
        F.avg("trans_cnt").alias("trans_count"),
        F.sum("trans_cnt").alias("total_trans"),
    )
    complete = F.col("business_days") == BUSINESS_DAYS
    labels = labels.select(
        "location_num",
        "business_days",
        *[
            F.when(complete, F.col(c)).alias(c)
            for c in ["label_end_date", "avg_sales", "total_sales", "trans_count", "total_trans"]
        ],
    )
    preprocessed_df = population.join(labels, "location_num", "left").withColumn(
        "business_days", F.coalesce(F.col("business_days"), F.lit(0))
    )
    for field in preprocessed_df.schema.fields:
        if isinstance(field.dataType, DecimalType):
            preprocessed_df = preprocessed_df.withColumn(
                field.name, F.col(field.name).cast("double")
            )

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
    mlflow.log_metric("labelled_rows", output.where(F.col("avg_sales").isNotNull()).count())

# COMMAND ----------

dbutils.jobs.taskValues.set(key="preprocessing_run_id", value=PREPROCESSING_RUN_ID)
