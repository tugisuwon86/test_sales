"""Input data sources.

Production reads the latest data straight from the feature tables that the upstream pipeline
refreshes (same tables and joins as sales-forecast.ipynb). Local files are only a fallback for
development outside Databricks.

    sales_features, location_attributes = load_inputs()                 # auto-detect
    sales_features, location_attributes = load_inputs("spark", spark=spark, schema=adh.conf.default_schema_qualified)
"""

import pandas as pd

from . import constants as C

BOUNDARY_QUERY = """
SELECT
    location_num,
    boundary_5, boundary_10, boundary_15, boundary_20, boundary_25, boundary_30,
    months_6m, months_12m, months_18m, months_24m, months_30m, months_36m,
    avg_monthly_sales_closest_1year, avg_monthly_sales_mean_1year,
    avg_monthly_sales_25th_1year, avg_monthly_sales_75th_1year,
    growth_rate_closest, growth_rate_mean, growth_rate_25th, growth_rate_75th,
    avg_monthly_sales
FROM {schema}.boundary_features
"""

HEXBIN_QUERY = """
SELECT
    location_num,
    bigbox_1, bigbox_2, bigbox_3,
    competitor_1, competitor_2, competitor_3,
    highway_5, highway_10, highway_5min, highway_10min,
    close_highway_dist, close_highway_min, close_ramp_1, close_ramp_2, interstate_within_2min,
    median_average_daily_traffic_1, max_average_daily_traffic_1,
    median_average_daily_traffic_2, max_average_daily_traffic_2,
    median_average_daily_traffic_3, max_average_daily_traffic_3
FROM {schema}.hexbin_level_features
"""

LOCATION_QUERY = """
SELECT
    location_num, location_name, latitude, longitude, subtype, location_type, location_status,
    state, dma, submarket, region, opened_dt AS open_date, hex_string, ssam_flag, no_dt_flag,
    total_population_1, worker_population_1, daytime_population_1, avg_population_growth_1,
    total_households_1, total_households_per_sq_mile_1, density_classification_1,
    med_household_income_1, med_disposable_income_1, avg_disposable_income_1,
    annual_budget_expenditure_1, annual_budget_expenditure_index_1,
    med_home_value_1, avg_home_value_1, food_away_from_home_1, competitor_count_1
FROM {schema}.location_features
WHERE location_type IN ('existing', 'pipeline')
"""

LOCATION_ATTRIBUTES_QUERY = "SELECT * FROM {schema}.{table}"


def _active_spark():
    try:
        from pyspark.sql import SparkSession

        return SparkSession.getActiveSession()
    except ImportError:
        return None


def _to_pandas(spark_df) -> pd.DataFrame:
    """Spark -> pandas with Decimal columns cast to double (as in sales-forecast.ipynb)."""
    from pyspark.sql.functions import col
    from pyspark.sql.types import DecimalType

    for field in spark_df.schema.fields:
        if isinstance(field.dataType, DecimalType):
            spark_df = spark_df.withColumn(field.name, col(field.name).cast("double"))
    return spark_df.toPandas()


def load_from_spark(spark, schema: str, attributes_table: str = C.LOCATION_ATTRIBUTES_TABLE):
    """Query the live feature tables and the store-attribute table."""
    location = spark.sql(LOCATION_QUERY.format(schema=schema))
    boundary = spark.sql(BOUNDARY_QUERY.format(schema=schema))
    hexbin = spark.sql(HEXBIN_QUERY.format(schema=schema))
    df_all = location.join(boundary, "location_num", "inner").join(hexbin, "location_num", "inner")
    sales_features = _to_pandas(df_all)
    location_attributes = _to_pandas(
        spark.sql(LOCATION_ATTRIBUTES_QUERY.format(schema=schema, table=attributes_table))
    )
    return sales_features, location_attributes


def load_from_files(sales_path=C.SALES_FEATURES_PATH, attributes_path=C.LOCATION_ATTRIBUTES_PATH):
    """Development fallback: local extracts with the same columns as the live tables."""
    return pd.read_csv(sales_path, dtype={"location_num": str}), pd.read_parquet(attributes_path)


def load_inputs(source: str = C.DATA_SOURCE, spark=None, schema: str = None):
    """Return (sales_features, location_attributes) as pandas DataFrames.

    source: "spark" (live tables), "files" (local extracts), or "auto" (spark if a session is active).
    """
    spark = spark or _active_spark()
    if source == "auto":
        source = "spark" if spark is not None else "files"
    if source == "spark":
        assert spark is not None, "no active Spark session"
        schema = schema or C.FEATURE_SCHEMA
        assert schema, "set FEATURE_SCHEMA (e.g. adh.conf.default_schema_qualified)"
        sales_features, location_attributes = load_from_spark(spark, schema)
    else:
        sales_features, location_attributes = load_from_files()
    print(f"{source=} {sales_features.shape=} {location_attributes.shape=}")
    return sales_features, location_attributes
