"""Configuration for the annual new-store sales forecast.

Every tunable value used by the notebook lives here so a run is fully described by
this file plus the two input tables.
"""

import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = Path(os.environ.get("SALES_OUTPUT_DIR", ROOT / "sales_forecast_results"))
MODEL_PATH = OUTPUT_DIR / "sales_forecast_model.pkl"
FIGURE_DIR = OUTPUT_DIR / "figures"

# ----------------------------------------------------------------------------------
# Input data (see sources.py). Production reads the live tables refreshed upstream.
# ----------------------------------------------------------------------------------
# "auto" = live Spark tables when a Spark session is active, else local development extracts.
DATA_SOURCE = os.environ.get("SALES_DATA_SOURCE", "auto")
# Schema holding location_features / boundary_features / hexbin_level_features and the
# store-attribute table. On Databricks pass adh.conf.default_schema_qualified instead.
FEATURE_SCHEMA = os.environ.get("SALES_FEATURE_SCHEMA", "")
LOCATION_ATTRIBUTES_TABLE = os.environ.get("SALES_LOCATION_ATTRIBUTES_TABLE", "loc_attributes")
# Development-only local extracts (same columns as the live tables).
SALES_FEATURES_PATH = ROOT / "sales_features.csv"
LOCATION_ATTRIBUTES_PATH = ROOT / "loc_attributes.parquet"

# Stores need a full first year of sales by this date to be used for training.
# Defaults to today so each run uses the latest data; pin it with SALES_AS_OF_DATE to reproduce a run.
AS_OF_DATE = pd.Timestamp(os.environ.get("SALES_AS_OF_DATE", pd.Timestamp.today().normalize()))
INCLUDED_SUBTYPES = ["FSU", "DTO"]

# Cohort = calendar year of the first anniversary (open_date + 1 year).
FIRST_OOF_COHORT = 2018  # earliest cohort that gets an out-of-fold base prediction
BACKTEST_COHORTS = list(range(2020, 2027))  # cohorts scored in the walk-forward backtest
REFERENCE_YEAR = pd.Timestamp("2015-01-01")  # origin for the time-trend feature

# Target: annualized first-year sales in $M, modelled in log space.
target = "y"
log_target = "log_y"

# ----------------------------------------------------------------------------------
# Feature lists (same grouping idea as constants.py in the Databricks pipeline)
# ----------------------------------------------------------------------------------
categorical = ["region_code", "state_code"]

integer = [
    "ssam_flag",
    "no_dt_flag",
    "is_dto",
    "month_of_opening",
    "density_classification_1",
    "boundary_5",
    "boundary_10",
    "boundary_15",
    "boundary_20",
    "boundary_25",
    "boundary_30",
    "months_6m",
    "months_12m",
    "months_18m",
    "months_24m",
    "bigbox_1",
    "bigbox_2",
    "bigbox_3",
    "competitor_1",
    "competitor_2",
    "competitor_3",
    "competitor_count_1",
    "highway_5",
    "highway_10",
    "highway_5min",
    "highway_10min",
    "close_ramp_1",
    "close_ramp_2",
    "interstate_within_2min",
    "cannibalization_score",
    "new_nearby_stores",
]

numerical = [
    "years_since_2015",
    "latitude",
    "longitude",
    "avg_population_growth_1",
    "close_highway_dist",
    "close_highway_min",
    "growth_rate_mean",
    "growth_rate_closest",
    "worker_resident_ratio",
    "daytime_resident_ratio",
    "neighbor_spread",
    "neighbor_closest_vs_mean",
]

# log1p-transformed demographic / spend columns (raw name -> "log_" + name)
log_demographics = [
    "total_population_1",
    "worker_population_1",
    "daytime_population_1",
    "total_households_1",
    "total_households_per_sq_mile_1",
    "med_household_income_1",
    "med_disposable_income_1",
    "avg_disposable_income_1",
    "annual_budget_expenditure_1",
    "med_home_value_1",
    "avg_home_value_1",
    "food_away_from_home_1",
]

traffic = [
    "median_average_daily_traffic_1",
    "max_average_daily_traffic_1",
    "median_average_daily_traffic_2",
    "max_average_daily_traffic_2",
    "median_average_daily_traffic_3",
    "max_average_daily_traffic_3",
]

# Nearby existing-store sales (captures local price level and demand).
neighbor_sales = [
    "avg_monthly_sales_closest_1year",
    "avg_monthly_sales_mean_1year",
    "avg_monthly_sales_25th_1year",
    "avg_monthly_sales_75th_1year",
]

# Store attributes known before opening for pipeline stores (from loc_attributes.parquet).
# Building size, lanes, occupancy etc. are excluded: they are mostly blank for pipeline
# stores, and unopened stores show 0 lanes, which the model would read as "no drive-thru".
store_attributes = ["is_relocated", "predecessor_found", "predecessor_age_years"]

add = (
    categorical
    + integer
    + numerical
    + ["log_" + c for c in log_demographics]
    + ["log_" + c for c in traffic]
    + ["log_" + c for c in neighbor_sales]
    + store_attributes
)

# ----------------------------------------------------------------------------------
# Feature cleaning
# ----------------------------------------------------------------------------------
MAX_NULL_SHARE = 0.5  # drop a feature if more than this share of training rows is null

# ----------------------------------------------------------------------------------
# Base models: geometric mean of three gradient-boosted models.
# All predict log sales; "residual" models predict log sales minus a robust time trend.
# ----------------------------------------------------------------------------------
BASE_MODELS = {
    "lgb_log": dict(family="lgb", objective="l2", target="log"),
    "lgb_huber": dict(family="lgb", objective="huber", target="residual"),
    "cat_d6": dict(family="cat", depth=6, target="residual"),
}
LGB_PARAMS = dict(
    n_estimators=400,
    num_leaves=7,
    learning_rate=0.03,
    min_child_samples=20,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.7,
    reg_lambda=5,
    verbosity=-1,
    n_jobs=4,
    random_state=0,
)
CAT_PARAMS = dict(
    iterations=600,
    learning_rate=0.03,
    l2_leaf_reg=10,
    loss_function="RMSE",
    verbose=False,
    random_seed=0,
    thread_count=4,
    allow_writing_files=False,
)
# Trend is never extrapolated more than this many years past the latest training store.
MAX_TREND_EXTRAPOLATION_YEARS = 2.0

# Time-safe features built from earlier stores' outcomes.
SPATIAL_KERNELS = [(10, 25), (30, 75), (60, 200)]  # (neighbours, bandwidth km)
TARGET_ENCODING_SHRINKAGE = {"state": 10, "dma": 8, "submarket": 5, "region": 20}

# ----------------------------------------------------------------------------------
# Residual correction (learned from out-of-fold residuals of earlier cohorts)
# ----------------------------------------------------------------------------------
# Biases that persisted year after year in the walk-forward residual analysis and clustering.
# log_cannibalization: stores with many same-brand stores nearby are over-forecast.
RESIDUAL_FACTORS = ["is_relocated", "ssam_flag", "is_dto", "open_q2", "open_q3", "open_q4", "log_cannibalization"]

# Clustering of residuals (diagnostics only; clusters did not persist across cohorts).
CLUSTER_FEATURES = [
    "log_total_population_1",
    "log_total_households_per_sq_mile_1",
    "log_med_household_income_1",
    "worker_resident_ratio",
    "daytime_resident_ratio",
    "log_max_average_daily_traffic_1",
    "log_avg_monthly_sales_mean_1year",
    "neighbor_closest_vs_mean",
    "neighbor_spread",
    "cannibalization_score",
    "new_nearby_stores",
    "competitor_count_1",
    "bigbox_1",
    "close_highway_min",
    "highway_5",
    "log_food_away_from_home_1",
    "avg_population_growth_1",
]
N_CLUSTERS = 6
RESIDUAL_RIDGE_ALPHA = 20.0  # shrinkage of the factor offsets
# Move each cohort's median forecast this far toward the latest cohort's actual median.
LEVEL_ANCHOR_STRENGTH = 0.5
# Assumed annual growth of the cohort median beyond the latest observed cohort (log).
LEVEL_ANNUAL_GROWTH = 0.0

# Inference: forecast pipeline stores and existing stores opened within the last year.
# Open dates more than this many years after AS_OF_DATE are flagged as likely placeholders.
PLACEHOLDER_OPEN_YEARS = 6
