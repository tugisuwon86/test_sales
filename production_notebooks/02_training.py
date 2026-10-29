# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "4"
# dependencies = []
# ///
# MAGIC %md
# MAGIC # 90 days sales forecast: training
# MAGIC Fits the approved attribute-aware subtype router, saves held-out predictions and range calibration.
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
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup
from roost.connector import RoostConnector
from roost.connector.core.types import WriteMode
from roost.connector.formats import PySparkFormatHandler
from roost.connector.systems.database import (
    DatabricksConnectionConfig,
    DatabricksSystem,
)
from sales_forecast_config import MODEL_CONFIGS
from sales_forecast_pipeline import (
    CODE_PATHS,
    DATE,
    ENTITY_KEYS,
    HORIZON_DAYS,
    MODEL_INPUT_COLUMNS,
    RANGE_CALIBRATION_SCHEMA,
    REQUIRED_INPUT_COLUMNS,
    TARGET,
    SalesForecastModel,
    calibrate_ranges,
    fit_router,
    nulls_to_none,
    prepare_model_input,
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
# Feature Engineering lookups use the table registered by feature engineering.
FEATURE_TABLE = f"{RAW_SCHEMA_FQN}.features_by_run"

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

# Labels mature 90 days after opening. Blank as_of_date means today.
dbutils.widgets.text("as_of_date", "")
dbutils.widgets.text("confirmation_start", "2025-01-01")
dbutils.widgets.text("confirmation_end", "2025-04-02")
dbutils.widgets.text("test_start", "2025-07-01")
AS_OF_DATE = pd.Timestamp(
    dbutils.widgets.get("as_of_date").strip() or pd.Timestamp.today().normalize()
)
HOLDOUT_PHASES = [
    (
        "confirmation",
        pd.Timestamp(dbutils.widgets.get("confirmation_start").strip()),
        pd.Timestamp(dbutils.widgets.get("confirmation_end").strip()),
    ),
    ("test", pd.Timestamp(dbutils.widgets.get("test_start").strip()), None),
]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Model specification
# MAGIC The selected design is an attribute-aware subtype router chosen from 336 base
# MAGIC configurations and 34 chronological ensembles (15.60% confirmation WAPE,
# MAGIC 17.66% benchmark WAPE). `v2_015_lgb_rich` scores FSU rows with at least 70%
# MAGIC coverage across seven core site attributes; `v2_128_lgb_static` scores DTO,
# MAGIC MALL, and incomplete FSU rows.
# MAGIC
# MAGIC The planning range uses no standard deviation. Each side starts at 15%, is
# MAGIC adjusted from held-out signed errors by subtype and region, and is capped at
# MAGIC 20%. Groups with fewer than 12 rows fall back to subtype and then overall.

# COMMAND ----------

ATTRIBUTE_CONFIG = {
    "features": "rich",
    "transform": "peer_ratio",
    "loss": "regression",
    "reg_lambda": 10,
    "family": "lgb",
    "name": "v2_015_lgb_rich",
}
FALLBACK_CONFIG = MODEL_CONFIGS["v2_128_lgb_static"]
MIN_CORE_FRACTION = 0.70

BASE_STRETCH = 0.15
MAX_STRETCH = 0.20
MIN_GROUP_ROWS = 12
CALIBRATION_QUANTILE = 0.60

# COMMAND ----------

# MAGIC %md
# MAGIC ### Choose features, split, and fit
# MAGIC Keep the feature list, split, model parameters, fit, and prediction
# MAGIC adjustments here so another scientist can follow them. Preserve approved
# MAGIC model logic when migrating. Choose the split before fitting.
# MAGIC
# MAGIC Each held-out phase trains only on rows whose label matured before the
# MAGIC phase starts. The test phase is the approved benchmark and calibrates the
# MAGIC range. The registered model is then refit on every mature row.

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

    missing = sorted(set([*ENTITY_KEYS, TARGET, *REQUIRED_INPUT_COLUMNS]) - set(features.columns))
    if missing:
        raise ValueError(f"The feature run is missing model inputs: {missing}")
    # Only opening-time inputs read by the model; post-opening sales stay out.
    FEATURE_NAMES = [
        c for c in MODEL_INPUT_COLUMNS if c in features.columns and c not in ENTITY_KEYS
    ]
    mature_cutoff = AS_OF_DATE - pd.Timedelta(days=HORIZON_DAYS)
    training_df = features.where(
        F.col(TARGET).isNotNull() & (F.col(DATE) <= F.lit(mature_cutoff.date()))
    ).select("feature_run_id", *ENTITY_KEYS, F.col(TARGET).cast("double").alias(TARGET))
    training_set = fe.create_training_set(
        df=training_df,
        feature_lookups=[
            FeatureLookup(
                table_name=FEATURE_TABLE,
                lookup_key=["feature_run_id", *ENTITY_KEYS],
                feature_names=FEATURE_NAMES,
            )
        ],
        label=TARGET,
    )
    # One row per opened location, so the mature population fits in pandas.
    data = (
        prepare_model_input(training_set.load_df().toPandas())
        .sort_values([DATE, *ENTITY_KEYS])
        .reset_index(drop=True)
    )
    if data.duplicated(ENTITY_KEYS).any():
        raise ValueError("Expected one row per location in the feature run")
    mlflow.log_params(
        {
            "as_of_date": str(AS_OF_DATE.date()),
            "attribute_model": ATTRIBUTE_CONFIG["name"],
            "fallback_model": FALLBACK_CONFIG["name"],
            "min_core_fraction": MIN_CORE_FRACTION,
            "base_stretch": BASE_STRETCH,
            "max_stretch": MAX_STRETCH,
            "min_group_rows": MIN_GROUP_ROWS,
            "calibration_quantile": CALIBRATION_QUANTILE,
        }
    )
    mlflow.log_dict(
        {
            "attribute_config": ATTRIBUTE_CONFIG,
            "fallback_config": FALLBACK_CONFIG,
            "feature_names": FEATURE_NAMES,
            "holdout_phases": [
                [split, str(start.date()), None if end is None else str(end.date())]
                for split, start, end in HOLDOUT_PHASES
            ],
        },
        "training/model_specification.json",
    )
    mlflow.log_metric("training_rows", len(data))

    held_out = []
    for split, start, end in HOLDOUT_PHASES:
        train = data[data[DATE] < start - pd.Timedelta(days=HORIZON_DAYS)]
        valid = data[data[DATE] >= start]
        if end is not None:
            valid = valid[valid[DATE] < end]
        if not len(valid):
            raise ValueError(f"The {split} phase has no mature rows")
        phase_router = fit_router(train, ATTRIBUTE_CONFIG, FALLBACK_CONFIG, MIN_CORE_FRACTION)
        frame = valid[[*ENTITY_KEYS, DATE, "subtype", "region", TARGET]].copy()
        frame["ssam_flag"] = pd.to_numeric(valid.ssam_flag, errors="coerce")
        frame["inline_flag"] = pd.to_numeric(valid.inline_flag, errors="coerce")
        frame["dataset_split"] = split
        frame["prediction"] = phase_router.predict(valid)
        frame["used_attribute_model"] = uses_attribute_model(valid, MIN_CORE_FRACTION)
        frame["feature_run_id"] = FEATURE_RUN_ID
        held_out.append(frame)
        mlflow.log_metric(f"{split}_train_rows", len(train))
        mlflow.log_metric(f"{split}_rows", len(valid))
    held_out = pd.concat(held_out, ignore_index=True)
    predictions_df = spark.createDataFrame(nulls_to_none(held_out))

    # The range is calibrated on the test split, as in the approved benchmark.
    calibration = calibrate_ranges(
        held_out[held_out.dataset_split.eq("test")],
        BASE_STRETCH,
        MAX_STRETCH,
        MIN_GROUP_ROWS,
        CALIBRATION_QUANTILE,
    ).assign(training_run_id=TRAINING_RUN_ID)
    connector.write(
        spark.createDataFrame(nulls_to_none(calibration), RANGE_CALIBRATION_SCHEMA),
        destination=f"{SCHEMA_FQN}.range_calibration",
        mode=WriteMode.APPEND,
    )

    # Held-out membership is fixed above; refit on every mature row.
    model = SalesForecastModel(
        fit_router(data, ATTRIBUTE_CONFIG, FALLBACK_CONFIG, MIN_CORE_FRACTION)
    )
    MODEL_FLAVOR = mlflow.pyfunc

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
        code_paths=CODE_PATHS,
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
