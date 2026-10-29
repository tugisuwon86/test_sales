"""Databricks adapters for the production attribute router.

The approved model logic lives in advanced_sales_models.py, attribute_router.py,
forecast_experiments.py and sales_forecast_config.py, copied unchanged from
production_ready. This module only normalizes Spark-loaded rows, wraps the
router for MLflow, and applies the capped subtype-and-region range rule.
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd
import mlflow
from forecast_experiments import DATE, TARGET
from advanced_sales_models import ACCESS, CATEGORIES, ORIGINAL, fit_model
from attribute_router import CORE_ATTRIBUTES, AttributeSubtypeRouter
from sales_forecast_config import FEATURE_COLUMNS

ENTITY_KEYS = ["location_num"]
HORIZON_DAYS = 90
# Raw columns read by engineered_features; every other feature is derived in the model.
MODEL_INPUT_COLUMNS = list(
    dict.fromkeys(["location_name", DATE, *ORIGINAL, *ACCESS, "dma", "submarket"])
)
REQUIRED_INPUT_COLUMNS = [DATE, *FEATURE_COLUMNS]
TEXT_INPUT_COLUMNS = {DATE, "location_name", "dma", "submarket"}
# Logged with the model so scoring imports the same code that trained it.
CODE_PATHS = [
    str(Path(__file__).with_name(name))
    for name in [
        "sales_forecast_pipeline.py",
        "forecast_experiments.py",
        "advanced_sales_models.py",
        "attribute_router.py",
        "sales_forecast_config.py",
    ]
]


def _category_text(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, (bool, int, float, np.bool_, np.integer, np.floating)):
        if float(value).is_integer():
            return str(int(value))
    return str(value)


def prepare_model_input(data):
    """Normalize Spark-loaded rows so training and scoring encode them identically.

    Arrow turns integer flags with nulls into floats, which would otherwise
    encode 1 and 1.0 as different categories between training and scoring.
    """
    d = data.copy()
    d[DATE] = pd.to_datetime(d[DATE])
    for c in d.columns:
        if c in CATEGORIES:
            d[c] = d[c].map(_category_text).astype(object)
        elif c in MODEL_INPUT_COLUMNS and c not in TEXT_INPUT_COLUMNS:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    return d


class SalesForecastModel(mlflow.pyfunc.PythonModel):
    """MLflow wrapper; returns the point forecast for each input row."""

    def __init__(self, router):
        self.router = router

    def predict(self, context, model_input, params=None):
        return self.router.predict(prepare_model_input(model_input))


def fit_router(train, attribute_config, fallback_config, min_core_fraction):
    return AttributeSubtypeRouter(
        fit_model(train, attribute_config),
        fit_model(train, fallback_config),
        min_core_fraction=min_core_fraction,
    )


def uses_attribute_model(data, min_core_fraction):
    complete = data.reindex(columns=CORE_ATTRIBUTES).notna().mean(axis=1)
    return (data.subtype.eq("FSU") & complete.ge(min_core_fraction)).to_numpy()


def wape(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return 100 * np.abs(y - p).sum() / np.abs(y).sum()


def metric_name(*parts):
    return "_".join(re.sub(r"[^0-9a-zA-Z]+", "_", str(p)).strip("_").lower() for p in parts)


def _flag_groups(frame):
    for column, label in [("ssam_flag", "SSAM"), ("inline_flag", "Inline")]:
        flag = pd.to_numeric(frame[column], errors="coerce").fillna(0).astype(int)
        for value in [0, 1]:
            group = frame[flag.eq(value).to_numpy()]
            if len(group):
                yield column, f"{label}: " + ("Yes" if value else "No"), group


def segment_wape(frame, split):
    """WAPE overall, by subtype, and by SSAM and inline status."""
    rows = [
        {"split": split, "dimension": "overall", "segment": "ALL",
         "locations": len(frame), "WAPE": wape(frame[TARGET], frame.prediction)}
    ]
    for subtype, g in frame.groupby("subtype"):
        rows.append({"split": split, "dimension": "subtype", "segment": subtype,
                     "locations": len(g), "WAPE": wape(g[TARGET], g.prediction)})
    for column, segment, g in _flag_groups(frame):
        rows.append({"split": split, "dimension": column, "segment": segment,
                     "locations": len(g), "WAPE": wape(g[TARGET], g.prediction)})
    return pd.DataFrame(rows, columns=["split", "dimension", "segment", "locations", "WAPE"])


RANGE_CALIBRATION_SCHEMA = (
    "level string, subtype string, region string, rows long, "
    "range_left_pct double, range_right_pct double, training_run_id string"
)


def calibrate_ranges(cal, base_stretch, max_stretch, min_group_rows, quantile):
    """Fit the side-specific stretch for every group the range rule can select.

    Signed errors set each side separately. Every side is clipped between the
    base and maximum stretch; no standard deviation is used.
    """
    required = [TARGET, "prediction", "subtype", "region"]
    cal = cal.replace([np.inf, -np.inf], np.nan).dropna(subset=required)
    cal = cal[cal.prediction.gt(0)]
    if len(cal) < min_group_rows:
        raise ValueError("Too few held-out rows to calibrate the range")
    ratio = cal[TARGET] / cal.prediction - 1

    def stretch(level, subtype, region, errors):
        lower_errors = -errors[errors.lt(0)]
        upper_errors = errors[errors.gt(0)]
        left = lower_errors.quantile(quantile) if len(lower_errors) else base_stretch
        right = upper_errors.quantile(quantile) if len(upper_errors) else base_stretch
        return {"level": level, "subtype": subtype, "region": region, "rows": len(errors),
                "range_left_pct": float(np.clip(left, base_stretch, max_stretch)),
                "range_right_pct": float(np.clip(right, base_stretch, max_stretch))}

    rows = [stretch("overall", None, None, ratio)]
    for subtype, g in cal.groupby("subtype"):
        if len(g) >= min_group_rows:
            rows.append(stretch("subtype", subtype, None, ratio.loc[g.index]))
    for (subtype, region), g in cal.groupby(["subtype", "region"]):
        if len(g) >= min_group_rows:
            rows.append(stretch("subtype_region", subtype, region, ratio.loc[g.index]))
    return pd.DataFrame(rows)


def apply_ranges(frame, calibration, point="forecast_point"):
    """Use subtype + region, then subtype, then overall, as in the approved rule."""
    overall = calibration[calibration.level.eq("overall")]
    if len(overall) != 1:
        raise ValueError("The range calibration needs exactly one overall row")
    overall = next(overall.itertuples())
    regional = {(r.subtype, r.region): r for r in
                calibration[calibration.level.eq("subtype_region")].itertuples()}
    by_subtype = {r.subtype: r for r in
                  calibration[calibration.level.eq("subtype")].itertuples()}
    chosen = [regional.get((s, r)) or by_subtype.get(s) or overall
              for s, r in zip(frame.subtype, frame.region)]
    out = frame.copy()
    out["range_left_pct"] = pd.Series([c.range_left_pct for c in chosen], index=out.index, dtype=float)
    out["range_right_pct"] = pd.Series([c.range_right_pct for c in chosen], index=out.index, dtype=float)
    out["range_calibration_level"] = pd.Series([c.level for c in chosen], index=out.index, dtype=object)
    out["range_calibration_rows"] = pd.Series([int(c.rows) for c in chosen], index=out.index, dtype="int64")
    out["forecast_lower"] = out[point] * (1 - out.range_left_pct)
    out["forecast_upper"] = out[point] * (1 + out.range_right_pct)
    return out


def range_coverage(check):
    """Share of actuals inside the applied range, by the reported segments."""
    covered = check[TARGET].between(check.forecast_lower, check.forecast_upper, inclusive="both")
    check = check.assign(covered=covered)
    rows = [{"dimension": "overall", "segment": "ALL", "locations": len(check),
             "coverage_pct": 100 * check.covered.mean()}]
    for column in ["subtype", "region"]:
        for segment, g in check.groupby(column):
            rows.append({"dimension": column, "segment": segment, "locations": len(g),
                         "coverage_pct": 100 * g.covered.mean()})
    for column, segment, g in _flag_groups(check):
        rows.append({"dimension": column, "segment": segment, "locations": len(g),
                     "coverage_pct": 100 * g.covered.mean()})
    return pd.DataFrame(rows, columns=["dimension", "segment", "locations", "coverage_pct"])


def fixed_range_coverage(check, point="forecast_point"):
    """Fixed proportional ranges for a direct width-versus-coverage comparison."""
    rows = []
    for pct in [.15, .20, .25]:
        for segment, g in [("Overall", check)] + list(check.groupby("subtype")):
            is_covered = g[TARGET].between(g[point] * (1 - pct), g[point] * (1 + pct), inclusive="both")
            rows.append({"range": f"±{int(round(pct * 100))}%", "subtype": segment,
                         "locations": len(g), "covered_locations": int(is_covered.sum()),
                         "coverage_pct": 100 * is_covered.mean()})
    return pd.DataFrame(rows)


def nulls_to_none(frame):
    """Spark cannot infer a type from NaN inside text columns."""
    out = frame.copy()
    for c in out.select_dtypes(include=["object", "string"]).columns:
        out[c] = out[c].astype(object).where(out[c].notna(), None)
    return out
