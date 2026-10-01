"""Feature selection, cleaning, and time-safe outcome features.

"Time-safe" features use other stores' sales. For a store being predicted they only use
stores in the training set; for training rows they only use stores from earlier cohorts,
so the model never sees features that were built from its own or later outcomes.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor
from sklearn.neighbors import BallTree

from . import constants as C


class FeatureSelector:
    """Learns which columns to keep and how to encode categoricals, from training rows only."""

    def fit(self, train: pd.DataFrame):
        self.category_maps_ = {
            code: {v: i for i, v in enumerate(sorted(train[raw].unique()))}
            for code, raw in [("region_code", "region"), ("state_code", "state")]
        }
        encoded = self._encode(train)
        report = pd.DataFrame(
            {
                "null_share": encoded[C.add].isna().mean(),
                "n_unique": encoded[C.add].nunique(),
            }
        )
        report["keep"] = (report["null_share"] <= C.MAX_NULL_SHARE) & (report["n_unique"] > 1)
        report["reason"] = np.select(
            [report["null_share"] > C.MAX_NULL_SHARE, report["n_unique"] <= 1],
            ["too many nulls", "constant"],
            "",
        )
        self.report_ = report
        self.features_ = report.index[report["keep"]].tolist()
        return self

    def _encode(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for code, raw in [("region_code", "region"), ("state_code", "state")]:
            df[code] = df[raw].map(self.category_maps_[code]).fillna(-1).astype(int)
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._encode(df)[self.features_].astype(float)


def fit_trend(train: pd.DataFrame) -> HuberRegressor:
    """Robust log-linear trend of sales over opening date."""
    return HuberRegressor().fit(train[["years_since_2015"]], train[C.log_target])


def predict_trend(trend: HuberRegressor, df: pd.DataFrame, max_years: float) -> np.ndarray:
    """Trend prediction, capped at `max_years` (years since 2015) to avoid runaway extrapolation."""
    years = df[["years_since_2015"]].clip(upper=max_years)
    return trend.predict(years)


def _spatial_smooth(reference: pd.DataFrame, values: np.ndarray, target: pd.DataFrame, prefix: str):
    """Gaussian-kernel average of `values` at reference stores near each target store."""
    tree = BallTree(np.radians(reference[["latitude", "longitude"]].to_numpy()), metric="haversine")
    out = {}
    for k, bandwidth in C.SPATIAL_KERNELS:
        dist, ind = tree.query(np.radians(target[["latitude", "longitude"]].to_numpy()), k=min(k, len(reference)))
        weight = np.exp(-((dist * 6371 / bandwidth) ** 2))
        out[f"{prefix}_spatial_{k}"] = (weight * values[ind]).sum(1) / (weight.sum(1) + 1.0)
        out[f"{prefix}_spatial_weight_{k}"] = weight.sum(1)
    return pd.DataFrame(out, index=target.index)


def _target_encode(reference: pd.DataFrame, values: np.ndarray, target: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for column, shrinkage in C.TARGET_ENCODING_SHRINKAGE.items():
        stats = pd.Series(values, index=reference.index).groupby(reference[column]).agg(["sum", "count"])
        encoding = stats["sum"] / (stats["count"] + shrinkage)
        out["te_" + column] = target[column].map(encoding).fillna(0).to_numpy()
    return pd.DataFrame(out, index=target.index)


def _outcome_features(reference: pd.DataFrame, detrended: np.ndarray, target: pd.DataFrame) -> pd.DataFrame:
    vs_neighbors = (
        reference[C.log_target] - reference["log_avg_monthly_sales_mean_1year"]
    ).fillna(0).to_numpy()
    return pd.concat(
        [
            _spatial_smooth(reference, detrended, target, "detrended"),
            _spatial_smooth(reference, vs_neighbors, target, "vs_neighbors"),
            _target_encode(reference, detrended, target),
        ],
        axis=1,
    )


def outcome_features(train: pd.DataFrame, score: pd.DataFrame, trend, max_years: float):
    """Time-safe outcome features for training rows (earlier cohorts only) and scoring rows."""
    train_trend = predict_trend(trend, train, max_years)
    score_trend = predict_trend(trend, score, max_years)
    detrended = (train[C.log_target] - train_trend).to_numpy()

    train_parts = []
    for cohort in sorted(train["cohort"].unique()):
        current = train[train["cohort"] == cohort]
        earlier = (train["cohort"] < cohort).to_numpy()
        if earlier.sum() < 30:
            train_parts.append(pd.DataFrame(index=current.index))
            continue
        train_parts.append(_outcome_features(train[earlier], detrended[earlier], current))
    score_features = _outcome_features(train, detrended, score)
    train_features = pd.concat(train_parts).reindex(index=train.index, columns=score_features.columns)

    train_features["trend"] = train_trend
    score_features["trend"] = score_trend
    return train_features, score_features
