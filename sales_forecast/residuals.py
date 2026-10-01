"""Residual diagnostics: do earlier cohorts' out-of-fold errors predict later cohorts' errors?

All functions take a frame with columns `cohort` and `residual` (log actual - log forecast,
out-of-fold). `centred` residuals remove each cohort's median so group effects are not
confused with the overall level of that year.
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree


def add_centred(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["centred"] = frame["residual"] - frame.groupby("cohort")["residual"].transform("median")
    return frame


def cohort_bias(frame: pd.DataFrame) -> pd.DataFrame:
    """Median residual per cohort and its lag-1 autocorrelation."""
    bias = frame.groupby("cohort")["residual"].median()
    return pd.DataFrame({"median_residual": bias, "previous_cohort": bias.shift(1)})


def group_lag_correlation(frame: pd.DataFrame, groups, lags=(1, 2, 3), min_count=3) -> pd.DataFrame:
    """Correlation between a group's mean centred residual in cohort Y-lag and in cohort Y."""
    rows = []
    for group in groups:
        stats = frame.groupby([group, "cohort"])["centred"].agg(["mean", "count"]).reset_index()
        for lag in lags:
            lagged = stats.assign(cohort=stats["cohort"] + lag)
            joined = stats.merge(lagged, on=[group, "cohort"], suffixes=("", "_lag"))
            joined = joined[(joined["count"] >= min_count) & (joined["count_lag"] >= min_count)]
            corr = joined["mean"].corr(joined["mean_lag"]) if len(joined) > 4 else np.nan
            rows.append(dict(group=group, lag=lag, pairs=len(joined), corr=corr))
    return pd.DataFrame(rows).pivot(index="group", columns="lag", values=["corr", "pairs"])


def spatial_lag_correlation(frame: pd.DataFrame, bandwidths=(10, 25, 60), lags=(1, 2, 3)) -> pd.DataFrame:
    """Correlation between a store's centred residual and nearby stores' residuals from cohort Y-lag."""
    rows = []
    cohorts = sorted(frame["cohort"].unique())
    for bandwidth in bandwidths:
        for lag in lags:
            xs, ys = [], []
            for cohort in cohorts:
                current, earlier = frame[frame["cohort"] == cohort], frame[frame["cohort"] == cohort - lag]
                if len(earlier) < 10:
                    continue
                tree = BallTree(np.radians(earlier[["latitude", "longitude"]].to_numpy()), metric="haversine")
                dist, ind = tree.query(np.radians(current[["latitude", "longitude"]].to_numpy()), k=min(15, len(earlier)))
                weight = np.exp(-((dist * 6371 / bandwidth) ** 2))
                has_neighbors = weight.sum(1) > 0.3
                smoothed = (weight * earlier["centred"].to_numpy()[ind]).sum(1) / weight.sum(1).clip(1e-9)
                xs += list(smoothed[has_neighbors])
                ys += list(current["centred"].to_numpy()[has_neighbors])
            rows.append(dict(bandwidth_km=bandwidth, lag=lag, n=len(xs), corr=np.corrcoef(xs, ys)[0, 1] if len(xs) > 4 else np.nan))
    return pd.DataFrame(rows).pivot(index="bandwidth_km", columns="lag", values=["corr", "n"])


def feature_bin_persistence(frame: pd.DataFrame, features, first_cohort=2021, bins=5) -> pd.DataFrame:
    """For each feature bin, does its mean residual over earlier cohorts predict cohort Y?

    wf_corr          walk-forward correlation (earlier-cohort bin mean vs current bin mean)
    lowest/highest   mean centred residual (%) in the lowest / highest bin, all cohorts
    sign_consistency share of cohorts in which the most persistent bin kept its overall sign
    """
    rows = []
    for feature in features:
        values = frame[feature]
        if values.nunique() > bins:
            binned = pd.qcut(values.rank(method="first"), bins, labels=False)
        else:
            binned = values.astype("category").cat.codes
        table = pd.DataFrame({"bin": binned, "cohort": frame["cohort"], "r": frame["centred"]})
        xs, ys = [], []
        for cohort in sorted(table["cohort"].unique()):
            if cohort < first_cohort:
                continue
            earlier = table[table["cohort"] < cohort].groupby("bin")["r"].mean()
            current = table[table["cohort"] == cohort].groupby("bin")["r"].agg(["mean", "count"])
            joined = current.join(earlier.rename("earlier")).dropna()
            joined = joined[joined["count"] >= 3]
            xs += list(joined["earlier"])
            ys += list(joined["mean"])
        overall = table.groupby("bin")["r"].mean()
        by_cohort = table.groupby(["bin", "cohort"])["r"].mean().unstack()
        consistency = np.sign(by_cohort).eq(np.sign(overall), axis=0).mean(axis=1)
        rows.append(
            dict(
                feature=feature,
                wf_corr=np.corrcoef(xs, ys)[0, 1],
                lowest_bin_pct=overall.iloc[0] * 100,
                highest_bin_pct=overall.iloc[-1] * 100,
                sign_consistency=consistency.max(),
            )
        )
    return pd.DataFrame(rows).sort_values("wf_corr", ascending=False).reset_index(drop=True)
