"""Clustering diagnostics: which kinds of stores are under- or over-forecast?

Input is a residual frame (out-of-fold, after the residual correction) with `cohort`,
`centred` residual (log actual - log forecast, cohort median removed), and the
CLUSTER_FEATURES columns. Positive residual = under-forecast.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor, export_text

from . import constants as C

UNDER_OVER_THRESHOLD = 0.15  # |log residual| above this counts as a large miss (about 16%)


def standardize(frame: pd.DataFrame, features=C.CLUSTER_FEATURES) -> pd.DataFrame:
    values = SimpleImputer(strategy="median").fit_transform(frame[features])
    return pd.DataFrame(StandardScaler().fit_transform(values), columns=features, index=frame.index)


def fit_clusters(frame: pd.DataFrame, k: int = C.N_CLUSTERS, features=C.CLUSTER_FEATURES):
    """K-means on standardized location characteristics. Returns (labels, centroid table in z-scores)."""
    z = standardize(frame, features)
    kmeans = KMeans(k, n_init=10, random_state=0).fit(z)
    centroids = pd.DataFrame(kmeans.cluster_centers_, columns=features).T
    return pd.Series(kmeans.labels_, index=frame.index, name="cluster"), centroids


def cluster_summary(frame: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    g = frame.assign(cluster=labels).groupby("cluster")
    return pd.DataFrame(
        {
            "stores": g.size(),
            "mean_residual_pct": g["centred"].mean() * 100,
            "under_forecast_share": g["centred"].apply(lambda s: (s > UNDER_OVER_THRESHOLD).mean()),
            "over_forecast_share": g["centred"].apply(lambda s: (s < -UNDER_OVER_THRESHOLD).mean()),
            "median_sales_m": g[C.target].median(),
        }
    )


def cluster_persistence(frame: pd.DataFrame, labels: pd.Series, first_cohort: int = 2021) -> dict:
    """Does a cluster's mean residual in earlier cohorts predict its mean residual in cohort Y?"""
    table = pd.DataFrame({"cluster": labels, "cohort": frame["cohort"], "r": frame["centred"]})
    xs, ys = [], []
    for cohort in sorted(table["cohort"].unique()):
        if cohort < first_cohort:
            continue
        earlier = table[table["cohort"] < cohort].groupby("cluster")["r"].agg(["mean", "count"])
        current = table[table["cohort"] == cohort].groupby("cluster")["r"].agg(["mean", "count"])
        joined = current.join(earlier, rsuffix="_earlier").dropna()
        joined = joined[(joined["count"] >= 4) & (joined["count_earlier"] >= 10)]
        xs += list(joined["mean_earlier"])
        ys += list(joined["mean"])
    return dict(walk_forward_corr=np.corrcoef(xs, ys)[0, 1], pairs=len(xs))


def k_sweep(frame: pd.DataFrame, ks=(4, 6, 8, 10)) -> pd.DataFrame:
    rows = []
    for k in ks:
        labels, _ = fit_clusters(frame, k)
        spread = frame.groupby(labels)["centred"].mean().std() * 100
        rows.append(dict(k=k, between_cluster_sd_pct=spread, **cluster_persistence(frame, labels)))
    return pd.DataFrame(rows)


def under_over_profile(frame: pd.DataFrame, features=C.CLUSTER_FEATURES, quantile: float = 0.2) -> pd.DataFrame:
    """Mean z-score of each feature among the most under- vs most over-forecast stores."""
    z = standardize(frame, features)
    low, high = frame["centred"].quantile([quantile, 1 - quantile])
    under, over = frame["centred"] > high, frame["centred"] < low
    table = pd.DataFrame({"under_forecast_z": z[under].mean(), "over_forecast_z": z[over].mean()})
    table["gap"] = table["under_forecast_z"] - table["over_forecast_z"]
    return table.sort_values("gap")


def residual_tree(frame: pd.DataFrame, features=C.CLUSTER_FEATURES, depth: int = 3, min_leaf: int = 60):
    """Shallow regression tree on the residual: readable rules plus a walk-forward check."""
    x = frame[features].fillna(frame[features].median())
    predicted = pd.Series(np.nan, index=frame.index)
    for cohort in sorted(frame["cohort"].unique()):
        if cohort < 2021:
            continue
        train, test = frame["cohort"] < cohort, frame["cohort"] == cohort
        tree = DecisionTreeRegressor(max_depth=depth, min_samples_leaf=min_leaf, random_state=0)
        predicted[test] = tree.fit(x[train], frame.loc[train, "centred"]).predict(x[test])
    scored = predicted.notna()
    walk_forward_corr = np.corrcoef(predicted[scored], frame.loc[scored, "centred"])[0, 1]
    tree = DecisionTreeRegressor(max_depth=depth, min_samples_leaf=min_leaf, random_state=0).fit(x, frame["centred"])
    return export_text(tree, feature_names=list(features), decimals=2), walk_forward_corr
