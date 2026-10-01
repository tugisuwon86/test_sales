"""Matplotlib figures for the analysis notebook. Each function returns a Figure."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from . import constants as C

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
RED, GRAY_MID = "#e34948", "#f0efec"
INK, INK_2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
# Diverging: blue = under-forecast (actual above forecast), red = over-forecast.
DIVERGING = LinearSegmentedColormap.from_list("bias", [RED, GRAY_MID, BLUE])


def style():
    plt.rcParams.update(
        {
            "figure.facecolor": "#fcfcfb",
            "axes.facecolor": "#fcfcfb",
            "axes.edgecolor": "#c3c2b7",
            "axes.labelcolor": INK_2,
            "axes.titlesize": 11,
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "font.size": 9,
            "legend.frameon": False,
            "lines.linewidth": 2,
        }
    )


def save(fig, name: str):
    C.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(C.FIGURE_DIR / f"{name}.png", dpi=130, bbox_inches="tight")
    return fig


def cohort_distribution(labeled: pd.DataFrame):
    """Median and interquartile band of annual sales by cohort, FSU vs DTO, plus store counts."""
    fig, (ax, ax_n) = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    for subtype, color in [("FSU", BLUE), ("DTO", ORANGE)]:
        q = labeled[labeled["subtype"] == subtype].groupby("cohort")[C.target].quantile([0.25, 0.5, 0.75]).unstack()
        ax.fill_between(q.index, q[0.25], q[0.75], color=color, alpha=0.15, lw=0)
        ax.plot(q.index, q[0.5], color=color, marker="o", ms=4, label=f"{subtype} median (band = middle 50%)")
    ax.set_ylabel("Annual sales ($M)")
    ax.set_title("First-year annual sales by cohort")
    ax.legend(loc="upper left")
    counts = labeled.groupby(["cohort", "subtype"]).size().unstack(fill_value=0)
    ax_n.bar(counts.index, counts["FSU"], color=BLUE, width=0.6, label="FSU")
    ax_n.bar(counts.index, counts.get("DTO", 0), bottom=counts["FSU"], color=ORANGE, width=0.6, label="DTO")
    ax_n.set_ylabel("Stores")
    ax_n.set_xlabel("Cohort (first-anniversary year)")
    fig.tight_layout()
    return save(fig, "cohort_distribution")


def region_small_multiples(labeled: pd.DataFrame):
    regions = sorted(labeled["region"].unique())
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharey=True)
    rng = np.random.default_rng(0)
    for ax, region in zip(axes.flat, regions):
        part = labeled[labeled["region"] == region]
        colors = np.where(part["subtype"] == "DTO", ORANGE, BLUE)
        ax.scatter(part["cohort"] + rng.uniform(-0.2, 0.2, len(part)), part[C.target], s=8, c=colors, alpha=0.45, lw=0)
        ax.plot(part.groupby("cohort")[C.target].median(), color=INK, lw=2)
        ax.set_title(f"{region} (n={len(part)})")
    for ax in axes[:, 0]:
        ax.set_ylabel("Annual sales ($M)")
    fig.suptitle("Stores by region: dots = stores (blue FSU, orange DTO), line = median", x=0.01, ha="left")
    fig.tight_layout()
    return save(fig, "region_small_multiples")


def accuracy_by_cohort(backtest: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, metric in zip(axes, ["WAPE", "MAPE"]):
        table = backtest.pivot(index="cohort", columns="model", values=metric) * 100
        ax.plot(table.index, table["base"], color=MUTED, marker="o", ms=4, label="Base ensemble")
        ax.plot(table.index, table["corrected"], color=BLUE, marker="o", ms=4, label="With residual correction")
        last = table.index.max()
        ax.annotate(f"{table.loc[last, 'corrected']:.1f}%", (last, table.loc[last, "corrected"]), xytext=(6, -2), textcoords="offset points", color=INK)
        ax.set_title(f"{metric} by cohort (walk-forward)")
        ax.set_ylabel(f"{metric} (%)")
        ax.set_xlabel("Cohort")
    axes[0].legend(loc="upper left")
    fig.tight_layout()
    return save(fig, "accuracy_by_cohort")


def actual_vs_forecast(frame: pd.DataFrame):
    actual, forecast = frame[C.target], np.exp(frame["log_corrected"])
    fig, (ax, ax_h) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax.scatter(forecast, actual, s=10, color=BLUE, alpha=0.4, lw=0)
    lo, hi = 0, max(actual.max(), forecast.max()) * 1.02
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1, ls="--")
    for band in [0.9, 1.1]:
        ax.plot([lo, hi], [lo * band, hi * band], color=GRID, lw=1)
    ax.set_xlabel("Forecast ($M)")
    ax.set_ylabel("Actual ($M)")
    ax.set_title("Actual vs forecast, all backtest cohorts (lines: exact and ±10%)")
    error = (forecast - actual) / actual * 100
    ax_h.hist(error.clip(-80, 80), bins=40, color=BLUE, edgecolor="#fcfcfb", lw=0.5)
    for x in [-10, 10]:
        ax_h.axvline(x, color=MUTED, lw=1, ls="--")
    within = (error.abs() <= 10).mean() * 100
    ax_h.set_title(f"Forecast error, % (clipped at ±80); {within:.0f}% of stores within ±10%")
    ax_h.set_xlabel("(forecast - actual) / actual, %")
    ax_h.set_ylabel("Stores")
    fig.tight_layout()
    return save(fig, "actual_vs_forecast")


def bias_heatmap(table: pd.DataFrame, title: str, name: str, limit: float = 20):
    """Rows = segments, columns = cohorts, values = mean centred residual in %. Blue = under, red = over."""
    fig, ax = plt.subplots(figsize=(1.0 + 0.75 * table.shape[1], 0.5 + 0.42 * table.shape[0]))
    image = ax.imshow(table.to_numpy(dtype=float), cmap=DIVERGING, norm=TwoSlopeNorm(0, -limit, limit), aspect="auto")
    ax.set_xticks(range(table.shape[1]), table.columns)
    ax.set_yticks(range(table.shape[0]), [str(i) for i in table.index])
    ax.grid(False)
    for (i, j), value in np.ndenumerate(table.to_numpy(dtype=float)):
        if np.isfinite(value):
            ax.text(j, i, f"{value:.0f}", ha="center", va="center", fontsize=8, color=INK)
    fig.colorbar(image, ax=ax, shrink=0.8, label="Mean residual, % (blue = under-forecast)")
    ax.set_title(title)
    fig.tight_layout()
    return save(fig, name)


def lag_correlation_bars(group_lags: pd.DataFrame):
    corr = group_lags["corr"]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    width = 0.26
    colors = [BLUE, ORANGE, AQUA]
    x = np.arange(len(corr.index))
    for i, lag in enumerate(corr.columns):
        ax.bar(x + (i - 1) * width, corr[lag], width=width - 0.03, color=colors[i], label=f"lag {lag} cohort(s)")
    ax.axhline(0, color="#c3c2b7", lw=1)
    ax.set_xticks(x, corr.index)
    ax.set_ylabel("Correlation of group error\nwith earlier cohort's error")
    ax.set_title("Which errors repeat? Positive = an earlier cohort's error predicts the next")
    ax.legend(ncol=3, loc="upper left")
    fig.tight_layout()
    return save(fig, "lag_correlation")


def cluster_profile(centroids: pd.DataFrame, summary: pd.DataFrame):
    fig, (ax, ax_r) = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw={"width_ratios": [3, 1.2]})
    values = centroids.clip(-2, 2)
    image = ax.imshow(values.to_numpy(), cmap=LinearSegmentedColormap.from_list("z", [ORANGE, GRAY_MID, BLUE]), norm=TwoSlopeNorm(0, -2, 2), aspect="auto")
    ax.set_xticks(range(values.shape[1]), [f"C{c}\n(n={summary.loc[c, 'stores']})" for c in values.columns])
    ax.set_yticks(range(values.shape[0]), values.index)
    ax.grid(False)
    fig.colorbar(image, ax=ax, shrink=0.7, label="Cluster centre, z-score (clipped ±2)")
    ax.set_title("Cluster characteristics")
    colors = np.where(summary["mean_residual_pct"] >= 0, BLUE, RED)
    ax_r.barh([f"C{c}" for c in summary.index], summary["mean_residual_pct"], color=colors, height=0.6)
    ax_r.axvline(0, color="#c3c2b7", lw=1)
    ax_r.invert_yaxis()
    ax_r.set_xlabel("Mean residual, %")
    ax_r.set_title("Error by cluster (blue = under)")
    fig.tight_layout()
    return save(fig, "cluster_profile")


def under_over_bars(profile: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 5))
    y = np.arange(len(profile))
    ax.barh(y - 0.2, profile["under_forecast_z"], height=0.38, color=BLUE, label="Most under-forecast 20%")
    ax.barh(y + 0.2, profile["over_forecast_z"], height=0.38, color=RED, label="Most over-forecast 20%")
    ax.axvline(0, color="#c3c2b7", lw=1)
    ax.set_yticks(y, profile.index)
    ax.set_xlabel("Mean z-score vs all stores")
    ax.set_title("What distinguishes under- from over-forecast stores?")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    fig.tight_layout()
    return save(fig, "under_over_profile")


def feature_importance_bars(importance: pd.DataFrame, top: int = 20):
    data = importance["mean"].head(top)[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.6 + 0.28 * len(data)))
    ax.barh(data.index, data.values, color=BLUE, height=0.65)
    ax.set_xlabel("Share of LightGBM split gain, % (mean of base models)")
    ax.set_title(f"Top {top} features")
    fig.tight_layout()
    return save(fig, "feature_importance")


def forecast_distribution(forecast: pd.DataFrame, labeled: pd.DataFrame):
    latest = labeled[labeled["cohort"] == labeled["cohort"].max()]
    pipeline = forecast[forecast["location_type"] == "pipeline"]
    cohorts = [c for c, g in pipeline.groupby("cohort") if len(g) >= 20]
    fig, ax = plt.subplots(figsize=(9, 4))
    data = [latest[C.target]] + [pipeline.loc[pipeline["cohort"] == c, "forecast_annual_sales"] / 1e6 for c in cohorts]
    labels = [f"{latest['cohort'].iloc[0]}\nactual"] + [f"{c}\nforecast" for c in cohorts]
    parts = ax.boxplot(data, tick_labels=labels, widths=0.5, patch_artist=True, showfliers=False)
    for i, box in enumerate(parts["boxes"]):
        box.set(facecolor=MUTED if i == 0 else BLUE, alpha=0.35, edgecolor=INK_2)
    for median in parts["medians"]:
        median.set(color=INK, lw=2)
    ax.set_ylabel("Annual sales ($M)")
    ax.set_title("Pipeline forecasts vs latest cohort's actual sales (cohorts with 20+ stores)")
    fig.tight_layout()
    return save(fig, "forecast_distribution")
