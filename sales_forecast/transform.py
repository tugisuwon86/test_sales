"""Data transformation: load raw tables, define cohorts, engineer features."""

import re

import numpy as np
import pandas as pd

from . import constants as C


def prepare_sales_features(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize types of the joined location / boundary / hexbin features."""
    df = df.copy()
    df["location_num"] = df["location_num"].astype(str).str.zfill(5)
    df["open_date"] = pd.to_datetime(df["open_date"])
    return df


def prepare_location_attributes(attributes: pd.DataFrame) -> pd.DataFrame:
    """Standardize types of the store-attribute table (relocation flag, store details)."""
    attributes = attributes.copy()
    attributes["location_number"] = attributes["location_number"].astype(str).str.zfill(5)
    attributes["projected_open_dt"] = pd.to_datetime(
        attributes["projected_open_dt"], errors="coerce"
    )
    # Accept booleans or "true"/"false" strings from upstream; anything else counts as not relocated.
    attributes["is_relocated"] = (
        attributes["is_relocated"].astype(str).str.strip().str.lower().isin(["true", "1", "t", "yes"])
    )
    return attributes


def _name_key(name) -> str:
    """Normalize a location name so a relocated store matches its '(OLD)' predecessor."""
    if not isinstance(name, str):
        return ""
    name = name.lower()
    name = re.sub(r"\((old|closed|old/closed)[^)]*\)", "", name)
    name = re.sub(
        r"\b(fsu|dto|drive[- ]thru only|drive[- ]thru|in-line|inline|dt|old|relo|terminated|closed)\b",
        "",
        name,
    )
    return re.sub(r"[^a-z0-9]+", " ", name).strip()


def build_store_attributes(attributes: pd.DataFrame) -> pd.DataFrame:
    """Relocation flag plus predecessor match.

    A predecessor is a closed/terminated location with the same normalized name that
    opened before this store. Its sales are not available, only its age.
    """
    attributes = attributes.copy()
    attributes["name_key"] = attributes["location_name"].map(_name_key)
    closed = attributes[
        attributes["status"].isin(["closed", "terminated"])
        & attributes["projected_open_dt"].notna()
    ]
    closed_by_key = {k: g.sort_values("projected_open_dt") for k, g in closed.groupby("name_key")}

    found, age = [], []
    for key, opened in zip(attributes["name_key"], attributes["projected_open_dt"]):
        candidates = closed_by_key.get(key)
        if candidates is not None and pd.notna(opened):
            candidates = candidates[candidates["projected_open_dt"] < opened]
        if candidates is None or len(candidates) == 0 or pd.isna(opened):
            found.append(0)
            age.append(np.nan)
            continue
        found.append(1)
        age.append((opened - candidates["projected_open_dt"].iloc[-1]).days / 365.25)

    return pd.DataFrame(
        {
            "location_num": attributes["location_number"].values,
            "is_relocated": attributes["is_relocated"].astype(int).values,
            "predecessor_found": found,
            "predecessor_age_years": age,
        }
    )


def engineer_features(df: pd.DataFrame, store_attributes: pd.DataFrame) -> pd.DataFrame:
    """Add target, cohort, and engineered features. Does not drop any rows."""
    df = df[df["subtype"].isin(C.INCLUDED_SUBTYPES)].copy()
    df = df.merge(store_attributes, on="location_num", how="left")
    df[["is_relocated", "predecessor_found"]] = (
        df[["is_relocated", "predecessor_found"]].fillna(0).astype(int)
    )
    df["predecessor_age_years"] = df["predecessor_age_years"].fillna(0)  # 0 = no predecessor

    for column in df.select_dtypes("number").columns:
        df[column] = df[column].replace([np.inf, -np.inf], np.nan)

    # Target and cohort
    df["anniversary"] = df["open_date"] + pd.DateOffset(years=1)
    df["cohort"] = df["anniversary"].dt.year
    df[C.target] = df["avg_monthly_sales"] * 12 / 1e6  # annualized, $M
    df[C.log_target] = np.log(df[C.target].where(df[C.target] > 0))

    # Time and format
    df["years_since_2015"] = (df["open_date"] - C.REFERENCE_YEAR).dt.days / 365.25
    df["is_dto"] = (df["subtype"] == "DTO").astype(int)
    df["month_of_opening"] = df["open_date"].dt.month
    for quarter in [2, 3, 4]:
        df[f"open_q{quarter}"] = (((df["month_of_opening"] - 1) // 3 + 1) == quarter).astype(int)

    # Log transforms (skewed population, spend, traffic, neighbour sales)
    logs = {"log_" + c: np.log1p(df[c].clip(lower=0)) for c in C.log_demographics + C.traffic}
    logs.update({"log_" + c: np.log(df[c].where(df[c] > 0) * 12 / 1e6) for c in C.neighbor_sales})
    df = pd.concat([df, pd.DataFrame(logs, index=df.index)], axis=1)

    # Ratios and composite scores
    df["worker_resident_ratio"] = df["log_worker_population_1"] - df["log_total_population_1"]
    df["daytime_resident_ratio"] = df["log_daytime_population_1"] - df["log_total_population_1"]
    df["neighbor_spread"] = (
        df["log_avg_monthly_sales_75th_1year"] - df["log_avg_monthly_sales_25th_1year"]
    )
    df["neighbor_closest_vs_mean"] = (
        df["log_avg_monthly_sales_closest_1year"] - df["log_avg_monthly_sales_mean_1year"]
    )
    df["cannibalization_score"] = df["boundary_5"] * 3 + df["boundary_10"] * 2 + df["boundary_15"]
    df["log_cannibalization"] = np.log1p(df["cannibalization_score"].fillna(0))
    df["new_nearby_stores"] = df["months_6m"] + df["months_12m"]
    df["region"] = _impute_region(df)
    for column in ["state", "dma", "region", "submarket"]:
        df[column] = df[column].fillna("__MISSING__").astype(str)
    return df.reset_index(drop=True)


def _impute_region(df: pd.DataFrame) -> pd.Series:
    """Fill missing region (about 13% of pipeline stores) from existing stores in the same DMA, then state."""
    existing = df[df["location_type"].eq("existing") & df["region"].notna()]
    most_common = lambda s: s.value_counts().index[0]
    by_dma = existing.groupby("dma")["region"].agg(most_common)
    by_state = existing.groupby("state")["region"].agg(most_common)
    return df["region"].fillna(df["dma"].map(by_dma)).fillna(df["state"].map(by_state))


def split_cohorts(df: pd.DataFrame, as_of: pd.Timestamp = C.AS_OF_DATE):
    """Return (labeled, inference).

    labeled:   existing stores open more than one year at `as_of` with positive sales.
    inference: pipeline stores plus existing stores opened within the last year
               (no full first year yet). Older existing stores without a sales record
               are neither trained on nor forecast.
    """
    one_year_ago = as_of - pd.DateOffset(years=1)
    labeled_mask = (
        df["location_type"].eq("existing") & (df["open_date"] < one_year_ago) & df[C.target].gt(0)
    )
    inference_mask = df["location_type"].eq("pipeline") | (
        df["location_type"].eq("existing") & (df["open_date"] >= one_year_ago)
    )
    labeled = df[labeled_mask].reset_index(drop=True)
    inference = df[inference_mask & ~labeled_mask].reset_index(drop=True)
    assert labeled["anniversary"].max() <= as_of, "training rows must have a completed first year"
    return labeled, inference
