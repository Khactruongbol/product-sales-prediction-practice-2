"""Clean raw data, create leakage-safe features, and make temporal splits."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    CATEGORICAL_COLUMNS,
    CLEANING_REPORT_PATH,
    DATA_DICTIONARY_PATH,
    FEATURE_COLUMNS,
    MODELING_DATA_PATH,
    NUMERIC_COLUMNS,
    RAW_DATA_PATH,
    TARGET_COLUMN,
    TEST_DATA_PATH,
    TRAIN_DATA_PATH,
    VALIDATION_DATA_PATH,
    ensure_directories,
)


def snake_case(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def add_time_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    dates = pd.to_datetime(result["date"], errors="raise")
    result["year"] = dates.dt.year
    result["month"] = dates.dt.month
    result["quarter"] = dates.dt.quarter
    result["iso_week"] = dates.dt.isocalendar().week.astype(int)
    result["day_of_week"] = dates.dt.dayofweek
    result["is_weekend"] = dates.dt.dayofweek.isin([5, 6]).astype(int)
    result["is_month_start"] = dates.dt.is_month_start.astype(int)
    result["is_month_end"] = dates.dt.is_month_end.astype(int)
    result["month_sin"] = np.sin(2 * np.pi * result["month"] / 12)
    result["month_cos"] = np.cos(2 * np.pi * result["month"] / 12)
    result["day_of_week_sin"] = np.sin(2 * np.pi * result["day_of_week"] / 7)
    result["day_of_week_cos"] = np.cos(2 * np.pi * result["day_of_week"] / 7)
    return result


def add_history_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.sort_values(["store_id", "product_id", "date"]).copy()
    keys = ["store_id", "product_id"]
    grouped = result.groupby(keys, sort=False)[TARGET_COLUMN]
    for lag in (1, 7, 14, 28):
        result[f"units_sold_lag_{lag}"] = grouped.shift(lag)
    for window in (7, 28):
        result[f"units_sold_rolling_mean_{window}"] = grouped.transform(
            lambda values: values.shift(1).rolling(window, min_periods=window).mean()
        )
        result[f"units_sold_rolling_std_{window}"] = grouped.transform(
            lambda values: values.shift(1).rolling(window, min_periods=window).std()
        )
    return result.sort_values(["date", "store_id", "product_id"]).reset_index(drop=True)


def temporal_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = np.array(sorted(pd.to_datetime(frame["date"]).unique()))
    if len(dates) < 10:
        raise ValueError("At least 10 unique dates are required for temporal splitting")
    train_end = max(1, int(len(dates) * 0.70))
    validation_end = max(train_end + 1, int(len(dates) * 0.85))
    train_dates = set(dates[:train_end])
    validation_dates = set(dates[train_end:validation_end])
    test_dates = set(dates[validation_end:])
    date_values = pd.to_datetime(frame["date"])
    train = frame[date_values.isin(train_dates)].copy()
    validation = frame[date_values.isin(validation_dates)].copy()
    test = frame[date_values.isin(test_dates)].copy()
    if train.empty or validation.empty or test.empty:
        raise ValueError("Temporal split produced an empty partition")
    if not (train["date"].max() < validation["date"].min() < test["date"].min()):
        raise AssertionError("Temporal partitions overlap")
    return train, validation, test


def prepare_dataset(raw_path: Path = RAW_DATA_PATH) -> dict[str, object]:
    ensure_directories()
    raw = pd.read_csv(raw_path)
    raw_rows = len(raw)
    raw.columns = [snake_case(column) for column in raw.columns]
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")

    numeric_raw = [
        "inventory_level",
        "units_sold",
        "units_ordered",
        "demand_forecast",
        "price",
        "discount",
        "competitor_pricing",
    ]
    for column in numeric_raw:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")

    before_duplicates = len(raw)
    raw = raw.drop_duplicates().copy()
    duplicates_removed = before_duplicates - len(raw)
    valid = (
        raw["date"].notna()
        & raw[TARGET_COLUMN].ge(0)
        & raw["price"].gt(0)
        & raw["discount"].between(0, 100)
        & raw["inventory_level"].ge(0)
        & raw["units_ordered"].ge(0)
    )
    invalid_rows_removed = int((~valid).sum())
    clean = raw.loc[valid].copy()
    for column in CATEGORICAL_COLUMNS:
        clean[column] = clean[column].astype("string")

    clean = add_time_features(clean)
    clean = add_history_features(clean)
    history_missing = clean[[c for c in NUMERIC_COLUMNS if "lag_" in c or "rolling_" in c]].isna().any(axis=1)
    history_rows_removed = int(history_missing.sum())
    modeling = clean.loc[~history_missing].copy()
    modeling["date"] = pd.to_datetime(modeling["date"]).dt.strftime("%Y-%m-%d")

    required = ["date", TARGET_COLUMN, "demand_forecast"] + FEATURE_COLUMNS
    modeling = modeling[required]
    train, validation, test = temporal_split(modeling)
    for path, partition in (
        (MODELING_DATA_PATH, modeling),
        (TRAIN_DATA_PATH, train),
        (VALIDATION_DATA_PATH, validation),
        (TEST_DATA_PATH, test),
    ):
        partition.to_csv(path, index=False)

    report = {
        "raw_rows": int(raw_rows),
        "duplicates_removed": int(duplicates_removed),
        "invalid_rows_removed": invalid_rows_removed,
        "history_rows_removed": history_rows_removed,
        "processed_rows": int(len(modeling)),
        "missing_values_raw": int(raw.isna().sum().sum()),
        "outlier_policy": "No target rows removed solely as statistical outliers; valid seasonal peaks are retained.",
        "leakage_exclusion": ["demand_forecast"],
        "split": {
            "train": _partition_summary(train),
            "validation": _partition_summary(validation),
            "test": _partition_summary(test),
        },
    }
    CLEANING_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    dictionary = {
        "target": {"name": TARGET_COLUMN, "meaning": "Units sold per store-product-day"},
        "excluded": {"demand_forecast": "Provided forecast; excluded to avoid target leakage"},
        "categorical_features": CATEGORICAL_COLUMNS,
        "numeric_features": NUMERIC_COLUMNS,
        "history_rule": "All lag and rolling features use shift(1) within store_id + product_id.",
    }
    DATA_DICTIONARY_PATH.write_text(json.dumps(dictionary, indent=2), encoding="utf-8")
    return report


def _partition_summary(frame: pd.DataFrame) -> dict[str, object]:
    return {
        "rows": int(len(frame)),
        "start_date": str(frame["date"].min()),
        "end_date": str(frame["date"].max()),
        "unique_dates": int(frame["date"].nunique()),
    }


if __name__ == "__main__":
    print(json.dumps(prepare_dataset(), indent=2))

