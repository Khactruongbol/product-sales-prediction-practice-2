import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.collect_data import validate_raw_schema
from src.config import (
    BEST_MODEL_INFO_PATH,
    BEST_MODEL_PATH,
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    FEATURE_SCHEMA_PATH,
    FIGURES_DIR,
    MODEL_COMPARISON_PATH,
    RAW_DATA_PATH,
    TEST_DATA_PATH,
    TRAIN_DATA_PATH,
    VALIDATION_DATA_PATH,
)
from src.evaluate import validate_figure_artifacts
from src.predict import predict_units
from src.prepare_data import add_history_features, temporal_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FINAL_NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "99_product_sales_workflow.ipynb"


def test_raw_schema_is_valid():
    stats = validate_raw_schema(RAW_DATA_PATH)
    assert stats["rows"] > 0
    assert stats["columns"] == 15


def test_history_features_only_use_previous_rows():
    dates = pd.date_range("2024-01-01", periods=35, freq="D")
    frame = pd.DataFrame(
        {
            "date": dates,
            "store_id": "S001",
            "product_id": "P0001",
            "units_sold": np.arange(35, dtype=float),
        }
    )
    featured = add_history_features(frame)
    row = featured.loc[featured["date"] == pd.Timestamp("2024-01-29")].iloc[0]
    assert row["units_sold_lag_1"] == 27
    assert row["units_sold_lag_7"] == 21
    assert row["units_sold_lag_28"] == 0
    assert row["units_sold_rolling_mean_7"] == np.mean(np.arange(21, 28))


def test_temporal_partitions_do_not_overlap():
    frame = pd.DataFrame(
        {
            "date": np.repeat(pd.date_range("2024-01-01", periods=20, freq="D").strftime("%Y-%m-%d"), 2),
            "units_sold": 1,
        }
    )
    train, validation, test = temporal_split(frame)
    assert train["date"].max() < validation["date"].min() < test["date"].min()
    assert set(train["date"]).isdisjoint(validation["date"])
    assert set(validation["date"]).isdisjoint(test["date"])


def test_saved_temporal_partitions_and_feature_contract():
    train = pd.read_csv(TRAIN_DATA_PATH)
    validation = pd.read_csv(VALIDATION_DATA_PATH)
    test = pd.read_csv(TEST_DATA_PATH)
    assert train["date"].max() < validation["date"].min() < test["date"].min()
    assert "demand_forecast" not in FEATURE_COLUMNS
    assert set(FEATURE_COLUMNS).issubset(train.columns)


def test_model_metrics_figures_and_prediction_are_valid():
    assert BEST_MODEL_PATH.exists()
    assert FEATURE_SCHEMA_PATH.exists()
    info = json.loads(BEST_MODEL_INFO_PATH.read_text(encoding="utf-8"))
    comparison = pd.read_csv(MODEL_COMPARISON_PATH)
    assert len(comparison) == 5
    assert {"mae", "mse", "rmse", "r2"} == set(info["test_metrics"])
    assert np.isfinite(comparison.filter(regex="test_").select_dtypes("number").to_numpy()).all()
    assert len(validate_figure_artifacts()) == 9

    test = pd.read_csv(TEST_DATA_PATH, dtype={column: "string" for column in CATEGORICAL_COLUMNS})
    row = test.iloc[0]
    values = {column: row[column] for column in FEATURE_COLUMNS if column not in {
        "year", "month", "quarter", "iso_week", "day_of_week", "is_weekend", "is_month_start", "is_month_end",
        "month_sin", "month_cos", "day_of_week_sin", "day_of_week_cos"
    }}
    values["date"] = row["date"]
    model = joblib.load(BEST_MODEL_PATH)
    prediction = predict_units(model, values)
    assert np.isfinite(prediction)
    assert prediction >= 0


def test_final_notebook_is_report_only_and_all_images_exist():
    notebook = json.loads(FINAL_NOTEBOOK_PATH.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert len(notebook["cells"]) == 19
    assert all(cell["cell_type"] == "markdown" for cell in notebook["cells"])
    assert not any(cell["cell_type"] == "code" for cell in notebook["cells"])

    markdown = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    image_targets = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", markdown)
    assert len(image_targets) == 9
    for target in image_targets:
        image_path = (FINAL_NOTEBOOK_PATH.parent / target).resolve()
        assert image_path.exists(), f"Notebook image is missing: {target}"
        assert image_path.stat().st_size > 1000
