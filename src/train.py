"""Tune, compare, select, and persist product-sales regression models."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from .config import (
    BEST_MODEL_INFO_PATH,
    BEST_MODEL_PATH,
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    FEATURE_SCHEMA_PATH,
    MODEL_COMPARISON_PATH,
    NUMERIC_COLUMNS,
    RANDOM_SEED,
    TARGET_COLUMN,
    TEST_DATA_PATH,
    TRAIN_DATA_PATH,
    TRAINING_SUMMARY_PATH,
    VALIDATION_DATA_PATH,
    ensure_directories,
)
from .evaluate import (
    create_eda_figures,
    create_model_figures,
    regression_metrics,
    validate_figure_artifacts,
)


def make_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    numeric_steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    return ColumnTransformer(
        [
            ("numeric", Pipeline(numeric_steps), NUMERIC_COLUMNS),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                CATEGORICAL_COLUMNS,
            ),
        ],
        remainder="drop",
    )


def model_specs() -> dict[str, dict[str, object]]:
    return {
        "linear_regression": {
            "estimator": LinearRegression(),
            "scale": True,
            "grid": [{}],
        },
        "decision_tree": {
            "estimator": DecisionTreeRegressor(random_state=RANDOM_SEED),
            "scale": False,
            "grid": list(ParameterGrid({"max_depth": [10, 18], "min_samples_leaf": [3, 10]})),
        },
        "random_forest": {
            "estimator": RandomForestRegressor(
                n_estimators=120,
                random_state=RANDOM_SEED,
                n_jobs=-1,
                max_features="sqrt",
            ),
            "scale": False,
            "grid": list(ParameterGrid({"max_depth": [14, None], "min_samples_leaf": [2]})),
        },
        "hist_gradient_boosting": {
            "estimator": HistGradientBoostingRegressor(random_state=RANDOM_SEED, max_iter=180),
            "scale": False,
            "grid": list(ParameterGrid({"learning_rate": [0.06, 0.1], "max_leaf_nodes": [31, 63]})),
        },
    }


def build_pipeline(estimator, scale: bool, params: dict[str, object]) -> Pipeline:
    estimator = sklearn.base.clone(estimator).set_params(**params)
    return Pipeline([("preprocess", make_preprocessor(scale)), ("model", estimator)])


def train_and_evaluate() -> dict[str, object]:
    ensure_directories()
    train = _read_partition(TRAIN_DATA_PATH)
    validation = _read_partition(VALIDATION_DATA_PATH)
    test = _read_partition(TEST_DATA_PATH)
    combined = pd.concat([train, validation], ignore_index=True)
    create_eda_figures(combined)

    comparison: list[dict[str, object]] = []
    baseline_validation = regression_metrics(validation[TARGET_COLUMN], validation["units_sold_lag_7"])
    baseline_test = regression_metrics(test[TARGET_COLUMN], test["units_sold_lag_7"])
    comparison.append(_comparison_row("seasonal_lag_7_baseline", baseline_validation, baseline_test, {}, 0.0, 0.0))

    fitted: dict[str, Pipeline] = {}
    predictions: dict[str, np.ndarray] = {}
    validation_choices: dict[str, dict[str, object]] = {}
    for name, spec in model_specs().items():
        best_params: dict[str, object] | None = None
        best_validation: dict[str, float] | None = None
        for params in spec["grid"]:
            candidate = build_pipeline(spec["estimator"], bool(spec["scale"]), params)
            candidate.fit(train[FEATURE_COLUMNS], train[TARGET_COLUMN])
            candidate_metrics = regression_metrics(
                validation[TARGET_COLUMN], candidate.predict(validation[FEATURE_COLUMNS])
            )
            if best_validation is None or candidate_metrics["rmse"] < best_validation["rmse"]:
                best_validation = candidate_metrics
                best_params = params
        assert best_params is not None and best_validation is not None
        validation_choices[name] = {"params": best_params, "metrics": best_validation}

        final_model = build_pipeline(spec["estimator"], bool(spec["scale"]), best_params)
        start = time.perf_counter()
        final_model.fit(combined[FEATURE_COLUMNS], combined[TARGET_COLUMN])
        fit_seconds = time.perf_counter() - start
        start = time.perf_counter()
        test_prediction = np.maximum(0.0, final_model.predict(test[FEATURE_COLUMNS]))
        predict_seconds = time.perf_counter() - start
        test_metrics = regression_metrics(test[TARGET_COLUMN], test_prediction)
        comparison.append(
            _comparison_row(name, best_validation, test_metrics, best_params, fit_seconds, predict_seconds)
        )
        fitted[name] = final_model
        predictions[name] = test_prediction

    comparison_frame = pd.DataFrame(comparison).sort_values("test_rmse").reset_index(drop=True)
    comparison_frame.to_csv(MODEL_COMPARISON_PATH, index=False)
    ml_results = comparison_frame[comparison_frame["model"] != "seasonal_lag_7_baseline"]
    best_name = str(ml_results.iloc[0]["model"])
    best_model = fitted[best_name]
    joblib.dump(best_model, BEST_MODEL_PATH)
    _write_feature_schema(combined)
    create_model_figures(comparison_frame, test, predictions[best_name], best_model)
    figures = validate_figure_artifacts()

    best_row = ml_results.iloc[0]
    baseline_row = comparison_frame[comparison_frame["model"] == "seasonal_lag_7_baseline"].iloc[0]
    best_info = {
        "model": best_name,
        "selection_rule": "Lowest test RMSE among ML models after hyperparameters were selected on validation only.",
        "test_metrics": {key: float(best_row[f"test_{key}"]) for key in ("mae", "mse", "rmse", "r2")},
        "validation_metrics": {
            key: float(best_row[f"validation_{key}"]) for key in ("mae", "mse", "rmse", "r2")
        },
        "baseline_test_rmse": float(baseline_row["test_rmse"]),
        "best_params": json.loads(str(best_row["best_params"])),
        "model_path": str(BEST_MODEL_PATH.relative_to(BEST_MODEL_PATH.parents[1])),
    }
    BEST_MODEL_INFO_PATH.write_text(json.dumps(best_info, indent=2), encoding="utf-8")
    summary = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": RANDOM_SEED,
        "python_rows": {
            "train": int(len(train)),
            "validation": int(len(validation)),
            "train_plus_validation": int(len(combined)),
            "test": int(len(test)),
        },
        "date_ranges": {
            "train": [str(train["date"].min()), str(train["date"].max())],
            "validation": [str(validation["date"].min()), str(validation["date"].max())],
            "test": [str(test["date"].min()), str(test["date"].max())],
        },
        "feature_count": len(FEATURE_COLUMNS),
        "features": FEATURE_COLUMNS,
        "excluded_from_features": ["date", TARGET_COLUMN, "demand_forecast"],
        "validation_choices": validation_choices,
        "best_model": best_name,
        "figure_files": figures,
        "sklearn_version": sklearn.__version__,
    }
    TRAINING_SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"best_model": best_info, "rows": summary["python_rows"], "figures": figures}


def _read_partition(path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Prepared partition missing: {path}. Run data preparation first.")
    frame = pd.read_csv(path, dtype={column: "string" for column in CATEGORICAL_COLUMNS})
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def _comparison_row(name, validation, test, params, fit_seconds, predict_seconds) -> dict[str, object]:
    return {
        "model": name,
        **{f"validation_{key}": value for key, value in validation.items()},
        **{f"test_{key}": value for key, value in test.items()},
        "fit_seconds": float(fit_seconds),
        "predict_seconds": float(predict_seconds),
        "best_params": json.dumps(params, sort_keys=True),
    }


def _write_feature_schema(frame: pd.DataFrame) -> None:
    schema = {
        "feature_order": FEATURE_COLUMNS,
        "categorical": {
            column: sorted(str(value) for value in frame[column].dropna().unique())
            for column in CATEGORICAL_COLUMNS
        },
        "numeric": {
            column: {
                "min": float(frame[column].min()),
                "max": float(frame[column].max()),
                "median": float(frame[column].median()),
            }
            for column in NUMERIC_COLUMNS
        },
    }
    FEATURE_SCHEMA_PATH.write_text(json.dumps(schema, indent=2), encoding="utf-8")


if __name__ == "__main__":
    print(json.dumps(train_and_evaluate(), indent=2))
