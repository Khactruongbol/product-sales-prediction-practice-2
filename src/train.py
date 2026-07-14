"""Tune, compare, select, audit, and persist product-sales regressors."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
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
    FEATURE_SIGNAL_AUDIT_PATH,
    LEAKAGE_BENCHMARK_PATH,
    LEGACY_TEST_R2,
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
    create_improvement_figures,
    create_model_figures,
    regression_metrics,
    validate_figure_artifacts,
)


def make_preprocessor(
    scale_numeric: bool,
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> ColumnTransformer:
    transformers: list[tuple[str, object, list[str]]] = []
    if numeric_columns:
        numeric_steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
        if scale_numeric:
            numeric_steps.append(("scaler", StandardScaler()))
        transformers.append(("numeric", Pipeline(numeric_steps), numeric_columns))
    if categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_columns,
            )
        )
    return ColumnTransformer(transformers, remainder="drop")


def model_specs() -> dict[str, dict[str, object]]:
    full_features = {"numeric": NUMERIC_COLUMNS, "categorical": CATEGORICAL_COLUMNS}
    return {
        "inventory_only_linear": {
            "estimator": LinearRegression(),
            "scale": True,
            "grid": [{}],
            "numeric": ["inventory_level"],
            "categorical": [],
        },
        "linear_regression": {
            "estimator": LinearRegression(),
            "scale": True,
            "grid": [{}],
            **full_features,
        },
        "ridge_regression": {
            "estimator": Ridge(),
            "scale": True,
            "grid": list(ParameterGrid({"alpha": [100.0, 1000.0, 10000.0]})),
            **full_features,
        },
        "decision_tree": {
            "estimator": DecisionTreeRegressor(random_state=RANDOM_SEED),
            "scale": False,
            "grid": list(ParameterGrid({"max_depth": [10, 18], "min_samples_leaf": [3, 10]})),
            **full_features,
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
            **full_features,
        },
        "hist_gradient_boosting": {
            "estimator": HistGradientBoostingRegressor(random_state=RANDOM_SEED, max_iter=180),
            "scale": False,
            "grid": list(ParameterGrid({"learning_rate": [0.06, 0.1], "max_leaf_nodes": [31, 63]})),
            **full_features,
        },
        "extra_trees": {
            "estimator": ExtraTreesRegressor(
                n_estimators=160,
                random_state=RANDOM_SEED,
                n_jobs=-1,
                max_features=1.0,
            ),
            "scale": False,
            "grid": list(ParameterGrid({"max_depth": [14, None], "min_samples_leaf": [2, 5]})),
            **full_features,
        },
    }


def build_pipeline(estimator, scale: bool, params: dict[str, object], spec: dict[str, object]) -> Pipeline:
    fitted_estimator = sklearn.base.clone(estimator).set_params(**params)
    preprocessor = make_preprocessor(
        scale,
        list(spec["numeric"]),
        list(spec["categorical"]),
    )
    return Pipeline([("preprocess", preprocessor), ("model", fitted_estimator)])


def train_and_evaluate() -> dict[str, object]:
    ensure_directories()
    train = _read_partition(TRAIN_DATA_PATH)
    validation = _read_partition(VALIDATION_DATA_PATH)
    test = _read_partition(TEST_DATA_PATH)
    combined = pd.concat([train, validation], ignore_index=True)
    complete = pd.concat([combined, test], ignore_index=True)
    create_eda_figures(combined)

    comparison: list[dict[str, object]] = []
    baseline_validation = regression_metrics(validation[TARGET_COLUMN], validation["units_sold_lag_7"])
    baseline_test = regression_metrics(test[TARGET_COLUMN], test["units_sold_lag_7"])
    comparison.append(_comparison_row("seasonal_lag_7_baseline", baseline_validation, baseline_test, {}, 0.0, 0.0))

    specs = model_specs()
    fitted: dict[str, Pipeline] = {}
    predictions: dict[str, np.ndarray] = {}
    validation_predictions: dict[str, np.ndarray] = {}
    validation_choices: dict[str, dict[str, object]] = {}
    for name, spec in specs.items():
        best_params: dict[str, object] | None = None
        best_validation: dict[str, float] | None = None
        best_validation_prediction: np.ndarray | None = None
        for params in spec["grid"]:
            candidate = build_pipeline(spec["estimator"], bool(spec["scale"]), params, spec)
            candidate.fit(train[FEATURE_COLUMNS], train[TARGET_COLUMN])
            candidate_prediction = np.maximum(0.0, candidate.predict(validation[FEATURE_COLUMNS]))
            candidate_metrics = regression_metrics(validation[TARGET_COLUMN], candidate_prediction)
            if best_validation is None or candidate_metrics["rmse"] < best_validation["rmse"]:
                best_validation = candidate_metrics
                best_params = params
                best_validation_prediction = candidate_prediction
        assert best_params is not None and best_validation is not None and best_validation_prediction is not None
        validation_predictions[name] = best_validation_prediction
        selected_features = list(spec["categorical"]) + list(spec["numeric"])
        validation_choices[name] = {
            "params": best_params,
            "metrics": best_validation,
            "selected_features": selected_features,
        }

        final_model = build_pipeline(spec["estimator"], bool(spec["scale"]), best_params, spec)
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
    best_name = min(validation_choices, key=lambda model: validation_choices[model]["metrics"]["rmse"])
    best_model = fitted[best_name]
    best_row = comparison_frame[comparison_frame["model"] == best_name].iloc[0]
    best_test_metrics = {key: float(best_row[f"test_{key}"]) for key in ("mae", "mse", "rmse", "r2")}
    if best_test_metrics["r2"] < LEGACY_TEST_R2:
        raise AssertionError(
            f"Improved safe model regressed below legacy R2: {best_test_metrics['r2']} < {LEGACY_TEST_R2}"
        )

    validation_abs_error = np.abs(
        validation[TARGET_COLUMN].to_numpy(dtype=float) - validation_predictions[best_name]
    )
    interval_90 = float(np.quantile(validation_abs_error, 0.90))
    selected_features = validation_choices[best_name]["selected_features"]
    joblib.dump(best_model, BEST_MODEL_PATH)
    _write_feature_schema(combined, selected_features)
    create_model_figures(comparison_frame, test, predictions[best_name], best_model)

    signal_audit = _write_signal_audit(complete)
    leakage_prediction = test["demand_forecast"].clip(lower=0, upper=test["inventory_level"])
    leakage_test_metrics = regression_metrics(test[TARGET_COLUMN], leakage_prediction)
    leakage_validation_metrics = regression_metrics(
        validation[TARGET_COLUMN],
        validation["demand_forecast"].clip(lower=0, upper=validation["inventory_level"]),
    )
    leakage_benchmark = {
        "deployable": False,
        "feature": "demand_forecast",
        "safety_reason": "Near-direct target proxy in this synthetic dataset; excluded from saved model and UI.",
        "correlation_with_target": signal_audit["correlations"]["demand_forecast"],
        "validation_metrics": leakage_validation_metrics,
        "test_metrics": leakage_test_metrics,
        "safe_model": best_name,
        "safe_test_metrics": best_test_metrics,
    }
    LEAKAGE_BENCHMARK_PATH.write_text(json.dumps(leakage_benchmark, indent=2), encoding="utf-8")
    stability = create_improvement_figures(
        best_test_metrics,
        leakage_test_metrics,
        test,
        predictions[best_name],
    )
    figures = validate_figure_artifacts()

    baseline_row = comparison_frame[comparison_frame["model"] == "seasonal_lag_7_baseline"].iloc[0]
    best_info = {
        "model": best_name,
        "selection_rule": "Lowest validation RMSE among deployable ML candidates; test used for final reporting only.",
        "selected_features": selected_features,
        "test_metrics": best_test_metrics,
        "validation_metrics": validation_choices[best_name]["metrics"],
        "legacy_test_r2": LEGACY_TEST_R2,
        "test_r2_improvement": float(best_test_metrics["r2"] - LEGACY_TEST_R2),
        "prediction_interval_90_abs_error": interval_90,
        "baseline_test_rmse": float(baseline_row["test_rmse"]),
        "best_params": validation_choices[best_name]["params"],
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
        "available_feature_count": len(FEATURE_COLUMNS),
        "available_features": FEATURE_COLUMNS,
        "selected_model_features": selected_features,
        "excluded_from_deployable_features": ["date", TARGET_COLUMN, "demand_forecast"],
        "validation_choices": validation_choices,
        "best_model": best_name,
        "prediction_interval_90_abs_error": interval_90,
        "leakage_benchmark_deployable": False,
        "test_stability_months": stability["month"].tolist(),
        "figure_files": figures,
        "sklearn_version": sklearn.__version__,
    }
    TRAINING_SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {
        "best_model": best_info,
        "rows": summary["python_rows"],
        "leakage_benchmark": leakage_benchmark,
        "figures": figures,
    }


def _write_signal_audit(frame: pd.DataFrame) -> dict[str, object]:
    audit_columns = [
        "inventory_level",
        "units_ordered",
        "price",
        "discount",
        "competitor_pricing",
        "demand_forecast",
        "units_sold_lag_1",
        "units_sold_lag_7",
        "units_sold_rolling_mean_28",
    ]
    correlations = {
        column: float(frame[TARGET_COLUMN].corr(frame[column])) for column in audit_columns
    }
    sell_through = frame[TARGET_COLUMN] / frame["inventory_level"]
    forecast_error = frame["demand_forecast"] - frame[TARGET_COLUMN]
    audit = {
        "rows": int(len(frame)),
        "correlations": correlations,
        "units_sold_not_above_inventory_rate": float(
            (frame[TARGET_COLUMN] <= frame["inventory_level"]).mean()
        ),
        "sell_through_ratio": {
            "mean": float(sell_through.mean()),
            "std": float(sell_through.std()),
            "min": float(sell_through.min()),
            "max": float(sell_through.max()),
        },
        "demand_forecast_error": {
            "mean": float(forecast_error.mean()),
            "std": float(forecast_error.std()),
            "mae": float(forecast_error.abs().mean()),
        },
        "conclusion": (
            "Inventory level contains the only material deployable signal. Demand Forecast is a near-direct "
            "target proxy and remains excluded from the deployed model."
        ),
    }
    FEATURE_SIGNAL_AUDIT_PATH.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit


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


def _write_feature_schema(frame: pd.DataFrame, selected_features: list[str]) -> None:
    schema = {
        "feature_order": FEATURE_COLUMNS,
        "selected_model_features": selected_features,
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
