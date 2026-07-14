"""Metrics, plots, and artifact checks for trained regressors."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .config import FEATURE_COLUMNS, FIGURES_DIR, TARGET_COLUMN

sns.set_theme(style="whitegrid", context="notebook")


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    actual = np.asarray(y_true, dtype=float)
    predicted = np.maximum(0.0, np.asarray(y_pred, dtype=float))
    mse = mean_squared_error(actual, predicted)
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(actual, predicted)),
    }


def create_eda_figures(modeling: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    data = modeling.copy()
    data["date"] = pd.to_datetime(data["date"])

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(data[TARGET_COLUMN], bins=45, kde=True, ax=ax, color="#2563eb")
    ax.set(title="Distribution of Daily Units Sold", xlabel="Units sold", ylabel="Number of records")
    _save(fig, "target_distribution.png")

    weekly = data.set_index("date")[TARGET_COLUMN].resample("W").sum()
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(weekly.index, weekly.values, color="#0f766e", linewidth=1.5)
    ax.set(title="Total Units Sold by Week", xlabel="Week", ylabel="Units sold")
    _save(fig, "sales_over_time.png")

    category = data.groupby("category", as_index=False)[TARGET_COLUMN].mean().sort_values(TARGET_COLUMN)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(category, x=TARGET_COLUMN, y="category", ax=ax, color="#7c3aed")
    ax.set(title="Average Daily Units Sold by Category", xlabel="Average units sold", ylabel="Category")
    _save(fig, "sales_by_category.png")

    promo = data.groupby("holiday_promotion", as_index=False)[TARGET_COLUMN].mean()
    promo["holiday_promotion"] = promo["holiday_promotion"].map({"0": "No", "1": "Yes"}).fillna(
        promo["holiday_promotion"]
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(promo, x="holiday_promotion", y=TARGET_COLUMN, ax=ax, color="#ea580c")
    ax.set(title="Promotion/Holiday Effect on Average Sales", xlabel="Promotion or holiday", ylabel="Average units sold")
    _save(fig, "promotion_effect.png")


def create_model_figures(
    comparison: pd.DataFrame,
    test: pd.DataFrame,
    predictions: np.ndarray,
    model,
) -> None:
    ordered = comparison.sort_values("test_rmse", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.barplot(ordered, x="test_rmse", y="model", ax=ax, color="#0891b2")
    ax.set(title="Model Comparison on Temporal Test Set", xlabel="RMSE (lower is better)", ylabel="Model")
    _save(fig, "model_comparison_rmse.png")

    actual = test[TARGET_COLUMN].to_numpy()
    predicted = np.maximum(0.0, np.asarray(predictions))
    sample = np.linspace(0, len(test) - 1, min(5000, len(test)), dtype=int)
    bounds = [float(min(actual.min(), predicted.min())), float(max(actual.max(), predicted.max()))]
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(actual[sample], predicted[sample], alpha=0.25, s=14, color="#2563eb")
    ax.plot(bounds, bounds, "--", color="#dc2626", label="Perfect prediction")
    ax.set(title="Actual vs Predicted Units Sold", xlabel="Actual units sold", ylabel="Predicted units sold")
    ax.legend()
    _save(fig, "actual_vs_predicted.png")

    residuals = actual - predicted
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(residuals, bins=45, kde=True, ax=ax, color="#9333ea")
    ax.axvline(0, linestyle="--", color="#111827")
    ax.set(title="Residual Distribution", xlabel="Actual - predicted units", ylabel="Number of records")
    _save(fig, "residual_distribution.png")

    daily = pd.DataFrame({"date": pd.to_datetime(test["date"]), "residual": residuals}).groupby("date").mean()
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(daily.index, daily["residual"], color="#b45309")
    ax.axhline(0, linestyle="--", color="#111827")
    ax.set(title="Mean Residual over the Test Period", xlabel="Date", ylabel="Mean residual (units)")
    _save(fig, "residuals_over_time.png")

    importance_sample = test.sample(min(2000, len(test)), random_state=42)
    importance = permutation_importance(
        model,
        importance_sample[FEATURE_COLUMNS],
        importance_sample[TARGET_COLUMN],
        n_repeats=2,
        random_state=42,
        scoring="neg_root_mean_squared_error",
        n_jobs=1,
    )
    ranking = pd.DataFrame(
        {"feature": FEATURE_COLUMNS, "importance": importance.importances_mean}
    ).sort_values("importance", ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(ranking, x="importance", y="feature", ax=ax, color="#16a34a")
    ax.set(title="Permutation Importance of the Best Model", xlabel="Increase in score when feature is intact", ylabel="Feature")
    _save(fig, "feature_importance.png")


def validate_figure_artifacts() -> list[str]:
    expected = [
        "target_distribution.png",
        "sales_over_time.png",
        "sales_by_category.png",
        "promotion_effect.png",
        "model_comparison_rmse.png",
        "actual_vs_predicted.png",
        "residual_distribution.png",
        "residuals_over_time.png",
        "feature_importance.png",
    ]
    missing = [name for name in expected if not (FIGURES_DIR / name).exists() or (FIGURES_DIR / name).stat().st_size < 1000]
    if missing:
        raise AssertionError(f"Missing or invalid figure artifacts: {missing}")
    return expected


def _save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / name, dpi=160, bbox_inches="tight")
    plt.close(fig)

