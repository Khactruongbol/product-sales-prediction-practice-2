"""Helpers shared by the Streamlit app and prediction tests."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import BEST_MODEL_PATH, FEATURE_COLUMNS, FEATURE_SCHEMA_PATH
from .prepare_data import add_time_features


def load_artifacts(model_path: Path = BEST_MODEL_PATH, schema_path: Path = FEATURE_SCHEMA_PATH):
    if not model_path.exists() or not schema_path.exists():
        raise FileNotFoundError("Model artifacts are missing. Run: python -m src.run_pipeline")
    model = joblib.load(model_path)
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    return model, schema


def build_prediction_frame(values: dict[str, object]) -> pd.DataFrame:
    frame = pd.DataFrame([values])
    if "date" not in frame:
        raise ValueError("Prediction input must include date")
    frame = add_time_features(frame)
    missing = sorted(set(FEATURE_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Prediction input is missing fields: {missing}")
    return frame[FEATURE_COLUMNS]


def predict_units(model, values: dict[str, object]) -> float:
    prediction = float(model.predict(build_prediction_frame(values))[0])
    return float(max(0.0, np.nan_to_num(prediction, nan=0.0)))

