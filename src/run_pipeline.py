"""Run collection, preparation, training, evaluation, and artifact validation."""

from __future__ import annotations

import json

from .collect_data import collect_raw_data
from .config import ensure_directories
from .prepare_data import prepare_dataset
from .train import train_and_evaluate


def run_pipeline(force_download: bool = False) -> dict[str, object]:
    ensure_directories()
    source = collect_raw_data(force=force_download)
    cleaning = prepare_dataset()
    training = train_and_evaluate()
    return {"source": source, "cleaning": cleaning, "training": training}


if __name__ == "__main__":
    print(json.dumps(run_pipeline(), indent=2))
