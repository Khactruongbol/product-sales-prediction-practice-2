"""Download and validate the raw public Kaggle dataset."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from .config import (
    DATASET_PAGE,
    DATASET_REF,
    DATASET_URL,
    RAW_DATA_PATH,
    RAW_REQUIRED_COLUMNS,
    SOURCE_METADATA_PATH,
    ensure_directories,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_raw_schema(path: Path = RAW_DATA_PATH) -> dict[str, object]:
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Raw CSV is missing or empty: {path}")
    frame = pd.read_csv(path)
    missing = sorted(set(RAW_REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Raw CSV is missing required columns: {missing}")
    if frame.empty:
        raise ValueError("Raw CSV contains no rows")
    return {"rows": int(len(frame)), "columns": int(len(frame.columns))}


def collect_raw_data(force: bool = False, timeout: int = 120) -> dict[str, object]:
    ensure_directories()
    if RAW_DATA_PATH.exists() and not force:
        stats = validate_raw_schema(RAW_DATA_PATH)
        if SOURCE_METADATA_PATH.exists():
            return json.loads(SOURCE_METADATA_PATH.read_text(encoding="utf-8"))
        metadata = _build_metadata(stats)
        SOURCE_METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return metadata

    try:
        response = requests.get(DATASET_URL, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Unable to download Kaggle dataset: {exc}") from exc

    try:
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            candidates = [name for name in archive.namelist() if name.lower().endswith(".csv")]
            expected = next(
                (name for name in candidates if Path(name).name == RAW_DATA_PATH.name),
                None,
            )
            if expected is None:
                raise ValueError(f"Expected {RAW_DATA_PATH.name} in archive; found {candidates}")
            with archive.open(expected) as source, RAW_DATA_PATH.open("wb") as destination:
                destination.write(source.read())
    except (zipfile.BadZipFile, OSError, ValueError) as exc:
        RAW_DATA_PATH.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded dataset archive is invalid: {exc}") from exc

    stats = validate_raw_schema(RAW_DATA_PATH)
    metadata = _build_metadata(stats)
    SOURCE_METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def _build_metadata(stats: dict[str, object]) -> dict[str, object]:
    return {
        "dataset_ref": DATASET_REF,
        "dataset_page": DATASET_PAGE,
        "download_url": DATASET_URL,
        "license": "CC0: Public Domain",
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "file": RAW_DATA_PATH.name,
        "bytes": RAW_DATA_PATH.stat().st_size,
        "sha256": sha256_file(RAW_DATA_PATH),
        **stats,
    }


if __name__ == "__main__":
    print(json.dumps(collect_raw_data(), indent=2))

