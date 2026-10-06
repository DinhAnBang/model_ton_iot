"""Data reader for the Phase-1 research pipeline."""

from __future__ import annotations

import os

from pathlib import Path

import pandas as pd

from ..common.errors import PipelineError
from ..common.paths import REPORT_DIR
from .models import SourceRecord


def group_for_file(platform: str, file_name: str) -> str:
    if platform == "Network":
        return "Network"
    if platform == "IoT":
        return f"IoT::{Path(file_name).stem}"
    if platform == "Windows":
        return f"Windows::{Path(file_name).stem}"
    lower = file_name.lower()
    if "disk" in lower:
        family = "disk"
    elif "memory" in lower:
        family = "memory"
    elif "process" in lower:
        family = "process"
    else:
        family = Path(file_name).stem.lower()
    return f"Linux::{family}"


def make_source_records(root: Path) -> list[SourceRecord]:
    inventory_path = REPORT_DIR / "dataset_inventory.csv"
    if not inventory_path.exists():
        raise PipelineError(f"Missing inventory: {inventory_path}")
    inventory = pd.read_csv(inventory_path, dtype=str, keep_default_na=False)
    required = {
        "platform", "file_name", "file_path", "rows", "columns",
        "timestamp_column", "timestamp_min", "timestamp_max",
        "duplicate_count",
    }
    missing = required - set(inventory.columns)
    if missing:
        raise PipelineError(f"Inventory is missing columns: {sorted(missing)}")

    records: list[SourceRecord] = []
    for row in inventory.to_dict("records"):
        relative = str(row["file_path"]).replace("/", os.sep).replace("\\", os.sep)
        path = root / relative
        if not path.exists():
            raise PipelineError(f"Inventory input does not exist: {path}")
        header = pd.read_csv(path, nrows=0)
        columns = [str(c) for c in header.columns]
        expected_columns = int(row["columns"])
        if len(columns) != expected_columns:
            raise PipelineError(
                f"{row['file_name']}: inventory says {expected_columns} columns, actual header has {len(columns)}"
            )
        rec = SourceRecord(
            platform=str(row["platform"]),
            file_name=str(row["file_name"]),
            file_path=path,
            relative_path=str(row["file_path"]).replace("\\", "/"),
            columns=columns,
            rows_expected=int(row["rows"]),
            timestamp_column=str(row["timestamp_column"]),
            timestamp_min=str(row["timestamp_min"]),
            timestamp_max=str(row["timestamp_max"]),
            duplicate_count=int(row["duplicate_count"] or 0),
            group_id=group_for_file(str(row["platform"]), str(row["file_name"])),
        )
        records.append(rec)
    return records
