"""Small helpers shared by the validation and preprocessing stages."""

from __future__ import annotations

import csv
import json
import logging
import math
import re
from pathlib import Path
from typing import Any

import numpy as np


def log(message: str) -> None:
    logging.getLogger("research_pipeline").info(message)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")
    return value.lower() or "unnamed"


def safe_json_number(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def relative_path(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def write_csv_rows(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def update_report_section(root: Path, marker: str, section: str) -> None:
    """Replace one phase section without deleting later phase results."""
    path = root / "research_pipeline" / "reports" / "audit_summary.md"
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    start = existing.find(marker)
    if start < 0:
        parts = [existing.strip(), section.strip()]
    else:
        next_heading = existing.find("\n## ", start + len(marker))
        if next_heading < 0:
            parts = [existing[:start].strip(), section.strip()]
        else:
            parts = [existing[:start].strip(), section.strip(), existing[next_heading + 1:].strip()]
    path.write_text("\n".join(part for part in parts if part) + "\n", encoding="utf-8")
