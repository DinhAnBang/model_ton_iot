"""Pure protocol invariants for the final ToN-IoT split decision.

This module does not read or rewrite the 31M-row source dataset.  It protects
the frozen protocol configuration and the compact, audit-backed artifacts.
"""

from __future__ import annotations

import csv
import inspect
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


TRAIN_END = datetime(2019, 4, 24, tzinfo=timezone.utc)
VALIDATION_END = datetime(2019, 4, 26, tzinfo=timezone.utc)
NETWORK_ENDPOINT_1 = datetime(2019, 4, 26, 2, 30, 34, tzinfo=timezone.utc)
NETWORK_ENDPOINT_2 = datetime(2019, 4, 27, 1, 25, 17, tzinfo=timezone.utc)
_TRAIN_END_LOCAL = TRAIN_END.replace(tzinfo=None)
_VALIDATION_END_LOCAL = VALIDATION_END.replace(tzinfo=None)


def _parse_time(value: object) -> datetime | None:
    """Parse an explicitly absolute ISO timestamp.

    A timezone-less value is intentionally rejected.  Treating it as UTC
    would turn an unverified source-local clock into a false cross-platform
    claim.
    """

    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "nat", "none", "<na>"}:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed.astimezone(timezone.utc)


def _parse_source_local_time(value: object) -> datetime | None:
    """Parse a timezone-less wall-clock value under an explicit local-clock rule."""

    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "nat", "none", "<na>"}:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is not None and parsed.utcoffset() is not None:
        return None
    return parsed.replace(tzinfo=None)


def assign_calendar_split(event_time: object) -> str:
    """Assign an explicitly absolute time using [start, end) boundaries."""

    parsed = _parse_time(event_time)
    if parsed is None:
        return "quarantine"
    if parsed < TRAIN_END:
        return "train"
    if parsed < VALIDATION_END:
        return "validation"
    return "test"


def assign_source_local_calendar_split(event_time: object) -> str:
    """Assign an explicit source-local wall-clock value.

    This is deliberately separate from :func:`assign_calendar_split`: it does
    not claim that the source-local clock is UTC or synchronized with another
    platform.
    """

    parsed = _parse_source_local_time(event_time)
    if parsed is None:
        return "quarantine"
    if parsed < _TRAIN_END_LOCAL:
        return "train"
    if parsed < _VALIDATION_END_LOCAL:
        return "validation"
    return "test"


def assign_network_endpoint_split(event_time: object) -> str:
    """Assign the two cross-fold Network endpoints with half-open ownership."""

    parsed = _parse_time(event_time)
    if parsed is None:
        return "quarantine"
    if parsed < NETWORK_ENDPOINT_1:
        return "train"
    if parsed < NETWORK_ENDPOINT_2:
        return "validation"
    return "test"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def artifact_root() -> Path:
    return Path(__file__).resolve().parents[2] / "reports" / "final_split_protocol"


def load_distribution(root: Path | None = None) -> list[dict[str, str]]:
    return _read_csv((root or artifact_root()) / "SPLIT_DISTRIBUTION.csv")


def load_coverage(root: Path | None = None) -> list[dict[str, str]]:
    return _read_csv((root or artifact_root()) / "CLASS_COVERAGE.csv")


def load_linux_schema_coverage(root: Path | None = None) -> list[dict[str, str]]:
    return _read_csv((root or artifact_root()) / "LINUX_SCHEMA_COVERAGE.csv")


def load_manifest(root: Path | None = None) -> list[dict[str, str]]:
    return _read_csv((root or artifact_root()) / "SPLIT_MANIFEST.csv")


def load_config(root: Path | None = None) -> dict[str, object]:
    path = (root or artifact_root()) / "split_protocol_config.json"
    return json.loads(path.read_text(encoding="utf-8"))


def validate_distribution(rows: Iterable[dict[str, str]]) -> list[str]:
    """Return integrity errors for the compact distribution artifact."""

    rows = list(rows)
    errors: list[str] = []
    observed_total = 0
    by_platform: dict[str, int] = {}
    for row in rows:
        count = int(row["rows"])
        normal = int(row["normal_count"])
        attack = int(row["attack_count"])
        if count != normal + attack:
            errors.append(f"{row['platform']}/{row['split']}: rows != normal + attack")
        if not math.isclose(float(row["normal_rate"]) + float(row["attack_rate"]), 1.0, abs_tol=2e-9):
            errors.append(f"{row['platform']}/{row['split']}: rates do not sum to one")
        observed_total += count
        by_platform[row["platform"]] = by_platform.get(row["platform"], 0) + count

    expected = {"Network": 22_339_021, "IoT": 3_505_271, "Linux": 5_855_329, "Windows": 64_342}
    if observed_total != sum(expected.values()):
        errors.append(f"all platforms: observed {observed_total} rows, expected {sum(expected.values())}")
    for platform, count in expected.items():
        if by_platform.get(platform) != count:
            errors.append(f"{platform}: observed {by_platform.get(platform, 0)} rows, expected {count}")
    return errors


def validate_manifest(rows: Iterable[dict[str, str]]) -> list[str]:
    """Reconcile the compact source-file manifest without reading row data."""

    rows = list(rows)
    errors: list[str] = []
    if len(rows) != 38:
        errors.append(f"manifest has {len(rows)} files, expected 38")
    total = sum(int(row["source_rows"]) for row in rows)
    if total != 31_763_963:
        errors.append(f"manifest has {total} source rows, expected 31763963")
    expected = {"Network": 22_339_021, "IoT": 3_505_271, "Linux": 5_855_329, "Windows": 64_342}
    by_platform: dict[str, int] = {}
    for row in rows:
        by_platform[row["platform"]] = by_platform.get(row["platform"], 0) + int(row["source_rows"])
        if row["research_role"] == "primary_known_scanning_conditional":
            errors.append(f"{row['platform']}/{row['source_file']}: stale aggregate primary role")
    for platform, expected_rows in expected.items():
        if by_platform.get(platform, 0) != expected_rows:
            errors.append(
                f"manifest {platform} has {by_platform.get(platform, 0)} rows, expected {expected_rows}"
            )
    return errors


def validate_primary_known_coverage(rows: Iterable[dict[str, str]]) -> list[str]:
    errors: list[str] = []
    for row in rows:
        role = row["protocol_a_role"]
        if role == "primary_known_scanning":
            errors.append(
                f"{row['platform']}/{row['attack_type']}: obsolete aggregate primary scanning claim"
            )
        if role == "schema_specific_known_scanning" and row["platform"] != "Linux":
            errors.append(f"{row['platform']}/{row['attack_type']}: schema-specific role is not Linux")
        if row["known_attack_in_test"] == "YES" and row["platform"] not in {"IoT", "Linux"}:
            errors.append(f"{row['platform']}/{row['attack_type']}: out-of-scope known Test claim")
        if (
            row["platform"] == "IoT"
            and row["attack_type"] == "scanning"
            and row["validation_attack_rows"] != "0"
        ):
            errors.append("IoT/scanning: Validation unexpectedly contains scanning rows")
    return errors


def validate_linux_schema_coverage(rows: Iterable[dict[str, str]]) -> list[str]:
    """Check that Linux eligibility is assessed per schema, never in aggregate."""

    expected = {
        "disk": "supporting_known_attack",
        "process": "main_known_attack",
        "memory": "diagnostic_no_known_attack",
    }
    rows = list(rows)
    errors: list[str] = []
    by_schema: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        if row["platform"] != "Linux":
            errors.append(f"unexpected platform in Linux schema artifact: {row['platform']}")
        by_schema.setdefault(row["schema"], []).append(row)
        if int(row["rows"]) != int(row["normal_count"]) + int(row["attack_count"]):
            errors.append(f"{row['schema']}/{row['split']}: rows != normal + attack")
        if int(row["scanning_count"]) > int(row["attack_count"]):
            errors.append(f"{row['schema']}/{row['split']}: scanning exceeds attack count")
    if set(by_schema) != set(expected):
        errors.append(f"Linux schemas observed {sorted(by_schema)}, expected {sorted(expected)}")
    for schema, role in expected.items():
        schema_rows = by_schema.get(schema, [])
        if {row["split"] for row in schema_rows} != {"train", "validation", "test"}:
            errors.append(f"{schema}: expected exactly train/validation/test rows")
        if any(row.get("decision_role") != role for row in schema_rows):
            errors.append(f"{schema}: inconsistent decision role")
        if schema in {"disk", "process"}:
            for row in schema_rows:
                if int(row["normal_count"]) <= 0 or int(row["scanning_count"]) <= 0:
                    errors.append(f"{schema}/{row['split']}: missing Normal or Scanning")
        else:
            if any(int(row["scanning_count"]) != 0 for row in schema_rows):
                errors.append("memory: Scanning must remain absent in every fold")
            if int(next((row["attack_count"] for row in schema_rows if row["split"] == "train"), "0")) != 0:
                errors.append("memory/train: attack rows must remain zero")
    return errors


def pipeline_contract_status(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Compare the frozen artifact with the currently implemented pipeline.

    This is a source-contract check, not a full export.  It intentionally
    reports the current Network and IoT timestamp limitations instead of
    silently treating them as solved.
    """

    from ..common.config import CORRECTIVE_TRAIN_END, CORRECTIVE_VALIDATION_END
    from ..pipeline import normalization, split_data

    config = config or load_config()
    calendar = config["calendar_boundaries"]
    expected_train = datetime.fromtimestamp(CORRECTIVE_TRAIN_END, tz=timezone.utc)
    expected_validation = datetime.fromtimestamp(CORRECTIVE_VALIDATION_END, tz=timezone.utc)
    train_text = expected_train.isoformat().replace("+00:00", "Z")
    validation_text = expected_validation.isoformat().replace("+00:00", "Z")
    split_source = inspect.getsource(split_data)
    iot_parser_source = inspect.getsource(normalization.parse_time_bound)
    row_time_source = inspect.getsource(normalization.event_time_numeric)
    return {
        "calendar_cutoffs_match": (
            calendar["train"].endswith(train_text + ")")
            and calendar["validation"].startswith("[" + train_text)
            and calendar["validation"].endswith(validation_text + ")")
            and calendar["test"].startswith("[" + validation_text)
        ),
        "quarantine_available_in_remediated_path": (
            "def remediated_split_codes" in split_source
            and "QUARANTINE_CODE" in split_source
        ),
        "network_half_open_endpoint_implemented": (
            "NETWORK_ENDPOINT_1" in split_source and "NETWORK_ENDPOINT_2" in split_source
        ),
        "iot_timezone_less_input_is_explicitly_source_local": "utc=True" not in iot_parser_source,
        "iot_timezone_semantics": (
            "UNVERIFIED: current parser represents IoT source-local date/time with utc=True; "
            "do not claim physical UTC"
        ),
        "non_iot_unix_row_timestamp_path": (
            "pd.to_numeric" in row_time_source and "unit=\"s\"" in row_time_source
        ),
        "non_iot_iso_row_timestamp_path": (
            "fromisoformat" in row_time_source or "format=\"ISO8601\"" in row_time_source
        ),
        "full_export_executed": bool(config["preprocessing"]["full_remediated_export_executed"]),
    }


def sample_contract_check(project_root: Path | None = None, rows_per_file: int = 128) -> dict[str, Any]:
    """Read a bounded prefix of each source file and exercise split routing only."""

    import pandas as pd

    from ..common.config import raw_type_mapping
    from ..pipeline.data_reader import make_source_records
    from ..pipeline.models import RunState
    from ..pipeline.normalization import validate_chunk
    from ..pipeline.split_data import assign_split_plan, remediated_split_codes

    root = project_root or Path(__file__).resolve().parents[3]
    records = make_source_records(root)
    assign_split_plan(records)
    mapping = raw_type_mapping()
    state = RunState()
    split_counts = {"train": 0, "validation": 0, "test": 0, "quarantine": 0}
    invalid_timestamps = 0
    rows_checked = 0
    for record in records:
        frame = pd.read_csv(record.file_path, nrows=rows_per_file, low_memory=False)
        _, _, _, times = validate_chunk(frame, record, mapping, state, validation_only=True)
        codes = remediated_split_codes(times, record)
        invalid_timestamps += int(times.isna().sum())
        rows_checked += len(frame)
        for code in codes:
            split = {0: "train", 1: "validation", 2: "test", -1: "quarantine"}[int(code)]
            split_counts[split] += 1
    return {
        "files_checked": len(records),
        "rows_per_file_limit": rows_per_file,
        "rows_checked": rows_checked,
        "invalid_timestamps_in_sample": invalid_timestamps,
        "split_counts": split_counts,
        "full_export_executed": False,
    }

