#!/usr/bin/env python3
"""Phase-1 audit for the local ToN-IoT project.

The script only reads source data.  It scans Processed_datasets recursively,
uses chunked CSV reads, and writes reports below research_pipeline/reports.
Raw_datasets, Processed_datasets, Train_Test_datasets, and
Description_stats_datasets are never modified by this script.

Usage from the project root:
    python -m research_pipeline.src.validation.dataset_audit

Optional arguments are available so the same audit can be rerun from another
working directory, or with a smaller/larger chunk size:
    python -m research_pipeline.src.validation.dataset_audit --project-root . --chunksize 100000
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import re
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from ..common.config import PLATFORM_ORDER
from ..common.paths import PROJECT_ROOT

LOGGER = logging.getLogger(__name__)


UNIQUE_EXACT_LIMIT = 100_000
HLL_PRECISION = 14
RISK_CLASSES = [
    "SAFE_FEATURE",
    "REVIEW_FEATURE",
    "LEAKAGE_FEATURE",
    "IDENTIFIER",
    "LABEL",
    "TIMESTAMP",
]

LABEL_NAMES = {
    "label",
    "target",
    "class",
    "ground_truth",
    "ground truth",
    "y",
}
ATTACK_NAMES = {
    "type",
    "attack",
    "attack_type",
    "attack type",
    "category",
    "attack_category",
    "attack category",
    "class_name",
}
TIMESTAMP_NAMES = {
    "ts",
    "timestamp",
    "time_stamp",
    "datetime",
    "date_time",
    "event_time",
}


class HyperLogLog:
    """Small-memory approximate distinct counter for high-cardinality columns."""

    def __init__(self, precision: int = HLL_PRECISION) -> None:
        self.precision = precision
        self.register_count = 1 << precision
        self.registers = bytearray(self.register_count)

    def add_hash(self, value: int) -> None:
        value = int(value) & ((1 << 64) - 1)
        index = value & (self.register_count - 1)
        remaining = value >> self.precision
        rank = 64 - self.precision + 1 if remaining == 0 else 64 - self.precision - remaining.bit_length() + 1
        if rank > self.registers[index]:
            self.registers[index] = min(rank, 255)

    def add_hashes(self, values: Iterable[int]) -> None:
        for value in values:
            self.add_hash(int(value))

    def estimate(self) -> int:
        m = self.register_count
        inverse_sum = sum(2.0 ** (-rank) for rank in self.registers)
        alpha = 0.7213 / (1.0 + 1.079 / m)
        estimate = alpha * m * m / inverse_sum if inverse_sum else 0.0
        zero_registers = self.registers.count(0)
        if estimate <= 2.5 * m and zero_registers:
            estimate = m * math.log(m / zero_registers)
        return max(0, int(round(estimate)))


class UniqueTracker:
    """Exact distinct values up to a limit, then HLL without retaining values."""

    def __init__(self, exact_limit: int = UNIQUE_EXACT_LIMIT) -> None:
        self.exact_limit = exact_limit
        self.exact_values: set[Any] | None = set()
        self.exact_hashes: set[int] | None = set()
        self.hll: HyperLogLog | None = None

    def update(self, series: pd.Series) -> None:
        nonnull = series.dropna()
        if nonnull.empty:
            return
        hashes = pd.util.hash_pandas_object(nonnull, index=False).astype("uint64").tolist()
        if self.hll is not None:
            self.hll.add_hashes(hashes)
            return
        assert self.exact_values is not None and self.exact_hashes is not None
        try:
            self.exact_values.update(value_key(value) for value in pd.unique(nonnull))
        except TypeError:
            self.exact_values.update(clean_text(value) for value in pd.unique(nonnull))
        self.exact_hashes.update(int(value) for value in hashes)
        if len(self.exact_values) > self.exact_limit:
            self.hll = HyperLogLog()
            self.hll.add_hashes(self.exact_hashes)
            self.exact_values = None
            self.exact_hashes = None

    @property
    def method(self) -> str:
        return "HLL_APPROX" if self.hll is not None else "EXACT_SET"

    @property
    def count(self) -> int:
        if self.hll is not None:
            return self.hll.estimate()
        return len(self.exact_values or set())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit ToN-IoT processed CSV files")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Project root; defaults to the package project root.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=100_000,
        help="Rows per chunked CSV read (default: 100000).",
    )
    return parser.parse_args()


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def clean_text(value: Any, max_length: int = 200) -> str:
    if pd.isna(value):
        return "<NA>"
    if isinstance(value, float) and math.isnan(value):
        return "<NA>"
    text = str(value).strip()
    text = re.sub(r"\s+", " ", text)
    if len(text) > max_length:
        return text[: max_length - 1] + "…"
    return text


def normalized_name(value: Any) -> str:
    return re.sub(r"[\s\-]+", "_", str(value).strip().lower())


def infer_platform(path: Path) -> str:
    for part in path.parts:
        match = re.match(r"(?:Processed|Train_Test|Description_stats)_(.+?)_dataset$", part, flags=re.IGNORECASE)
        if match:
            raw = match.group(1).lower()
            return {"network": "Network", "iot": "IoT", "linux": "Linux", "windows": "Windows"}.get(
                raw, raw.title()
            )
    return path.parent.name


def rel_path(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def find_column(columns: Iterable[str], names: set[str]) -> str | None:
    by_normalized = {normalized_name(column): column for column in columns}
    for name in names:
        if name in by_normalized:
            return by_normalized[name]
    return None


def detect_columns(columns: list[str]) -> tuple[str | None, str | None, str | None]:
    label = find_column(columns, LABEL_NAMES)
    attack = find_column(columns, ATTACK_NAMES)

    timestamp: str | None = None
    if find_column(columns, {"date"}) and find_column(columns, {"time"}):
        timestamp = "date + time"
    else:
        timestamp = find_column(columns, TIMESTAMP_NAMES)
        if timestamp is None:
            for candidate in columns:
                n = normalized_name(candidate)
                if "timestamp" in n or n.endswith("_time") or n.endswith("_datetime"):
                    timestamp = candidate
                    break
    return timestamp, label, attack


def timestamp_format(timestamp_column: str | None, chunk: pd.DataFrame | None = None) -> str:
    if timestamp_column is None:
        return "not_detected"
    if timestamp_column == "date + time":
        return "date=%d-%b-%y + time=%H:%M:%S (timezone not encoded)"
    if chunk is not None and timestamp_column in chunk:
        series = chunk[timestamp_column].dropna()
        if not series.empty:
            value = series.iloc[0]
            if isinstance(value, (int, float)) or re.fullmatch(r"\d+(?:\.\d+)?", str(value).strip()):
                number = float(value)
                if number >= 10**12:
                    return "unix_milliseconds (timezone not encoded)"
                if number >= 10**9:
                    return "unix_seconds (timezone not encoded)"
    return "parseable datetime string (timezone not encoded)"


def parse_timestamp(chunk: pd.DataFrame, timestamp_column: str | None) -> pd.Series | None:
    if timestamp_column is None:
        return None
    try:
        if timestamp_column == "date + time":
            date_col = find_column(list(chunk.columns), {"date"})
            time_col = find_column(list(chunk.columns), {"time"})
            if not date_col or not time_col:
                return None
            raw = chunk[date_col].astype("string").str.strip() + " " + chunk[time_col].astype("string").str.strip()
            return pd.to_datetime(raw, errors="coerce", format="mixed")

        raw = chunk[timestamp_column]
        numeric = pd.to_numeric(raw, errors="coerce")
        numeric_nonnull = numeric.dropna()
        if not numeric_nonnull.empty:
            median = float(numeric_nonnull.abs().median())
            if median >= 10**12:
                return pd.to_datetime(numeric, errors="coerce", unit="ms")
            if median >= 10**9:
                return pd.to_datetime(numeric, errors="coerce", unit="s")
        return pd.to_datetime(raw, errors="coerce", format="mixed")
    except Exception:
        return None


def iso_timestamp(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    try:
        return pd.Timestamp(value).isoformat()
    except Exception:
        return str(value)


def value_key(value: Any) -> Any:
    if pd.isna(value):
        return None
    try:
        return value.item()
    except Exception:
        return value


def serialize_sample(values: list[Any]) -> str:
    return json_text([clean_text(value) for value in values[:3]])


def raw_row_digest(row: list[str]) -> bytes:
    """Create a length-delimited digest independent of pandas chunk dtypes."""
    digest = hashlib.blake2b(digest_size=16)
    for value in row:
        encoded = value.encode("utf-8", errors="replace")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.digest()


def next_csv_data_row(reader: csv.reader) -> list[str] | None:
    for row in reader:
        if not (len(row) == 1 and row[0] == ""):
            return row
    return None


def percentage(count: int, total: int) -> float:
    return round((count / total) * 100, 8) if total else 0.0


def classification_for_feature(feature: str, timestamp_column: str | None, label_column: str | None, attack_column: str | None) -> tuple[str, str]:
    normalized = normalized_name(feature)
    if label_column and feature == label_column:
        return "LABEL", "Ground-truth binary target; never use as an input feature."
    if attack_column and feature == attack_column:
        return "LEAKAGE_FEATURE", "Attack/type/category field exposes the target semantics."
    if feature == timestamp_column or normalized in {"date", "time", "ts", "timestamp", "datetime", "date_time"}:
        return "TIMESTAMP", "Needed for temporal ordering/alignment; split by time before fitting transforms."
    if feature in {"__file_name__", "__file_path__", "__parent_folder__"}:
        return "LEAKAGE_FEATURE", "Source metadata can encode attack scenario or label through file/folder naming."
    identifier_names = {
        "ip",
        "mac",
        "pid",
        "port",
        "process_id",
        "device_id",
        "flow_id",
        "conn_id",
        "session_id",
        "uid",
        "guid",
        "identifier",
        "id",
    }
    tokens = re.findall(r"[a-z0-9]+", normalized)
    is_address_or_port = (
        normalized in {"ip", "src_ip", "dst_ip", "mac", "src_mac", "dst_mac", "port", "src_port", "dst_port"}
        or normalized.endswith("_ip")
        or normalized.endswith("_mac")
        or normalized.endswith("_port")
    )
    is_process_or_id = normalized in identifier_names or any(
        normalized.endswith("_" + suffix) for suffix in {"pid", "process_id", "device_id", "flow_id", "conn_id", "session_id", "uid", "guid", "identifier"}
    ) or "id" in tokens
    if is_address_or_port or is_process_or_id or normalized in {"cmd", "command", "process", "device", "hostname"}:
        return "IDENTIFIER", "Identity/address/process key; preserve for semantics or graph construction, review for ML use."
    if normalized in {"type", "attack", "category", "scenario", "scenario_name"} or "attack" in normalized:
        return "LEAKAGE_FEATURE", "Field name indicates attack/scenario information."
    if normalized in {"proto", "protocol", "service", "conn_state", "status", "state", "poli", "ssl_version", "ssl_cipher"}:
        return "REVIEW_FEATURE", "Categorical protocol/service/state feature; encode using training data only and check semantic stability."
    return "SAFE_FEATURE", "Observed measurement or traffic statistic; still validate missingness and train-only preprocessing."


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    frame = pd.DataFrame(rows, columns=columns)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def audit_file(path: Path, root: Path, chunksize: int) -> dict[str, Any]:
    platform = infer_platform(path)
    file_name = path.name
    file_path = rel_path(path, root)
    header = pd.read_csv(path, nrows=0)
    columns = [str(c) for c in header.columns]
    timestamp_column, label_column, attack_column = detect_columns(columns)

    dtype_values: dict[str, set[str]] = {column: set() for column in columns}
    unique_values: dict[str, UniqueTracker] = {column: UniqueTracker() for column in columns}
    samples: dict[str, list[str]] = {column: [] for column in columns}
    missing_values: Counter[str] = Counter()
    placeholder_values: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    attack_counts: Counter[str] = Counter()
    row_hashes: set[bytes] = set()
    duplicate_count = 0
    placeholder_count = 0
    rows = 0
    timestamp_valid = 0
    timestamp_total = 0
    timestamp_min = None
    timestamp_max = None
    timestamp_last = None
    timestamp_unsorted = False
    timestamp_order_violations = 0
    min_positive_delta_seconds: float | None = None
    first_timestamp_format = "not_detected"

    reader = pd.read_csv(path, chunksize=chunksize, low_memory=False)
    raw_handle = path.open("r", encoding="utf-8-sig", errors="replace", newline="")
    raw_reader = csv.reader(raw_handle)
    next_csv_data_row(raw_reader)  # header
    for chunk in reader:
        chunk.columns = [str(c) for c in chunk.columns]
        chunk_rows = len(chunk)
        rows += chunk_rows

        null_counts = chunk.isna().sum()
        for column, value in null_counts.items():
            missing_values[column] += int(value)
        try:
            dash_counts = chunk.eq("-").sum()
            placeholder_count += int(dash_counts.sum())
            for column, value in dash_counts.items():
                placeholder_values[column] += int(value)
        except Exception:
            pass

        # Fingerprints are inserted/checked row-by-row in a persistent set for
        # this file. This counts duplicates both within the current chunk and
        # against all previous chunks; the first occurrence is never counted.
        # Raw CSV fields are used so dtype inference cannot change a fingerprint
        # at a chunk boundary. BLAKE2b-128 has a negligible residual collision
        # risk, which is documented in audit_summary.md.
        for _ in range(chunk_rows):
            raw_row = next_csv_data_row(raw_reader)
            if raw_row is None:
                break
            row_hash = raw_row_digest(raw_row)
            if row_hash in row_hashes:
                duplicate_count += 1
            row_hashes.add(row_hash)

        for column in columns:
            dtype_values[column].add(str(chunk[column].dtype))
            nonnull = chunk[column].dropna()
            if not nonnull.empty:
                unique_values[column].update(nonnull)
                for value in pd.unique(nonnull):
                    sample = clean_text(value)
                    if sample not in samples[column] and len(samples[column]) < 3:
                        samples[column].append(sample)

        if label_column and label_column in chunk:
            label_counts.update(clean_text(value) for value in chunk[label_column].dropna())
        if attack_column and attack_column in chunk:
            attack_counts.update(clean_text(value) for value in chunk[attack_column].dropna())

        parsed = parse_timestamp(chunk, timestamp_column)
        if parsed is not None:
            if first_timestamp_format == "not_detected":
                first_timestamp_format = timestamp_format(timestamp_column, chunk)
            valid = parsed.dropna()
            timestamp_total += len(parsed)
            timestamp_valid += len(valid)
            if not valid.empty:
                current_min = valid.min()
                current_max = valid.max()
                timestamp_min = current_min if timestamp_min is None or current_min < timestamp_min else timestamp_min
                timestamp_max = current_max if timestamp_max is None or current_max > timestamp_max else timestamp_max

                as_int = valid.astype("int64").to_numpy()
                if len(as_int) > 1:
                    sorted_valid = valid.sort_values().drop_duplicates()
                    deltas = sorted_valid.diff().dt.total_seconds().dropna()
                    deltas = deltas[deltas > 0]
                    if not deltas.empty:
                        current_delta = float(deltas.min())
                        if min_positive_delta_seconds is None or current_delta < min_positive_delta_seconds:
                            min_positive_delta_seconds = current_delta
                    order_diffs = (as_int[1:] < as_int[:-1]).sum()
                    timestamp_order_violations += int(order_diffs)
                if timestamp_last is not None and int(as_int[0]) < timestamp_last:
                    timestamp_order_violations += 1
                timestamp_last = int(as_int[-1])
                if timestamp_order_violations:
                    timestamp_unsorted = True

    raw_handle.close()

    label_rows = []
    for value, count in sorted(label_counts.items(), key=lambda item: item[0]):
        label_rows.append(
            {
                "platform": platform,
                "file_name": file_name,
                "label_column": label_column or "",
                "label_value": value,
                "count": count,
                "percentage": percentage(count, rows),
            }
        )
    attack_rows = []
    for value, count in sorted(attack_counts.items(), key=lambda item: item[0]):
        attack_rows.append(
            {
                "platform": platform,
                "file_name": file_name,
                "attack_column": attack_column or "",
                "attack_value": value,
                "count": count,
                "percentage": percentage(count, rows),
            }
        )

    schema_rows = []
    for column in columns:
        schema_rows.append(
            {
                "platform": platform,
                "file_name": file_name,
                "column_name": column,
                "dtype": "|".join(sorted(dtype_values[column])),
                "unique_count": unique_values[column].count,
                "unique_count_method": unique_values[column].method,
                "missing_count": missing_values[column],
                "placeholder_count": placeholder_values[column],
                "sample_values": serialize_sample(samples[column]),
            }
        )

    feature_rows = []
    for column in columns:
        classification, reason = classification_for_feature(column, timestamp_column, label_column, attack_column)
        feature_rows.append(
            {
                "platform": platform,
                "file": file_name,
                "feature": column,
                "classification": classification,
                "reason": reason,
            }
        )
    for synthetic_feature, reason in [
        ("__file_name__", "Processed file name can encode a scenario or attack source."),
        ("__file_path__", "Processed path/folder is source metadata and must not be an ML input."),
    ]:
        feature_rows.append(
            {
                "platform": platform,
                "file": file_name,
                "feature": synthetic_feature,
                "classification": "LEAKAGE_FEATURE",
                "reason": reason,
            }
        )

    timestamp_row = {
        "platform": platform,
        "file": file_name,
        "timestamp_column": timestamp_column or "",
        "timestamp_format": first_timestamp_format,
        "min": iso_timestamp(timestamp_min),
        "max": iso_timestamp(timestamp_max),
        "parse_success_rate": round(timestamp_valid / timestamp_total, 8) if timestamp_total else 0.0,
        "resolution_seconds": min_positive_delta_seconds if min_positive_delta_seconds is not None else "",
        "timezone": "not encoded / not established",
        "sorted_by_timestamp": not timestamp_unsorted if timestamp_column else "not_detected",
        "order_violations": timestamp_order_violations,
    }
    inventory = {
        "platform": platform,
        "file_name": file_name,
        "file_path": file_path,
        "file_size_mb": round(path.stat().st_size / (1024 * 1024), 6),
        "rows": rows,
        "columns": len(columns),
        "timestamp_column": timestamp_column or "",
        "timestamp_min": iso_timestamp(timestamp_min),
        "timestamp_max": iso_timestamp(timestamp_max),
        "label_column": label_column or "",
        "attack_column": attack_column or "",
        "missing_count": int(sum(missing_values.values())),
        "duplicate_count": int(duplicate_count),
        "placeholder_count": int(placeholder_count),
        "sorted_by_timestamp": not timestamp_unsorted if timestamp_column else "not_detected",
        "timestamp_order_violations": int(timestamp_order_violations),
        "error": "",
        "schema_rows": schema_rows,
        "label_rows": label_rows,
        "attack_rows": attack_rows,
        "feature_rows": feature_rows,
        "timestamp_row": timestamp_row,
        "columns_set": set(columns),
    }
    return inventory


def empty_error(path: Path, root: Path, exc: Exception) -> dict[str, Any]:
    return {
        "platform": infer_platform(path),
        "file_name": path.name,
        "file_path": rel_path(path, root),
        "error": f"{type(exc).__name__}: {exc}",
    }


def scan_train_test(root: Path) -> list[dict[str, Any]]:
    rows = []
    train_root = root / "Train_Test_datasets"
    if not train_root.exists():
        return rows
    for path in sorted(train_root.rglob("*.csv")):
        try:
            header = pd.read_csv(path, nrows=0)
            rows.append(
                {
                    "platform": infer_platform(path),
                    "file_name": path.name,
                    "file_path": rel_path(path, root),
                    "file_size_mb": round(path.stat().st_size / (1024 * 1024), 6),
                    "columns": len(header.columns),
                    "column_names": json_text([str(c) for c in header.columns]),
                    "error": "",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "platform": infer_platform(path),
                    "file_name": path.name,
                    "file_path": rel_path(path, root),
                    "file_size_mb": round(path.stat().st_size / (1024 * 1024), 6),
                    "columns": "",
                    "column_names": "",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return rows


def scan_description_files(root: Path) -> list[dict[str, Any]]:
    rows = []
    description_root = root / "Description_stats_datasets"
    if not description_root.exists():
        return rows
    for path in sorted(description_root.rglob("*")):
        if not path.is_file():
            continue
        rows.append(
            {
                "platform": infer_platform(path),
                "file_name": path.name,
                "file_path": rel_path(path, root),
                "extension": path.suffix.lower(),
                "file_size_mb": round(path.stat().st_size / (1024 * 1024), 6),
                "readable_by_audit": path.suffix.lower() in {".pdf", ".docx", ".xlsx"} and path.stat().st_size > 0,
            }
        )
    return rows


def build_schema_comparison(inventories: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_platform: dict[str, set[str]] = defaultdict(set)
    for inventory in inventories:
        by_platform[inventory["platform"]].update(inventory["columns_set"])
    all_columns = sorted(set().union(*by_platform.values()) if by_platform else set())
    rows = []
    for column in all_columns:
        row = {"column": column}
        for platform in PLATFORM_ORDER:
            row[platform] = column in by_platform.get(platform, set())
        rows.append(row)
    return rows


def format_count(value: Any) -> str:
    try:
        return f"{int(value):,}"
    except Exception:
        return str(value)


def platform_timestamp_summary(timestamp_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {}
    for platform in PLATFORM_ORDER:
        rows = [row for row in timestamp_rows if row["platform"] == platform and row["min"] and row["max"]]
        if not rows:
            result[platform] = {"min": "", "max": "", "files": 0, "columns": [], "formats": []}
            continue
        mins = [pd.Timestamp(row["min"]) for row in rows]
        maxs = [pd.Timestamp(row["max"]) for row in rows]
        result[platform] = {
            "min": iso_timestamp(min(mins)),
            "max": iso_timestamp(max(maxs)),
            "files": len(rows),
            "columns": sorted(set(row["timestamp_column"] for row in rows)),
            "formats": sorted(set(row["timestamp_format"] for row in rows)),
        }
    return result


def interval_overlap(summary: dict[str, dict[str, Any]], left: str, right: str) -> bool:
    try:
        left_min, left_max = pd.Timestamp(summary[left]["min"]), pd.Timestamp(summary[left]["max"])
        right_min, right_max = pd.Timestamp(summary[right]["min"]), pd.Timestamp(summary[right]["max"])
        return max(left_min, right_min) <= min(left_max, right_max)
    except Exception:
        return False


def build_audit_summary(
    root: Path,
    inventories: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    timestamp_rows: list[dict[str, Any]],
    description_rows: list[dict[str, Any]],
    train_test_rows: list[dict[str, Any]],
) -> str:
    counts = {platform: sum(item["rows"] for item in inventories if item["platform"] == platform) for platform in PLATFORM_ORDER}
    file_counts = {platform: sum(1 for item in inventories if item["platform"] == platform) for platform in PLATFORM_ORDER}
    schema_by_platform: dict[str, set[str]] = defaultdict(set)
    for item in inventories:
        schema_by_platform[item["platform"]].update(item["columns_set"])
    common_all = set.intersection(*(schema_by_platform[p] for p in PLATFORM_ORDER if schema_by_platform[p])) if schema_by_platform else set()
    timestamp_summary = platform_timestamp_summary(timestamp_rows)
    total_missing = sum(item["missing_count"] for item in inventories)
    total_duplicates = sum(item["duplicate_count"] for item in inventories)
    total_placeholders = sum(item["placeholder_count"] for item in inventories)
    risk_rows = [feature for item in inventories for feature in item["feature_rows"]]
    leakage = sorted(set(row["feature"] for row in risk_rows if row["classification"] == "LEAKAGE_FEATURE"))
    identifiers = sorted(set(row["feature"] for row in risk_rows if row["classification"] == "IDENTIFIER"))
    labels = sorted(set(row["feature"] for row in risk_rows if row["classification"] == "LABEL"))
    timestamps = sorted(set(row["feature"] for row in risk_rows if row["classification"] == "TIMESTAMP"))
    attack_values = sorted(set(row["attack_value"] for item in inventories for row in item["attack_rows"]))
    label_values = sorted(set(row["label_value"] for item in inventories for row in item["label_rows"]))
    hll_unique_columns = sum(
        1 for item in inventories for row in item["schema_rows"] if row["unique_count_method"] == "HLL_APPROX"
    )

    overlap_lines = []
    for index, left in enumerate(PLATFORM_ORDER):
        for right in PLATFORM_ORDER[index + 1 :]:
            overlap_lines.append(f"- {left} ↔ {right}: **{'có' if interval_overlap(timestamp_summary, left, right) else 'không thấy'}** overlap theo khoảng min/max.")

    schema_lines = []
    for platform in PLATFORM_ORDER:
        columns = sorted(schema_by_platform.get(platform, set()))
        schema_lines.append(f"- **{platform}** ({len(columns)} cột trên hợp của các file): `{', '.join(columns)}`")

    protocol_path = "research_pipeline/reports/data_protocol_proposal.md"
    error_note = "Không có CSV lỗi." if not errors else f"Có {len(errors)} CSV lỗi; xem `audit_errors.csv`."
    unreadable_descriptions = [row["file_name"] for row in description_rows if not row["readable_by_audit"]]
    description_note = (
        "Không có file mô tả rỗng/ngoài định dạng đã hỗ trợ."
        if not unreadable_descriptions
        else f"Không đọc được hoặc rỗng: {', '.join(unreadable_descriptions)}."
    )
    file_counts_line = ", ".join(f"{p} **{file_counts[p]}**" for p in PLATFORM_ORDER)
    inventory_table = "\n".join(f"| {p} | {file_counts[p]:,} | {counts[p]:,} |" for p in PLATFORM_ORDER)
    timestamp_table = "\n".join(
        f"| {p} | {', '.join(timestamp_summary[p]['columns']) or 'not_detected'} | {timestamp_summary[p]['min'] or 'NA'} | {timestamp_summary[p]['max'] or 'NA'} | {'; '.join(timestamp_summary[p]['formats']) or 'NA'} | not encoded / not established |"
        for p in PLATFORM_ORDER
    )
    missing_details = ", ".join(
        f"{item['platform']}/{item['file_name']} ({format_count(item['missing_count'])} cell; {item['missing_count'] / (item['rows'] * item['columns']) * 100:.2f}%)"
        for item in inventories
        if item["missing_count"]
    ) or "không có"
    duplicate_details = ", ".join(
        f"{item['platform']}/{item['file_name']} ({format_count(item['duplicate_count'])})"
        for item in inventories
        if item["duplicate_count"]
    ) or "không có"
    unsorted_details = ", ".join(
        f"{item['platform']}/{item['file_name']} ({item['timestamp_order_violations']} violations)"
        for item in inventories
        if item["sorted_by_timestamp"] is False
    ) or "không có"
    return f"""# ToN-IoT Phase-1 dataset audit

Generated: {datetime.now().astimezone().isoformat(timespec='seconds')}

## Scope and source roles

- `Processed_datasets`: nguồn chính được audit ở cấp CSV, không ghi đè.
- `Description_stats_datasets`: {len(description_rows)} file được inventory để đối chiếu ngữ nghĩa; {description_note}
- `Raw_datasets`: nguồn gốc/đầu vào chưa chuẩn hóa; không sửa.
- `Train_Test_datasets`: {len(train_test_rows)} CSV được đọc header để hiểu split tác giả; không dùng làm split chính.
- {error_note}

## 1. Inventory

- Tổng số processed CSV: **{len(inventories)}**.
- Số file theo platform: {file_counts_line}.
- Tổng số record: **{format_count(sum(counts.values()))}**.

| Platform | Processed CSV | Records |
|---|---:|---:|
{inventory_table}

## 2. Schema thực tế

{chr(10).join(schema_lines)}

- Schema chung của cả bốn platform: **{len(common_all)} cột**: `{', '.join(sorted(common_all))}`.
- Kết luận: **không có schema chung đầy đủ**; `schema_comparison.csv` cho biết cột chung theo từng platform. `label` và `type` là hai cột chung nổi bật; phần lớn feature đo lường là platform-specific.
- `unique_count` trong `schema_by_file.csv` dùng exact set cho cột có tối đa {UNIQUE_EXACT_LIMIT:,} distinct values; {hll_unique_columns} cột high-cardinality dùng HLL approximation để không giữ hàng triệu giá trị trong RAM.

## 2b. Đối chiếu tài liệu mô tả

- IoT documents describe `date`/`time`, sensor/service-profile measurements, `label` (`0` normal, `1` attack), and `type` as attack category; this matches the processed CSV structure. The seven IoT files are seven different service profiles, not one common feature table.
- Linux documents describe `atop` disk, process-scheduling, and memory measurements, with semantic identifiers `PID`/`CMD`; this matches the six processed files (two disk, two memory, two process).
- Network documents describe Bro connection/statistical/DNS/SSL/HTTP/violation fields and explain `conn_state`; `bro_log_vars` additionally defines `uid` as a unique connection ID. The processed Network CSVs contain 47 columns, while the feature-description table enumerates 46 fields, so `uid` must be treated as a documented semantic identifier rather than an ordinary measurement.
- Windows documents describe processor, process, network-interface, memory, and logical-disk counters; this matches the wide Windows schemas. However, the Windows feature document calls `ts` a Bro connection timestamp although the remaining columns are Windows performance counters. This semantic mismatch must be resolved before cross-platform timestamp alignment.
- The description/statistics documents use title-case attack names (`DoS`, `DDoS`, etc.), while processed CSVs use lower-case values (`dos`, `ddos`, etc.). Preserve raw values and apply an explicit versioned normalization map later.

## 3. Timestamp và khả năng đồng bộ

| Platform | Timestamp column | Min | Max | Parse formats | Timezone |
|---|---|---|---|---|---|
{timestamp_table}

Observed pairwise interval checks:
{chr(10).join(overlap_lines)}

Các file có timestamp dạng Unix seconds (`ts`) ở Linux/Network/Windows; IoT dùng `date` + `time` dạng ngày-tháng-năm và giờ-phút-giây. Timestamp không mang timezone trong CSV. Vì vậy overlap theo lịch **chỉ là bằng chứng cần thiết, chưa đủ để khẳng định cùng phiên chạy hoặc cùng host/device**. Alignment an toàn nhất ở Phase sau là giữ từng platform như modality riêng và join theo cửa sổ thời gian sau khi xác nhận timezone/scenario; không `concat` trực tiếp.

## 4. Labels và attack categories

- Label columns phát hiện: `{', '.join(labels) or 'không phát hiện'}`; các giá trị quan sát: `{', '.join(label_values) or 'không phát hiện'}`.
- Attack/category columns phát hiện: `{', '.join(sorted(set(item['attack_column'] for item in inventories if item['attack_column']))) or 'không phát hiện'}`.
- Attack/type values quan sát được: `{', '.join(attack_values) or 'không phát hiện'}`.
- Nhãn binary `label` được giữ riêng với `type`: `label` là target nhị phân, còn `type` là category/tên kịch bản và có nguy cơ leakage nếu đưa vào input.
- Casing/tên attack không nên chuẩn hóa tại bước audit; cần normalization map có version ở bước protocol.

## 5. Missing values, duplicates và file lỗi

- Null/NaN thực tế: **{format_count(total_missing)}** cell.
- File có null/NaN: {missing_details}. Đây là missing thực tế, không phải toàn bộ các ô `-` của Network.
- Placeholder `-` (đặc biệt trong Network log fields): **{format_count(total_placeholders)}** cell; đây không tự động đồng nghĩa với NaN và cần xử lý theo semantic từng feature.
- Exact duplicate row fingerprints (đếm các bản lặp sau lần xuất hiện đầu tiên): **{format_count(total_duplicates)}**. Fingerprint là digest BLAKE2b-128 có length-delimited trên các field raw của CSV; rủi ro collision còn lại rất nhỏ nhưng không bằng 0.
- File có duplicate toàn hàng: {duplicate_details}. Cần phân biệt với việc nhiều bản ghi có cùng timestamp hoặc cùng PID/flow.
- File không sort hoàn toàn theo timestamp: {unsorted_details}. Vì vậy không được mặc định thứ tự dòng là thứ tự thời gian.
- Không được xem `duplicate_count=0` là bằng chứng không có bản ghi lặp theo khóa nghiệp vụ; audit này chỉ kiểm tra duplicate toàn hàng.

## 6. Data leakage / feature risk

- LABEL: `{', '.join(labels) or 'không phát hiện'}`.
- TIMESTAMP: `{', '.join(timestamps) or 'không phát hiện'}`; giữ để ordering/alignment nhưng không fit transform trên toàn bộ dữ liệu.
- IDENTIFIER/semantic keys: `{', '.join(identifiers) or 'không phát hiện'}`.
- LEAKAGE_FEATURE: `{', '.join(leakage) or 'không phát hiện'}`; gồm `type`/attack-category nếu có và metadata nguồn file.
- Kết luận leakage: **có nguy cơ rõ ràng nếu dùng `label`, `type` hoặc metadata tên file/path làm feature**. Audit không tự drop cột; `feature_risk.csv` ghi lý do và protocol tách ML input khỏi semantic fields.

## 7. Có dùng được split theo thời gian không?

Có thể xây dựng **time-based split ở cấp scenario/file/window**, nhưng chưa nên dùng một split ngẫu nhiên trên toàn bộ CSV. Cần sort/kiểm tra timestamp trong từng source, xác định scenario boundaries, chọn cutoff theo thời gian và fit imputer/encoder/scaler trên train only. Các file có `sorted_by_timestamp=False` hoặc nhiều order violations phải được xử lý trước khi tạo window.

## 8. Đánh giá nguồn dữ liệu

- `Processed_datasets` phù hợp làm nguồn chính cho Phase 1 audit vì có schema dạng bảng, nhãn và timestamp đã lộ rõ; vẫn phải giữ provenance và không gọi đây là unified dataset.
- `Train_Test_datasets` chỉ nên dùng để đối chiếu cấu trúc/split tác giả. Không mặc định dùng làm split chính cho nghiên cứu mới vì chưa chứng minh được split đó tránh temporal/scenario leakage.
- Không thể merge trực tiếp 4 platform: khác schema, khác granularity/semantic, IoT có date-time còn các nguồn khác có Unix `ts`, feature identity/telemetry khác nhau, và timestamp không có timezone/scenario linkage đủ mạnh.

## 9. Hướng tích hợp đề xuất

1. Giữ bốn modality/source riêng; lập data dictionary và mapping label/category có version.
2. Chuẩn hóa timestamp thành một trường nội bộ (`event_time`) nhưng giữ raw timestamp và timezone assumption; xác minh overlap theo scenario trước khi liên kết.
3. Chia train/validation/test theo thời gian hoặc scenario trước khi fit preprocessing.
4. Tạo ML baseline view từ feature đo lường hợp lệ, loại target/attack/source metadata; giữ IP, port, protocol, service, PID/CMD/device identity và timestamp trong semantic/provenance view để dùng cho graph sau này.
5. Nếu có bằng chứng đồng bộ, liên kết bằng time window và khóa semantic phù hợp (host/device/flow/process), lưu `source_platform`, `source_file`, `scenario_id`, `window_id`; không fill feature vắng bằng 0 để giả làm unified table.

Chi tiết protocol đã được sinh tại `{protocol_path}`.

## 10. Những điều Phase 1 tuyệt đối chưa làm

- Chưa train Random Forest/XGBoost/LightGBM/neural network.
- Chưa build graph, snapshot, GCN/GraphSAGE/GAT/RGCN/HGT/GRU/Transformer.
- Chưa `pd.concat` bốn platform.
- Chưa sửa/xóa/điền dữ liệu trong `Raw_datasets`, `Processed_datasets`, `Train_Test_datasets` hoặc `Description_stats_datasets`.

## Reports

- `dataset_inventory.csv`
- `schema_by_file.csv`
- `label_distribution.csv`
- `attack_distribution.csv`
- `schema_comparison.csv`
- `timestamp_analysis.csv`
- `feature_risk.csv`
- `audit_errors.csv`
- `description_inventory.csv`
- `train_test_inventory.csv`
"""


def write_protocol(path: Path) -> None:
    path.write_text(
        """# Data protocol proposal for ToN-IoT Phase 1 → Phase 2

This is a protocol proposal grounded in the Phase-1 audit. It does not alter source data and does not train a model or build a graph.

## Source and provenance

Use `Processed_datasets` as the primary tabular input for validation. Keep immutable references to `Raw_datasets`, the processed relative path, original file name, platform, scenario/folder provenance, and audit run timestamp. Use `Train_Test_datasets` only as a reference until its split semantics are verified.

```text
RAW/PROCESSED INPUT
        ↓
VALIDATION
        ↓
CLEANING
        ↓
LABEL NORMALIZATION
        ↓
TIMESTAMP NORMALIZATION
        ↓
TIME ALIGNMENT
        ↓
TRAIN / VALIDATION / TEST SPLIT
        ↓
FIT PREPROCESSOR ON TRAIN ONLY
        ↓
TRANSFORM VAL/TEST
        ↓
BASELINE DATASET
```

## 1. Validation

- Re-read the inventory and fail fast on missing required columns for the selected modality.
- Validate row width, CSV parse errors, dtypes, allowed binary labels, attack/type values, and timestamp parse rate.
- Check duplicate whole rows and, separately, any domain key selected for a modality.
- Record missing values and the literal `-` placeholder separately. Do not assume `-` means numeric zero or NaN.
- Preserve raw columns until semantic review is complete.

## 2. Cleaning

- Trim whitespace in categorical/text fields using a reversible transformation log.
- Convert numeric fields only after checking invalid tokens and range plausibility from the feature descriptions.
- Treat platform-specific missing fields as structurally absent, not as zero measurements.
- Do not impute with values learned from validation/test. Fit imputation rules on train only.

## 3. Label normalization

- Keep `label` as the binary target (`0`/`1`) after validating its meaning in the descriptions and observed distribution.
- Keep raw `type`/attack category in provenance/semantic columns.
- Create a versioned mapping for spelling/case variants (`normal`, `Normal`, `benign`; `DDoS`/`ddos`; `MITM`; `XSS`; `Injection`; `Password`; `Scanning`; `Backdoor`; `Ransomware`, including source typos). Do not silently overwrite raw values.
- Never include binary label, attack category, scenario name, source filename, or source path in ML inputs.

## 4. Timestamp normalization and alignment

- Create `event_time` from IoT `date + time`; convert Unix `ts` from Linux/Network/Windows using an explicitly recorded unit and timezone assumption.
- Preserve `date`, `time`, `ts`, raw strings, and timezone metadata alongside `event_time`.
- Establish timezone and scenario correspondence from source documentation/metadata before claiming cross-platform synchronization. The current CSVs do not encode timezone.
- Use a source-specific resolution (observed second-level timestamps) and evaluate second/minute windows with sensitivity analysis. Do not claim a row-level one-to-one join from timestamp overlap alone.
- Link records only when a time window plus semantic key (device/host/flow/process/source) is justified; keep unmatched records and an alignment quality flag.

## 5. Split protocol

- Prefer scenario-based holdout when scenario provenance is available; otherwise use chronological cutoffs within each source/scenario.
- Ensure all rows from a scenario/window or entity group stay in one split when needed to prevent near-duplicate leakage.
- Fit all imputers, categorical encoders, scalers, vocabulary maps, and feature-selection decisions on train only.
- Validate that validation/test timestamps are later or scenario-disjoint as declared. Never shuffle across the cutoff to create the primary research split.

## 6. Two feature views

### A. ML baseline view

Include validated numeric telemetry and traffic measurements, plus reviewed categorical observables encoded from train only. Exclude `label`, `type`/attack/category, scenario/source filename/path, and any post-event or future-derived field. Treat IP/PID/device identity as a separate decision with a documented leakage/overfitting test rather than silently encoding it.

### B. Semantic/GNN view to preserve

Do not permanently delete `timestamp/event_time`, `src_ip`, `dst_ip`, `src_port`, `dst_port`, `proto/protocol`, `service`, `PID`, `CMD/process`, device identity, process/network identity, or source/scenario provenance. These fields may define nodes, edges, relations, temporal windows, or provenance for a later heterogeneous temporal graph. They must still be excluded from a baseline input when they directly identify a split, scenario, or target.

## 7. Recommended Phase-2 artifact boundary

Write new derived artifacts only under `research_pipeline/output`, with a schema/data dictionary and transformation manifest. Never overwrite source CSVs. Keep one modality per artifact until the temporal/semantic linkage audit passes; only then create an explicitly named aligned view rather than a generic “unified dataset”.
""",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.chunksize <= 0:
        raise SystemExit("--chunksize must be positive")
    root = args.project_root.resolve() if args.project_root else PROJECT_ROOT
    processed_root = root / "Processed_datasets"
    reports_root = root / "research_pipeline" / "reports"
    reports_root.mkdir(parents=True, exist_ok=True)
    (root / "research_pipeline" / "output").mkdir(parents=True, exist_ok=True)
    (root / "research_pipeline" / "config").mkdir(parents=True, exist_ok=True)

    csv_files = sorted(processed_root.rglob("*.csv")) if processed_root.exists() else []
    inventories: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    schema_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    attack_rows: list[dict[str, Any]] = []
    timestamp_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    started = time.time()

    LOGGER.info("Scanning %s processed CSV files with chunksize=%s ...", len(csv_files), f"{args.chunksize:,}")
    for index, path in enumerate(csv_files, start=1):
        LOGGER.info("[%s/%s] %s", index, len(csv_files), rel_path(path, root))
        try:
            inventory = audit_file(path, root, args.chunksize)
            inventories.append(inventory)
            schema_rows.extend(inventory["schema_rows"])
            label_rows.extend(inventory["label_rows"])
            attack_rows.extend(inventory["attack_rows"])
            timestamp_rows.append(inventory["timestamp_row"])
            feature_rows.extend(inventory["feature_rows"])
        except Exception as exc:
            LOGGER.error("ERROR: %s: %s", type(exc).__name__, exc)
            errors.append(empty_error(path, root, exc))

    inventory_rows = []
    for item in inventories:
        inventory_rows.append({key: value for key, value in item.items() if key not in {"schema_rows", "label_rows", "attack_rows", "feature_rows", "timestamp_row", "columns_set"}})

    description_rows = scan_description_files(root)
    train_test_rows = scan_train_test(root)
    write_csv(
        reports_root / "dataset_inventory.csv",
        inventory_rows,
        [
            "platform",
            "file_name",
            "file_path",
            "file_size_mb",
            "rows",
            "columns",
            "timestamp_column",
            "timestamp_min",
            "timestamp_max",
            "label_column",
            "attack_column",
            "missing_count",
            "duplicate_count",
            "placeholder_count",
            "sorted_by_timestamp",
            "timestamp_order_violations",
            "error",
        ],
    )
    write_csv(
        reports_root / "schema_by_file.csv",
        schema_rows,
        ["platform", "file_name", "column_name", "dtype", "unique_count", "unique_count_method", "missing_count", "placeholder_count", "sample_values"],
    )
    write_csv(
        reports_root / "label_distribution.csv",
        label_rows,
        ["platform", "file_name", "label_column", "label_value", "count", "percentage"],
    )
    write_csv(
        reports_root / "attack_distribution.csv",
        attack_rows,
        ["platform", "file_name", "attack_column", "attack_value", "count", "percentage"],
    )
    write_csv(reports_root / "schema_comparison.csv", build_schema_comparison(inventories), ["column", *PLATFORM_ORDER])
    write_csv(
        reports_root / "timestamp_analysis.csv",
        timestamp_rows,
        [
            "platform",
            "file",
            "timestamp_column",
            "timestamp_format",
            "min",
            "max",
            "parse_success_rate",
            "resolution_seconds",
            "timezone",
            "sorted_by_timestamp",
            "order_violations",
        ],
    )
    write_csv(reports_root / "feature_risk.csv", feature_rows, ["platform", "file", "feature", "classification", "reason"])
    write_csv(reports_root / "audit_errors.csv", errors, ["platform", "file_name", "file_path", "error"])
    write_csv(
        reports_root / "description_inventory.csv",
        description_rows,
        ["platform", "file_name", "file_path", "extension", "file_size_mb", "readable_by_audit"],
    )
    write_csv(
        reports_root / "train_test_inventory.csv",
        train_test_rows,
        ["platform", "file_name", "file_path", "file_size_mb", "columns", "column_names", "error"],
    )
    (reports_root / "audit_summary.md").write_text(
        build_audit_summary(root, inventories, errors, timestamp_rows, description_rows, train_test_rows), encoding="utf-8"
    )
    write_protocol(reports_root / "data_protocol_proposal.md")

    elapsed = time.time() - started
    LOGGER.info("Completed %s/%s files in %.1f minutes.", len(inventories), len(csv_files), elapsed / 60)
    LOGGER.info("Reports: %s", reports_root)
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
