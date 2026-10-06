#!/usr/bin/env python3
"""Phase-1.2 ground-truth validation for the local ToN-IoT project.

The script reads the official SecurityEvents ground-truth CSVs, compares them
with the processed platform files, and writes evidence reports.  It never
modifies source data, creates a unified table, preprocesses features, creates
a train/test split, or trains a model.

Run from the project root:
    python -m research_pipeline.src.validation.ground_truth_validation

The published request spells the source directory
``SecurityEvents_GroundTruth_datasets``.  The checked-in directory currently
spells it ``SecuityEvents_GroundTruth_datasets``; the script records that
discrepancy and uses the directory that actually exists.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from ..common.config import ATTACK_CATEGORIES, PLATFORM_ORDER, canonical_type, normalized_type_mapping
from ..common.paths import PROJECT_ROOT
from ..common.utils import relative_path as relpath, update_report_section, write_csv_rows as write_csv

LOGGER = logging.getLogger(__name__)


UTC = timezone.utc
PLATFORMS = PLATFORM_ORDER
ATTACK_TYPES = ATTACK_CATEGORIES
DOCUMENTED_ATTACKER_RANGE = "192.168.159.30-192.168.159.39"


def clean_field(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def parse_epoch(value: Any) -> datetime | None:
    raw = clean_field(value)
    if not raw:
        return None
    try:
        return datetime.fromtimestamp(float(raw), tz=UTC)
    except (TypeError, ValueError, OverflowError, OSError):
        return None


def parse_iot_local(data_value: Any, time_value: Any) -> datetime | None:
    date_text = clean_field(data_value)
    time_text = clean_field(time_value)
    if not date_text or not time_text:
        return None
    for fmt in ("%d-%b-%y %H:%M:%S", "%d-%b-%Y %H:%M:%S"):
        try:
            return datetime.strptime(f"{date_text} {time_text}", fmt)
        except ValueError:
            continue
    return None


def iso(value: datetime | None) -> str:
    if value is None:
        return "UNKNOWN"
    return value.isoformat()


def naive_utc(value: datetime | None) -> datetime | None:
    return value.replace(tzinfo=None) if value is not None else None


def seconds_between(left: datetime | None, right: datetime | None) -> float | None:
    if left is None or right is None:
        return None
    return (left - right).total_seconds()


def serialize_values(values: Iterable[str], limit: int = 50) -> str:
    ordered = sorted({clean_field(value) for value in values if clean_field(value)})
    if not ordered:
        return "UNKNOWN"
    if len(ordered) <= limit:
        return ";".join(ordered)
    return ";".join(ordered[:limit]) + f";(+{len(ordered) - limit} more)"


def platform_from_ground_truth(path: Path) -> str:
    name = path.parent.name.lower()
    if "iot" in name:
        return "IoT"
    if "linux" in name:
        return "Linux"
    if "network" in name:
        return "Network"
    if "windows" in name:
        return "Windows"
    return "UNKNOWN"


def normalize_headers(fieldnames: list[str] | None) -> list[str]:
    return [clean_field(field).lstrip("\ufeff") for field in (fieldnames or [])]


@dataclass
class WindowAccumulator:
    raw_type: str
    attack_type: str
    ordinal: int
    start_utc: datetime | None = None
    end_utc: datetime | None = None
    start_local: datetime | None = None
    end_local: datetime | None = None
    raw_start_ts: str = ""
    raw_end_ts: str = ""
    row_count: int = 0
    src_ips: set[str] = field(default_factory=set)
    dst_ips: set[str] = field(default_factory=set)
    pids: set[str] = field(default_factory=set)

    def add(self, row: dict[str, Any], platform: str) -> None:
        epoch = parse_epoch(row.get("ts"))
        local = parse_iot_local(row.get("data"), row.get("time")) if platform == "IoT" else None
        self.row_count += 1
        if epoch is not None:
            if self.start_utc is None or epoch < self.start_utc:
                self.start_utc = epoch
                self.raw_start_ts = clean_field(row.get("ts"))
            if self.end_utc is None or epoch > self.end_utc:
                self.end_utc = epoch
                self.raw_end_ts = clean_field(row.get("ts"))
        if local is not None:
            self.start_local = local if self.start_local is None else min(self.start_local, local)
            self.end_local = local if self.end_local is None else max(self.end_local, local)
        src = clean_field(row.get("src_ip"))
        dst = clean_field(row.get("dst_ip"))
        pid = clean_field(row.get("PID"))
        if src:
            self.src_ips.add(src)
        if dst:
            self.dst_ips.add(dst)
        if pid:
            self.pids.add(pid)


def window_row(root: Path, gt_root: Path, platform: str, path: Path, window: WindowAccumulator) -> dict[str, Any]:
    ground_truth_id = f"GT::{platform}::{path.stem}::w{window.ordinal}"
    source_file = relpath(root, path)
    local_offset = seconds_between(naive_utc(window.start_utc), window.start_local)
    raw_timestamp = (
        f"ts_min={window.raw_start_ts or 'UNKNOWN'}; ts_max={window.raw_end_ts or 'UNKNOWN'}"
        f"; data_time_min={window.start_local.isoformat() if window.start_local else 'UNKNOWN'}"
        f"; data_time_max={window.end_local.isoformat() if window.end_local else 'UNKNOWN'}"
    )
    if platform == "Network":
        confidence = "MEDIUM"
        role_note = "src_ip/dst_ip are observed fields; the ground-truth schema does not explicitly name attacker/victim roles."
    else:
        confidence = "HIGH"
        role_note = "type and timestamp rows are explicit in the official ground-truth file; no IP role is present."
    if platform == "IoT" and local_offset is not None:
        timezone = f"UTC epoch plus local data/time offset {local_offset:.0f} seconds"
        time_note = "IoT data/time is retained as local raw evidence; ts is normalized as epoch UTC."
    else:
        timezone = "UTC epoch normalization"
        time_note = "normalized_timestamp is derived from the explicit numeric ts field."
    return {
        "ground_truth_id": ground_truth_id,
        "attack_type": window.attack_type,
        "start_time": iso(window.start_utc),
        "end_time": iso(window.end_utc),
        "attacker_ip": serialize_values(window.src_ips) if platform == "Network" else "UNKNOWN",
        "victim": serialize_values(window.dst_ips) if platform == "Network" else "UNKNOWN",
        "platform": platform,
        "source_file": source_file,
        "raw_timestamp": raw_timestamp,
        "normalized_timestamp": f"{iso(window.start_utc)}..{iso(window.end_utc)}",
        "timezone": timezone,
        "evidence": f"Official ground-truth rows in {source_file}; explicit type={window.raw_type}; rows={window.row_count}. {role_note}",
        "confidence": confidence,
        "notes": f"Window method=source_file+raw_type min/max envelope; row-order type transitions are not treated as event boundaries. No shared scenario/session identifier is present in this file. {time_note}; observed PID candidates={serialize_values(window.pids, 20) if window.pids else 'UNKNOWN'}.",
        "_start_utc": window.start_utc,
        "_end_utc": window.end_utc,
        "_start_local": window.start_local,
        "_end_local": window.end_local,
        "_raw_type": window.raw_type,
        "_source_path": path,
        "_row_count": window.row_count,
        "_src_ips": set(window.src_ips),
        "_dst_ips": set(window.dst_ips),
    }


def scan_ground_truth(root: Path, gt_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    windows: list[dict[str, Any]] = []
    ip_stats: dict[str, Counter[str]] = {
        "src_ip": Counter(),
        "dst_ip": Counter(),
    }
    ip_files: dict[tuple[str, str], set[str]] = defaultdict(set)
    for path in sorted(gt_root.rglob("*.csv")):
        platform = platform_from_ground_truth(path)
        rows = 0
        parse_failures = 0
        timestamp_values: list[datetime] = []
        local_values: list[datetime] = []
        raw_ts_min: float | None = None
        raw_ts_max: float | None = None
        type_counts: Counter[str] = Counter()
        raw_type_counts: Counter[str] = Counter()
        file_src_ips: set[str] = set()
        file_dst_ips: set[str] = set()
        file_pids: set[str] = set()
        file_windows: list[dict[str, Any]] = []
        type_windows: dict[tuple[str, str], WindowAccumulator] = {}
        type_order: list[tuple[str, str]] = []
        try:
            with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
                reader = csv.DictReader(handle)
                fields = normalize_headers(reader.fieldnames)
                if reader.fieldnames != fields:
                    reader.fieldnames = fields
                for raw_row in reader:
                    row = {clean_field(key).lstrip("\ufeff"): value for key, value in raw_row.items() if key is not None}
                    rows += 1
                    raw_type = clean_field(row.get("type"))
                    attack = canonical_type(raw_type) or f"UNKNOWN::{raw_type or 'EMPTY'}"
                    raw_type_counts[raw_type or "EMPTY"] += 1
                    type_counts[attack] += 1
                    epoch = parse_epoch(row.get("ts"))
                    local = parse_iot_local(row.get("data"), row.get("time")) if platform == "IoT" else None
                    if epoch is None:
                        parse_failures += 1
                    else:
                        timestamp_values.append(epoch)
                        try:
                            raw_numeric = float(clean_field(row.get("ts")))
                            raw_ts_min = raw_numeric if raw_ts_min is None else min(raw_ts_min, raw_numeric)
                            raw_ts_max = raw_numeric if raw_ts_max is None else max(raw_ts_max, raw_numeric)
                        except (TypeError, ValueError):
                            pass
                    if local is not None:
                        local_values.append(local)
                    src = clean_field(row.get("src_ip"))
                    dst = clean_field(row.get("dst_ip"))
                    pid = clean_field(row.get("PID"))
                    if src:
                        ip_stats["src_ip"][src] += 1
                        file_src_ips.add(src)
                        ip_files[("src_ip", src)].add(relpath(root, path))
                    if dst:
                        ip_stats["dst_ip"][dst] += 1
                        file_dst_ips.add(dst)
                        ip_files[("dst_ip", dst)].add(relpath(root, path))
                    if pid:
                        file_pids.add(pid)
                    key = (raw_type, attack)
                    if key not in type_windows:
                        type_order.append(key)
                        type_windows[key] = WindowAccumulator(raw_type=raw_type, attack_type=attack, ordinal=len(type_order))
                    type_windows[key].add(row, platform)
                file_windows = [window_row(root, gt_root, platform, path, type_windows[key]) for key in type_order]
        except Exception as exc:
            inventory.append(
                {
                    "platform": platform,
                    "source_file": relpath(root, path),
                    "file_name": path.name,
                    "file_size_mb": round(path.stat().st_size / (1024 * 1024), 6),
                    "rows": rows,
                    "columns": ",".join(fields if 'fields' in locals() else []),
                    "timestamp_fields": "ts" if 'fields' in locals() and "ts" in fields else "UNKNOWN",
                    "timestamp_unit_observed": "UNKNOWN",
                    "timestamp_min_raw": "UNKNOWN",
                    "timestamp_max_raw": "UNKNOWN",
                    "timestamp_min_normalized": "UNKNOWN",
                    "timestamp_max_normalized": "UNKNOWN",
                    "type_field": "type" if 'fields' in locals() and "type" in fields else "UNKNOWN",
                    "ip_fields": ",".join(field for field in ("src_ip", "dst_ip") if 'fields' in locals() and field in fields) or "NONE",
                    "entity_fields": ",".join(field for field in ("PID", "device", "host") if 'fields' in locals() and field in fields) or "NONE",
                    "attack_types": ",".join(sorted(type_counts)),
                    "read_status": "ERROR",
                    "notes": repr(exc),
                }
            )
            continue
        windows.extend(file_windows)
        inventory.append(
            {
                "platform": platform,
                "source_file": relpath(root, path),
                "file_name": path.name,
                "file_size_mb": round(path.stat().st_size / (1024 * 1024), 6),
                "rows": rows,
                "columns": ",".join(fields),
                "timestamp_fields": ",".join(field for field in ("ts", "data", "time") if field in fields) or "UNKNOWN",
                "timestamp_unit_observed": "Unix epoch seconds from ts" if timestamp_values else "UNKNOWN",
                "timestamp_min_raw": str(raw_ts_min) if raw_ts_min is not None else "UNKNOWN",
                "timestamp_max_raw": str(raw_ts_max) if raw_ts_max is not None else "UNKNOWN",
                "timestamp_min_normalized": iso(min(timestamp_values)) if timestamp_values else "UNKNOWN",
                "timestamp_max_normalized": iso(max(timestamp_values)) if timestamp_values else "UNKNOWN",
                "type_field": "type" if "type" in fields else "UNKNOWN",
                "ip_fields": ",".join(field for field in ("src_ip", "dst_ip") if field in fields) or "NONE",
                "entity_fields": ",".join(field for field in ("PID", "device", "host") if field in fields) or "NONE",
                "attack_types": ",".join(sorted(type_counts)),
                "read_status": "OK",
                "notes": f"raw_type_counts={json.dumps(dict(raw_type_counts), ensure_ascii=False)}; local_time_rows={len(local_values)}; timestamp_parse_failures={parse_failures}; file_src_ips={len(file_src_ips)}; file_dst_ips={len(file_dst_ips)}; file_pids={len(file_pids)}",
            }
        )
        LOGGER.info("ground truth scanned: %s rows=%s windows=%s", relpath(root, path), f"{rows:,}", len(file_windows))
    return inventory, windows, {"ip_stats": ip_stats, "ip_files": ip_files}


def processed_paths(root: Path, platform: str) -> list[Path]:
    directories = {
        "IoT": root / "Processed_datasets" / "Processed_IoT_dataset",
        "Linux": root / "Processed_datasets" / "Processed_Linux_dataset",
        "Network": root / "Processed_datasets" / "Processed_Network_dataset",
        "Windows": root / "Processed_datasets" / "Processed_Windows_dataset",
    }
    return sorted(directories[platform].glob("*.csv"))


def network_processed_metadata(root: Path) -> list[tuple[Path, datetime, datetime, set[str]]]:
    inventory_path = root / "research_pipeline" / "reports" / "dataset_inventory.csv"
    attack_path = root / "research_pipeline" / "reports" / "attack_distribution.csv"
    attack_types: dict[str, set[str]] = defaultdict(set)
    with attack_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("platform") == "Network":
                attack = canonical_type(row.get("attack_value", ""))
                if attack:
                    attack_types[row["file_name"]].add(attack)
    metadata: list[tuple[Path, datetime, datetime, set[str]]] = []
    with inventory_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("platform") != "Network":
                continue
            try:
                start = pd.Timestamp(row["timestamp_min"]).to_pydatetime()
                end = pd.Timestamp(row["timestamp_max"]).to_pydatetime()
            except Exception:
                continue
            metadata.append((root / row["file_path"], start, end, attack_types.get(row["file_name"], set())))
    return metadata


def map_ground_truth_to_processed(root: Path, platform: str, source_file: str, window: dict[str, Any] | None = None) -> list[Path]:
    name = Path(source_file).name
    if platform == "IoT":
        suffix = name.removeprefix("GroundTruth_IoT_").removesuffix(".csv")
        candidate = root / "Processed_datasets" / "Processed_IoT_dataset" / f"IoT_{suffix}.csv"
        return [candidate] if candidate.exists() else []
    if platform == "Linux":
        suffix = name.removeprefix("GroundTruth_Linux_").removesuffix(".csv")
        return [path for path in processed_paths(root, platform) if suffix.lower() in path.stem.lower()]
    if platform == "Network":
        if window is not None and window.get("_start_utc") is not None and window.get("_end_utc") is not None:
            start = naive_utc(window["_start_utc"])
            end = naive_utc(window["_end_utc"])
            candidates = [
                path
                for path, file_start, file_end, attack_types in network_processed_metadata(root)
                if window["attack_type"] in attack_types and file_start <= end and start <= file_end
            ]
            if candidates:
                return sorted(candidates)
        suffix = name.removeprefix("GroundTruth_Network_").removesuffix(".csv")
        candidate = root / "Processed_datasets" / "Processed_Network_dataset" / f"Network_dataset_{suffix}.csv"
        return [candidate] if candidate.exists() else []
    if platform == "Windows":
        version = name.removeprefix("GroundTruth_Windows").removesuffix(".csv")
        candidate = root / "Processed_datasets" / "Processed_Windows_dataset" / f"windows{version}_dataset.csv"
        return [candidate] if candidate.exists() else []
    return []


def raw_provenance(root: Path, platform: str, attack_type: str) -> str:
    if attack_type.startswith("UNKNOWN"):
        return "UNKNOWN"
    raw_name = {
        "ransomware": "runsomware",
        "mitm": "MITM",
        "ddos": "DDoS",
        "dos": "DoS",
        "xss": "XSS",
        "injection": "injection",
        "password": "password",
        "scanning": "scanning",
        "backdoor": "backdoor",
    }.get(attack_type, attack_type)
    base = root / "Raw_datasets"
    candidates: list[Path] = []
    if platform == "IoT":
        candidates = [
            base / "telemtry_data" / "IoT_original_normal_attack" / f"IoT_{raw_name}_normal",
            base / "telemtry_data" / "IoT_filtered_normal_attack" / f"IoT_filtered_{raw_name}_normal",
        ]
    elif platform == "Network":
        candidates = [base / "network_data" / "Network_dataset_Bro" / "normal_attack_Bro" / f"normal_{raw_name}"]
    elif platform == "Linux":
        candidates = [
            base / "Linux_data" / "Original_Linux_dataset" / "normal_attack" / "atop_ubuntu14" / f"atop_{raw_name}",
            base / "Linux_data" / "Original_Linux_dataset" / "normal_attack" / "atop_ubuntu18" / f"atop_{raw_name}",
        ]
    elif platform == "Windows":
        candidates = [base / "windows_data" / "Windows_filtered-normal-attack" / f"Windows_filtered_{raw_name}_normal"]
    existing = [relpath(root, path) for path in candidates if path.exists()]
    return ";".join(existing) if existing else "UNKNOWN"


def canonical_series(series: pd.Series) -> pd.Series:
    normalized = series.astype("string").str.strip().str.lower()
    return normalized.map(normalized_type_mapping())


def parse_processed_time(chunk: pd.DataFrame, platform: str) -> pd.Series:
    if platform == "IoT":
        return pd.to_datetime(
            chunk["date"].astype("string").str.strip() + " " + chunk["time"].astype("string").str.strip(),
            format="%d-%b-%y %H:%M:%S",
            errors="coerce",
        )
    numeric = pd.to_numeric(chunk["ts"], errors="coerce")
    return pd.to_datetime(numeric, unit="s", errors="coerce").dt.tz_localize(None)


def init_window_stats() -> dict[str, Any]:
    return {
        "records": 0,
        "label_1": 0,
        "label_0": 0,
        "label_other": 0,
        "normal_type": 0,
        "canonical_match": 0,
        "canonical_mismatch": 0,
        "raw_types": Counter(),
    }


def add_range(current: dict[str, tuple[datetime, datetime]], key: str, values: pd.Series) -> None:
    valid = values.dropna()
    if valid.empty:
        return
    left = valid.min().to_pydatetime()
    right = valid.max().to_pydatetime()
    if key not in current:
        current[key] = (left, right)
    else:
        current[key] = (min(current[key][0], left), max(current[key][1], right))


def validate_processed_labels(
    root: Path,
    windows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_processed: dict[Path, list[dict[str, Any]]] = defaultdict(list)
    for window in windows:
        for path in map_ground_truth_to_processed(root, window["platform"], window["source_file"], window):
            window_copy = dict(window)
            if window["platform"] == "IoT" and window.get("_start_local") is not None:
                window_copy["_compare_start"] = window["_start_local"]
                window_copy["_compare_end"] = window["_end_local"]
                window_copy["_time_basis"] = "processed date/time vs ground-truth data/time local clock"
            else:
                window_copy["_compare_start"] = naive_utc(window["_start_utc"])
                window_copy["_compare_end"] = naive_utc(window["_end_utc"])
                window_copy["_time_basis"] = "processed ts vs ground-truth ts epoch normalized to UTC"
            by_processed[path].append(window_copy)

    results: list[dict[str, Any]] = []
    processed_ranges: dict[tuple[str, str], tuple[datetime, datetime]] = {}
    processed_label_counts: Counter[tuple[str, str]] = Counter()
    for path, file_windows in sorted(by_processed.items(), key=lambda item: str(item[0])):
        platform = next(window["platform"] for window in file_windows)
        stats = {window["ground_truth_id"]: init_window_stats() for window in file_windows}
        file_union_count = 0
        file_outside_label1 = 0
        file_outside_label0 = 0
        file_outside_attack_type = 0
        file_rows = 0
        header = pd.read_csv(path, nrows=0).columns.tolist()
        required = ["label", "type"] + (["date", "time"] if platform == "IoT" else ["ts"])
        if any(column not in header for column in required):
            for window in file_windows:
                results.append({
                    "platform": platform,
                    "processed_file": relpath(root, path),
                    "ground_truth_id": window["ground_truth_id"],
                    "attack_type": window["attack_type"],
                    "time_basis": window["_time_basis"],
                    "window_start": iso(window["_start_utc"]),
                    "window_end": iso(window["_end_utc"]),
                    "records_in_window": 0,
                    "label_1_in_window": 0,
                    "label_0_in_window": 0,
                    "normal_type_in_window": 0,
                    "canonical_type_match": 0,
                    "canonical_type_mismatch": 0,
                    "raw_types_in_window": "UNKNOWN",
                    "records_label_1_outside_windows": "UNKNOWN",
                    "records_label_0_outside_windows": "UNKNOWN",
                    "attack_type_outside_windows": "UNKNOWN",
                    "ground_truth_window_covered": "NO",
                    "confidence": "UNRESOLVED",
                    "notes": f"Required columns missing: {required}",
                })
            continue
        for chunk in pd.read_csv(path, usecols=required, chunksize=250_000, low_memory=False):
            file_rows += len(chunk)
            times = parse_processed_time(chunk, platform)
            canonical = canonical_series(chunk["type"])
            labels = pd.to_numeric(chunk["label"], errors="coerce")
            valid_time = times.notna()
            inside_any = pd.Series(False, index=chunk.index)
            for window in file_windows:
                window_mask = valid_time & (times >= window["_compare_start"]) & (times <= window["_compare_end"])
                if not window_mask.any():
                    continue
                inside_any |= window_mask
                item = stats[window["ground_truth_id"]]
                item["records"] += int(window_mask.sum())
                item["label_1"] += int(((labels == 1) & window_mask).sum())
                item["label_0"] += int(((labels == 0) & window_mask).sum())
                item["label_other"] += int((~labels.isin([0, 1]) & window_mask).sum())
                item["normal_type"] += int(((canonical == "normal") & window_mask).sum())
                item["canonical_match"] += int(((canonical == window["attack_type"]) & window_mask).sum())
                # A normal row inside a min/max envelope is reported separately
                # in normal_type_in_window; it is not an attack-type mismatch.
                item["canonical_mismatch"] += int((canonical.isin(ATTACK_TYPES) & (canonical != window["attack_type"]) & window_mask).sum())
                item["raw_types"].update(str(value) for value in chunk.loc[window_mask, "type"].dropna().unique())
            file_union_count += int(inside_any.sum())
            outside = valid_time & ~inside_any
            file_outside_label1 += int(((labels == 1) & outside).sum())
            file_outside_label0 += int(((labels == 0) & outside).sum())
            file_outside_attack_type += int((canonical.isin(ATTACK_TYPES) & outside).sum())
            for attack_type in canonical[labels == 1].dropna().unique():
                if attack_type not in ATTACK_TYPES:
                    continue
                values = times[(labels == 1) & (canonical == attack_type) & valid_time]
                add_range(processed_ranges, (platform, attack_type), values)  # type: ignore[arg-type]
                processed_label_counts[(platform, attack_type)] += int(len(values))
        for window in file_windows:
            item = stats[window["ground_truth_id"]]
            if item["records"] == 0:
                confidence = "UNRESOLVED"
            elif item["label_1"] > 0 and item["canonical_mismatch"] == 0:
                confidence = "HIGH" if platform != "Linux" else "MEDIUM"
            else:
                confidence = "LOW"
            results.append(
                {
                    "platform": platform,
                    "processed_file": relpath(root, path),
                    "ground_truth_id": window["ground_truth_id"],
                    "attack_type": window["attack_type"],
                    "time_basis": window["_time_basis"],
                    "window_start": iso(window["_start_utc"]),
                    "window_end": iso(window["_end_utc"]),
                    "records_in_window": item["records"],
                    "label_1_in_window": item["label_1"],
                    "label_0_in_window": item["label_0"],
                    "normal_type_in_window": item["normal_type"],
                    "canonical_type_match": item["canonical_match"],
                    "canonical_type_mismatch": item["canonical_mismatch"],
                    "raw_types_in_window": ";".join(sorted(item["raw_types"])) or "UNKNOWN",
                    "records_label_1_outside_windows": file_outside_label1,
                    "records_label_0_outside_windows": file_outside_label0,
                    "attack_type_outside_windows": file_outside_attack_type,
                    "ground_truth_window_covered": "YES" if item["records"] > 0 else "NO",
                    "confidence": confidence,
                    "notes": "Ground truth contains attack rows only; label_0 inside a window or label_1 outside windows is reported as evidence for review, not silently corrected.",
                }
            )
        LOGGER.info("processed labels validated: %s rows=%s windows=%s", relpath(root, path), f"{file_rows:,}", len(file_windows))
    return results, {"ranges": processed_ranges, "counts": processed_label_counts}


def build_time_offsets(windows: list[dict[str, Any]], processed_info: dict[str, Any]) -> list[dict[str, Any]]:
    gt_utc: dict[tuple[str, str], tuple[datetime, datetime]] = {}
    gt_local: dict[tuple[str, str], tuple[datetime, datetime]] = {}
    for window in windows:
        key = (window["platform"], window["attack_type"])
        if window["_start_utc"] is not None and window["_end_utc"] is not None:
            interval = (window["_start_utc"], window["_end_utc"])
            gt_utc[key] = interval if key not in gt_utc else (min(gt_utc[key][0], interval[0]), max(gt_utc[key][1], interval[1]))
        if window["_start_local"] is not None and window["_end_local"] is not None:
            interval_local = (window["_start_local"], window["_end_local"])
            gt_local[key] = interval_local if key not in gt_local else (min(gt_local[key][0], interval_local[0]), max(gt_local[key][1], interval_local[1]))
    processed_ranges = processed_info["ranges"]
    rows: list[dict[str, Any]] = []
    keys = sorted(set(gt_utc) | set(processed_ranges))
    for platform, attack_type in keys:
        processed = processed_ranges.get((platform, attack_type))
        if platform == "IoT" and (platform, attack_type) in gt_local:
            gt_local_interval = gt_local[(platform, attack_type)]
            local_offsets = [seconds_between(processed[0], gt_local_interval[0]) if processed else None, seconds_between(processed[1], gt_local_interval[1]) if processed else None]
            rows.append({
                "platform": platform,
                "attack_type": attack_type,
                "reference_ground_truth": f"{attack_type} local data/time field",
                "observed_offset_seconds": local_offsets[0] if local_offsets[0] is not None and local_offsets[1] is not None and abs(local_offsets[0] - local_offsets[1]) <= 1 else "UNKNOWN",
                "start_offset_seconds": local_offsets[0] if local_offsets[0] is not None else "UNKNOWN",
                "end_offset_seconds": local_offsets[1] if local_offsets[1] is not None else "UNKNOWN",
                "offset_consistent": "YES" if local_offsets[0] is not None and local_offsets[1] is not None and abs(local_offsets[0] - local_offsets[1]) <= 1 else "UNKNOWN",
                "timezone_interpretation": "Processed IoT date/time matches ground-truth data/time local clock; this does not prove UTC source clock.",
                "confidence": "HIGH" if local_offsets[0] is not None and local_offsets[1] is not None and abs(local_offsets[0]) <= 1 and abs(local_offsets[1]) <= 1 else "MEDIUM",
                "evidence": f"ground-truth local={gt_local_interval[0].isoformat()}..{gt_local_interval[1].isoformat()}; processed label=1 range={iso(processed[0]) if processed else 'UNKNOWN'}..{iso(processed[1]) if processed else 'UNKNOWN'}.",
            })
            if (platform, attack_type) in gt_utc:
                gt_interval = gt_utc[(platform, attack_type)]
                utc_offsets = [seconds_between(processed[0], naive_utc(gt_interval[0])) if processed else None, seconds_between(processed[1], naive_utc(gt_interval[1])) if processed else None]
                rows.append({
                    "platform": platform,
                    "attack_type": attack_type,
                    "reference_ground_truth": f"{attack_type} numeric ts epoch",
                    "observed_offset_seconds": utc_offsets[0] if utc_offsets[0] is not None and utc_offsets[1] is not None and abs(utc_offsets[0] - utc_offsets[1]) <= 1 else "UNKNOWN",
                    "start_offset_seconds": utc_offsets[0] if utc_offsets[0] is not None else "UNKNOWN",
                    "end_offset_seconds": utc_offsets[1] if utc_offsets[1] is not None else "UNKNOWN",
                    "offset_consistent": "YES" if utc_offsets[0] is not None and utc_offsets[1] is not None and abs(utc_offsets[0] - utc_offsets[1]) <= 1 else "UNKNOWN",
                    "timezone_interpretation": "Processed naive date/time is 7 hours behind numeric epoch in this ground truth comparison; retain local-vs-UTC ambiguity explicitly.",
                    "confidence": "MEDIUM" if utc_offsets[0] is not None and utc_offsets[1] is not None else "UNKNOWN",
                    "evidence": f"ground-truth UTC={iso(gt_interval[0])}..{iso(gt_interval[1])}; processed label=1 range={iso(processed[0]) if processed else 'UNKNOWN'}..{iso(processed[1]) if processed else 'UNKNOWN'}.",
                })
            continue
        gt_interval = gt_utc.get((platform, attack_type))
        offsets = [seconds_between(naive_utc(processed[0]), naive_utc(gt_interval[0])) if processed and gt_interval else None, seconds_between(naive_utc(processed[1]), naive_utc(gt_interval[1])) if processed and gt_interval else None]
        consistent = offsets[0] is not None and offsets[1] is not None and abs(offsets[0] - offsets[1]) <= 1
        rows.append({
            "platform": platform,
            "attack_type": attack_type,
            "reference_ground_truth": f"{attack_type} numeric ts epoch",
            "observed_offset_seconds": offsets[0] if consistent else "UNKNOWN",
            "start_offset_seconds": offsets[0] if offsets[0] is not None else "UNKNOWN",
            "end_offset_seconds": offsets[1] if offsets[1] is not None else "UNKNOWN",
            "offset_consistent": "YES" if consistent else ("NO" if offsets[0] is not None and offsets[1] is not None else "UNKNOWN"),
            "timezone_interpretation": "Numeric epoch comparison; this validates representation overlap, not independent host-clock correctness.",
            "confidence": "HIGH" if consistent and abs(offsets[0] or 0) <= 1 else ("MEDIUM" if offsets[0] is not None else "UNKNOWN"),
            "evidence": f"ground-truth UTC={iso(gt_interval[0]) if gt_interval else 'UNKNOWN'}..{iso(gt_interval[1]) if gt_interval else 'UNKNOWN'}; processed label=1 range={iso(processed[0]) if processed else 'UNKNOWN'}..{iso(processed[1]) if processed else 'UNKNOWN'}.",
        })
    return rows


def network_processed_ip_stats(root: Path) -> dict[str, Counter[str]]:
    stats = {"src_ip": Counter(), "dst_ip": Counter()}
    for path in processed_paths(root, "Network"):
        for chunk in pd.read_csv(path, usecols=["src_ip", "dst_ip"], chunksize=250_000, low_memory=False):
            for column in stats:
                stats[column].update(str(value).strip() for value in chunk[column].dropna() if str(value).strip())
        LOGGER.info("processed IPs scanned: %s", relpath(root, path))
    return stats


def load_windows_report(root: Path) -> list[dict[str, Any]]:
    """Reload the public window report after an interrupted long scan."""
    path = root / "research_pipeline" / "reports" / "ground_truth_attack_windows.csv"
    windows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            start_utc = pd.Timestamp(row["start_time"]).to_pydatetime()
            end_utc = pd.Timestamp(row["end_time"]).to_pydatetime()
            raw_timestamp = row.get("raw_timestamp", "")
            local_min = re.search(r"data_time_min=([^;]+)", raw_timestamp)
            local_max = re.search(r"data_time_max=([^;]+)", raw_timestamp)
            start_local = datetime.fromisoformat(local_min.group(1)) if local_min and local_min.group(1) != "UNKNOWN" else None
            end_local = datetime.fromisoformat(local_max.group(1)) if local_max and local_max.group(1) != "UNKNOWN" else None
            row_count_match = re.search(r"rows=(\d+)", row.get("evidence", ""))
            src_ips = {value for value in row.get("attacker_ip", "").split(";") if value and value != "UNKNOWN" and not value.startswith("(+")}
            dst_ips = {value for value in row.get("victim", "").split(";") if value and value != "UNKNOWN" and not value.startswith("(+")}
            row.update(
                {
                    "_start_utc": start_utc,
                    "_end_utc": end_utc,
                    "_start_local": start_local,
                    "_end_local": end_local,
                    "_raw_type": row["attack_type"],
                    "_source_path": root / row["source_file"],
                    "_row_count": int(row_count_match.group(1)) if row_count_match else 0,
                    "_src_ips": src_ips,
                    "_dst_ips": dst_ips,
                }
            )
            windows.append(row)
    return windows


def load_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def processed_label_ranges_only(root: Path) -> dict[str, Any]:
    """Rebuild only label=1 ranges for resume mode; no GT raw rescan is needed."""
    ranges: dict[tuple[str, str], tuple[datetime, datetime]] = {}
    counts: Counter[tuple[str, str]] = Counter()
    for platform in PLATFORMS:
        for path in processed_paths(root, platform):
            header = pd.read_csv(path, nrows=0).columns.tolist()
            required = ["label", "type"] + (["date", "time"] if platform == "IoT" else ["ts"])
            if any(column not in header for column in required):
                continue
            for chunk in pd.read_csv(path, usecols=required, chunksize=250_000, low_memory=False):
                times = parse_processed_time(chunk, platform)
                labels = pd.to_numeric(chunk["label"], errors="coerce")
                canonical = canonical_series(chunk["type"])
                for attack_type in canonical[labels == 1].dropna().unique():
                    if attack_type not in ATTACK_TYPES:
                        continue
                    values = times[(labels == 1) & (canonical == attack_type)].dropna()
                    if values.empty:
                        continue
                    add_range(ranges, (platform, attack_type), values)  # type: ignore[arg-type]
                    counts[(platform, attack_type)] += len(values)
            LOGGER.info("processed label ranges rescanned: %s", relpath(root, path))
    return {"ranges": ranges, "counts": counts}


def ground_truth_network_ip_stats(gt_root: Path) -> dict[str, Counter[str]]:
    stats = {"src_ip": Counter(), "dst_ip": Counter()}
    for path in sorted((gt_root / "SecurityEvents_Network_datasets").glob("*.csv")):
        for chunk in pd.read_csv(path, usecols=["src_ip", "dst_ip"], chunksize=250_000, low_memory=False):
            for column in stats:
                stats[column].update(str(value).strip() for value in chunk[column].dropna() if str(value).strip())
        LOGGER.info("ground-truth IPs rescanned: %s", path.name)
    return stats


def documented_range_ips() -> set[str]:
    match = re.fullmatch(r"(\d+\.\d+\.\d+)\.(\d+)-(\d+\.\d+\.\d+)\.(\d+)", DOCUMENTED_ATTACKER_RANGE)
    if not match or match.group(1) != match.group(3):
        return set()
    prefix = match.group(1)
    start, end = int(match.group(2)), int(match.group(4))
    return {f"{prefix}.{value}" for value in range(start, end + 1)}


def build_ip_validation(root: Path, gt_ip_info: dict[str, Any], processed_ips: dict[str, Counter[str]]) -> list[dict[str, Any]]:
    gt_stats: dict[str, Counter[str]] = gt_ip_info["ip_stats"]
    candidates = documented_range_ips()
    all_ips = set(candidates) | set(gt_stats["src_ip"]) | set(gt_stats["dst_ip"]) | set(processed_ips["src_ip"]) | set(processed_ips["dst_ip"])
    rows: list[dict[str, Any]] = []
    for ip in sorted(all_ips):
        gt_src = gt_stats["src_ip"].get(ip, 0)
        gt_dst = gt_stats["dst_ip"].get(ip, 0)
        processed_src = processed_ips["src_ip"].get(ip, 0)
        processed_dst = processed_ips["dst_ip"].get(ip, 0)
        observed_gt = gt_src + gt_dst > 0
        observed_processed = processed_src + processed_dst > 0
        if gt_src and processed_src:
            validation = "CONFIRMED_IN_GROUND_TRUTH_AND_NETWORK_SRC"
            confidence = "HIGH"
        elif gt_dst and processed_dst:
            validation = "CONFIRMED_IN_GROUND_TRUTH_AND_NETWORK_DST"
            confidence = "MEDIUM"
        elif observed_gt and observed_processed:
            validation = "OBSERVED_BUT_ROLE_OR_DIRECTION_DIFFERS"
            confidence = "LOW"
        elif observed_gt:
            validation = "GROUND_TRUTH_ONLY"
            confidence = "LOW"
        else:
            validation = "NETWORK_ONLY_OR_DOCUMENTED_CANDIDATE_ONLY"
            confidence = "LOW"
        rows.append({
            "ip": ip,
            "documented_candidate_range": DOCUMENTED_ATTACKER_RANGE,
            "documented_candidate_match": "YES" if ip in candidates else "NO",
            "ground_truth_src_ip_count": gt_src,
            "ground_truth_dst_ip_count": gt_dst,
            "processed_network_src_ip_count": processed_src,
            "processed_network_dst_ip_count": processed_dst,
            "ground_truth_role_evidence": "src_ip candidate" if gt_src and not gt_dst else ("dst_ip candidate" if gt_dst and not gt_src else ("src_ip and dst_ip" if gt_src and gt_dst else "NONE")),
            "observed_in_ground_truth": "YES" if observed_gt else "NO",
            "observed_in_processed_network": "YES" if observed_processed else "NO",
            "validation_status": validation,
            "confidence": confidence,
            "notes": "The documented range is treated only as a comparison candidate, not as a hard-coded labeling rule. Ground-truth schema names src_ip/dst_ip but does not explicitly declare attacker/victim role.",
        })
    return rows


def scenario_mapping_v2(root: Path, windows: list[dict[str, Any]], label_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    evidence_by_gt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in label_rows:
        evidence_by_gt[row["ground_truth_id"]].append(row)
    output: list[dict[str, Any]] = []
    for window in windows:
        gt_id = window["ground_truth_id"]
        linked = map_ground_truth_to_processed(root, window["platform"], window["source_file"], window)
        linked_names = ";".join(relpath(root, path) for path in linked) or "UNKNOWN"
        validation = evidence_by_gt.get(gt_id, [])
        total_records = sum(int(row["records_in_window"]) for row in validation)
        total_label1 = sum(int(row["label_1_in_window"]) for row in validation)
        total_mismatch = sum(int(row["canonical_type_mismatch"]) for row in validation)
        if window["platform"] == "Linux":
            confidence = "LOW" if total_records > 0 else "UNRESOLVED"
        elif total_records > 0 and total_label1 > 0 and total_mismatch == 0:
            confidence = "HIGH"
        elif total_records > 0:
            confidence = "MEDIUM"
        else:
            confidence = "UNRESOLVED"
        output.append({
            "ground_truth_id": gt_id,
            "scenario_id": f"SCENARIO::{gt_id}",
            "attack_type": window["attack_type"],
            "platform": window["platform"],
            "ground_truth_source_file": window["source_file"],
            "processed_source_file": linked_names,
            "raw_provenance": raw_provenance(root, window["platform"], window["attack_type"]),
            "start_time": iso(window["_start_utc"]),
            "end_time": iso(window["_end_utc"]),
            "attacker_ip": window["attacker_ip"],
            "victim": window["victim"],
            "mapping_confidence": confidence,
            "evidence": f"Ground-truth ID is derived from the official file/window; linked processed rows={total_records}, label_1={total_label1}, type mismatches={total_mismatch}.",
            "notes": "The scenario ID is ground-truth-backed and platform-specific. It is not treated as a shared cross-platform event ID because the official files do not expose a common session/run identifier.",
        })
    return output


def intervals_overlap(left: tuple[datetime, datetime], right: tuple[datetime, datetime]) -> bool:
    return left[0] <= right[1] and right[0] <= left[1]


def alignment_clusters(windows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for window in windows:
        if window["attack_type"] in ATTACK_TYPES and window["_start_utc"] is not None and window["_end_utc"] is not None:
            by_type[window["attack_type"]].append(window)
    clusters: list[list[dict[str, Any]]] = []
    for attack_type, items in sorted(by_type.items()):
        remaining = list(items)
        type_components: list[list[dict[str, Any]]] = []
        while remaining:
            component = [remaining.pop(0)]
            changed = True
            while changed:
                changed = False
                keep: list[dict[str, Any]] = []
                component_intervals = [(item["_start_utc"], item["_end_utc"]) for item in component]
                for item in remaining:
                    item_interval = (item["_start_utc"], item["_end_utc"])
                    if any(intervals_overlap(item_interval, interval) for interval in component_intervals):
                        component.append(item)
                        changed = True
                        component_intervals.append(item_interval)
                    else:
                        keep.append(item)
                remaining = keep
            type_components.append(component)
        # If an attack type has multiple disjoint components but no overlap
        # between different platforms at all, collapse them into one explicit
        # same-type/no-overlap row instead of hiding the finding as several
        # single-platform UNRESOLVED rows.
        cross_component_overlap = any(
            intervals_overlap((left["_start_utc"], left["_end_utc"]), (right["_start_utc"], right["_end_utc"]))
            and left["platform"] != right["platform"]
            for left_component_index, left_component in enumerate(type_components)
            for right_component in type_components[left_component_index + 1 :]
            for left in left_component
            for right in right_component
        )
        if len(type_components) > 1 and not cross_component_overlap:
            clusters.append([item for component in type_components for item in component])
        else:
            clusters.extend(type_components)
    return clusters


def build_alignment_v2(windows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, component in enumerate(alignment_clusters(windows), start=1):
        attack_type = component[0]["attack_type"]
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for window in component:
            grouped[window["platform"]].append(window)
        platform_intervals: dict[str, tuple[datetime, datetime]] = {}
        platform_windows: dict[str, str] = {}
        for platform in PLATFORMS:
            entries = grouped.get(platform, [])
            if entries:
                platform_intervals[platform] = (min(item["_start_utc"] for item in entries), max(item["_end_utc"] for item in entries))
                platform_windows[platform] = ";".join(f"{item['ground_truth_id']}[{iso(item['_start_utc'])}..{iso(item['_end_utc'])}]" for item in entries)
            else:
                platform_windows[platform] = "UNKNOWN"
        present_platforms = [platform for platform in PLATFORMS if platform in platform_intervals]
        if len(present_platforms) >= 2:
            common_start = max(platform_intervals[platform][0] for platform in present_platforms)
            common_end = min(platform_intervals[platform][1] for platform in present_platforms)
            common_overlap = max(0.0, (common_end - common_start).total_seconds())
            pair_overlap = any(
                intervals_overlap((left["_start_utc"], left["_end_utc"]), (right["_start_utc"], right["_end_utc"]))
                for left_index, left in enumerate(component)
                for right in component[left_index + 1 :]
                if left["platform"] != right["platform"]
            )
            status = "TIME_OVERLAP_ONLY" if pair_overlap else "NO_OVERLAP"
            common_start_text = iso(common_start) if common_overlap > 0 else "NONE"
            common_end_text = iso(common_end) if common_overlap > 0 else "NONE"
            confidence = "LOW"
        else:
            common_start_text = common_end_text = "UNKNOWN"
            common_overlap = "UNKNOWN"
            status = "UNRESOLVED"
            confidence = "UNKNOWN"
        rows.append({
            "ground_truth_id": ";".join(item["ground_truth_id"] for item in component),
            "scenario_id": f"GT_CLUSTER::{attack_type}::{index}",
            "attack_type": attack_type,
            "network_present": "yes" if "Network" in platform_intervals else "no",
            "iot_present": "yes" if "IoT" in platform_intervals else "no",
            "linux_present": "yes" if "Linux" in platform_intervals else "no",
            "windows_present": "yes" if "Windows" in platform_intervals else "no",
            "network_window": platform_windows["Network"],
            "iot_window": platform_windows["IoT"],
            "linux_window": platform_windows["Linux"],
            "windows_window": platform_windows["Windows"],
            "common_overlap_start": common_start_text,
            "common_overlap_end": common_end_text,
            "overlap_seconds": common_overlap,
            "alignment_status": status,
            "alignment_confidence": confidence,
            "evidence": "Connected component is based on overlapping normalized ground-truth ts intervals and attack_type only.",
            "notes": "No common ground_truth_id/session/run ID crosses platforms; therefore even positive overlap is not SAME_EVENT_CONFIRMED. A missing platform in this row is not proof that the attack was absent from that modality.",
        })
    return rows


def build_feasibility(
    root: Path,
    inventory: list[dict[str, Any]],
    windows: list[dict[str, Any]],
    ip_rows: list[dict[str, Any]],
    offsets: list[dict[str, Any]],
    alignment: list[dict[str, Any]],
) -> str:
    attack_types = sorted({window["attack_type"] for window in windows if window["attack_type"] in ATTACK_TYPES})
    status_counts = Counter(row["alignment_status"] for row in alignment)
    high_conf_rows = sorted(
        (row for row in ip_rows if row["validation_status"] == "CONFIRMED_IN_GROUND_TRUTH_AND_NETWORK_SRC"),
        key=lambda row: int(row["ground_truth_src_ip_count"]),
        reverse=True,
    )
    high_conf_ips = [row["ip"] for row in high_conf_rows[:30]]
    offset_summary = ", ".join(f"{row['platform']}/{row['attack_type']}={row['observed_offset_seconds']}s ({row['offset_consistent']})" for row in offsets if row["reference_ground_truth"].endswith("numeric ts epoch"))
    return f"""# Phase 1.2 multi-platform feasibility assessment

## Evidence scope

- Ground-truth root used: `{relpath(root, root / 'SecuityEvents_GroundTruth_datasets') if (root / 'SecuityEvents_GroundTruth_datasets').exists() else 'UNKNOWN'}`. The requested spelling `SecurityEvents_GroundTruth_datasets` was absent; the on-disk spelling is recorded in `ground_truth_inventory.csv`.
- Official ground-truth files scanned: **{len(inventory)}**; canonical attack windows: **{len(windows)}**; attack types: `{', '.join(attack_types)}`.
- Candidate Network source IPs confirmed in both ground truth and processed Network `src_ip`: `{', '.join(high_conf_ips[:30]) or 'NONE'}`.

## Answers

### 1. Were all four sources demonstrably collected in parallel?

No. The ground truth has corresponding files for Network, IoT, Linux, and Windows, and their numeric `ts` ranges overlap for several attack types. However, the files do not expose a common acquisition/session ID, common host clock declaration, or cross-platform run identifier. File correspondence and calendar overlap are not sufficient evidence of parallel collection.

### 2. Can the same attack event be identified across platforms?

No event is marked `SAME_EVENT_CONFIRMED`. The v2 alignment report contains only evidence classes such as `TIME_OVERLAP_ONLY`, `NO_OVERLAP`, or `UNRESOLVED`, because ground-truth IDs are platform/file-specific. Some attack types have overlapping normalized intervals across multiple platforms, but there is no shared event ID or entity tuple proving identity.

### 3. Is there a common clock or normalizable offset?

Numeric `ts` values can be normalized as epoch timestamps. For IoT, the co-present `data/time` field is a local clock that differs from the numeric `ts` by an observed offset; the report keeps both direct local-clock comparison and epoch comparison. This is evidence for a representation offset, not proof that every platform host clock was synchronized. See `time_offset_validation.csv`.

Observed offset summary: {offset_summary or 'UNKNOWN'}

### 4. Which platform alignment is most trustworthy?

Network has the strongest structural ground-truth linkage because the official files contain `ts`, `type`, `src_ip`, `dst_ip`, ports, and protocol; Network file numbering is not assumed, and links are reconstructed by type plus timestamp overlap. IoT has strong type/time correspondence but its processed `date/time` is local-clock data. Windows has direct type/time ground truth but no entity/IP fields. Linux remains weaker because the ground truth aggregates disk/memory/process views and does not expose an OS/run identifier in the ground-truth file.

### 5. Is Linux still unresolved?

Yes, partially. Linux ground-truth timestamps and processed `ts` can be compared, but raw OS/scenario provenance, host identity, and process/entity continuity are not established. Linux mappings therefore use lower confidence than direct Network/IoT/Windows file mappings.

### 6. Can common temporal windows be created?

Candidate windows can be computed from normalized ground-truth `ts`, and the report shows their overlap. They should be called **candidate temporal overlap windows**, not synchronized multimodal windows, until a common clock/session basis is documented.

### 7. Can event + time window + semantic entity link modalities?

Potentially, but only as a provenance-aware research protocol. Network has IP/port/protocol evidence; Linux has PID; IoT and Windows do not expose equivalent cross-platform entities in the official ground truth. A future linkage should require event evidence, timestamp basis, entity compatibility, and an uncertainty flag, with negative/ablation controls.

### 8. Is there enough scientific basis for a heterogeneous temporal multi-platform IDS?

**Not enough to claim a synchronized four-platform dataset or a confirmed same-event multimodal benchmark.** There is enough evidence to design a cautious event-conditioned feasibility study: keep modalities separate, use ground-truth event IDs within source, evaluate time-overlap-only hypotheses separately, and report Linux/timezone/entity uncertainty. A strong multi-platform model claim would require additional source documentation or a shared event/session manifest.

Alignment status counts: `{dict(status_counts)}`.

## Guardrails

This phase did not concatenate schemas, build a unified table, fill missing features, scale, encode, split, oversample, construct a graph, or train a model. The Phase 1.1 reports remain unchanged; all rebuilt outputs use separate v2 filenames.
"""


def update_audit_summary(root: Path, summary: dict[str, Any]) -> None:
    marker = "## Phase 1.2 Ground Truth Validation"
    section = f"""
## Phase 1.2 Ground Truth Validation

- Ground-truth source used: `{summary['ground_truth_root']}`. The request's spelling `SecurityEvents_GroundTruth_datasets` differs from the on-disk spelling `SecuityEvents_GroundTruth_datasets`; no rename was performed.
- Scanned **{summary['inventory_count']}** official ground-truth CSVs and reconstructed **{summary['window_count']}** canonical attack windows across `{', '.join(summary['attack_types'])}`.
- Numeric `ts` is available across the official files. IoT additionally contains `data` + `time`; the observed local-clock relation is retained separately from epoch-normalized UTC. No universal platform timezone or common host clock was assumed.
- The documented candidate range `{DOCUMENTED_ATTACKER_RANGE}` was checked against actual data. Ground-truth/processed Network IP evidence is reported in `attacker_ip_validation.csv`; the candidate range is not used as a hard-coded labeling rule.
- `ground_truth_label_validation.csv` compares each mapped processed file/window for label 0/1, canonical type, normal rows inside windows, attack rows outside windows, and uncovered windows. It reports evidence without correcting labels.
- `time_offset_validation.csv` reports per-platform/attack offsets. IoT has a local-vs-epoch representation distinction; other zero-offset results validate numeric representation overlap only, not independent host-clock synchronization.
- Rebuilt `scenario_mapping_v2.csv` uses ground-truth-backed IDs. Rebuilt `cross_platform_alignment_v2.csv` does not mark any event `SAME_EVENT_CONFIRMED` without a shared cross-platform event/session ID. Positive overlap is retained as `TIME_OVERLAP_ONLY`; ransomware has `NO_OVERLAP` across the observed platform windows.
- Current feasibility conclusion: **insufficient evidence to claim a synchronized four-platform temporal benchmark or confirmed same-event multimodal dataset**. A cautious event-conditioned feasibility study remains possible with modalities kept separate.

Reports: `ground_truth_inventory.csv`, `ground_truth_attack_windows.csv`, `attacker_ip_validation.csv`, `ground_truth_label_validation.csv`, `time_offset_validation.csv`, `scenario_mapping_v2.csv`, `cross_platform_alignment_v2.csv`, and `multiplatform_feasibility.md`.
"""
    update_report_section(root, marker, section)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--resume", action="store_true", help="Resume after ground_truth_attack_windows.csv and ground_truth_label_validation.csv already exist.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    root = args.project_root.resolve()
    requested = root / "SecurityEvents_GroundTruth_datasets"
    typo_root = root / "SecuityEvents_GroundTruth_datasets"
    if requested.exists():
        gt_root = requested
    elif typo_root.exists():
        gt_root = typo_root
    else:
        raise FileNotFoundError("Neither SecurityEvents_GroundTruth_datasets nor SecuityEvents_GroundTruth_datasets exists")
    reports = root / "research_pipeline" / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Using ground-truth root: %s", gt_root)
    if args.resume:
        inventory = load_csv_rows(reports / "ground_truth_inventory.csv")
        windows = load_windows_report(root)
        LOGGER.info("Resuming from reports: files=%s windows=%s; revalidating processed labels with timestamp/type mapping", len(inventory), len(windows))
        label_rows, processed_info = validate_processed_labels(root, windows)
        gt_ip_info = {"ip_stats": ground_truth_network_ip_stats(gt_root), "ip_files": {}}
    else:
        inventory, windows, gt_ip_info = scan_ground_truth(root, gt_root)
        label_rows, processed_info = validate_processed_labels(root, windows)
    write_csv(
        reports / "ground_truth_inventory.csv",
        inventory,
        ["platform", "source_file", "file_name", "file_size_mb", "rows", "columns", "timestamp_fields", "timestamp_unit_observed", "timestamp_min_raw", "timestamp_max_raw", "timestamp_min_normalized", "timestamp_max_normalized", "type_field", "ip_fields", "entity_fields", "attack_types", "read_status", "notes"],
    )
    write_csv(
        reports / "ground_truth_attack_windows.csv",
        windows,
        ["ground_truth_id", "attack_type", "start_time", "end_time", "attacker_ip", "victim", "platform", "source_file", "raw_timestamp", "normalized_timestamp", "timezone", "evidence", "confidence", "notes"],
    )

    write_csv(
        reports / "ground_truth_label_validation.csv",
        label_rows,
        ["platform", "processed_file", "ground_truth_id", "attack_type", "time_basis", "window_start", "window_end", "records_in_window", "label_1_in_window", "label_0_in_window", "normal_type_in_window", "canonical_type_match", "canonical_type_mismatch", "raw_types_in_window", "records_label_1_outside_windows", "records_label_0_outside_windows", "attack_type_outside_windows", "ground_truth_window_covered", "confidence", "notes"],
    )

    LOGGER.info("Scanning processed Network IP fields...")
    processed_ips = network_processed_ip_stats(root)
    ip_rows = build_ip_validation(root, gt_ip_info, processed_ips)
    write_csv(
        reports / "attacker_ip_validation.csv",
        ip_rows,
        ["ip", "documented_candidate_range", "documented_candidate_match", "ground_truth_src_ip_count", "ground_truth_dst_ip_count", "processed_network_src_ip_count", "processed_network_dst_ip_count", "ground_truth_role_evidence", "observed_in_ground_truth", "observed_in_processed_network", "validation_status", "confidence", "notes"],
    )

    offsets = build_time_offsets(windows, processed_info)
    write_csv(
        reports / "time_offset_validation.csv",
        offsets,
        ["platform", "attack_type", "reference_ground_truth", "observed_offset_seconds", "start_offset_seconds", "end_offset_seconds", "offset_consistent", "timezone_interpretation", "confidence", "evidence"],
    )

    mapping_v2 = scenario_mapping_v2(root, windows, label_rows)
    write_csv(
        reports / "scenario_mapping_v2.csv",
        mapping_v2,
        ["ground_truth_id", "scenario_id", "attack_type", "platform", "ground_truth_source_file", "processed_source_file", "raw_provenance", "start_time", "end_time", "attacker_ip", "victim", "mapping_confidence", "evidence", "notes"],
    )

    alignment_v2 = build_alignment_v2(windows)
    write_csv(
        reports / "cross_platform_alignment_v2.csv",
        alignment_v2,
        ["ground_truth_id", "scenario_id", "attack_type", "network_present", "iot_present", "linux_present", "windows_present", "network_window", "iot_window", "linux_window", "windows_window", "common_overlap_start", "common_overlap_end", "overlap_seconds", "alignment_status", "alignment_confidence", "evidence", "notes"],
    )

    feasibility = build_feasibility(root, inventory, windows, ip_rows, offsets, alignment_v2)
    (reports / "multiplatform_feasibility.md").write_text(feasibility, encoding="utf-8")
    summary = {
        "ground_truth_root": relpath(root, gt_root),
        "inventory_count": len(inventory),
        "window_count": len(windows),
        "attack_types": sorted({window["attack_type"] for window in windows if window["attack_type"] in ATTACK_TYPES}),
    }
    update_audit_summary(root, summary)
    LOGGER.info("Phase 1.2 reports written: files=%s windows=%s label_rows=%s alignment_rows=%s", len(inventory), len(windows), len(label_rows), len(alignment_v2))


if __name__ == "__main__":
    main()
