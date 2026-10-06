#!/usr/bin/env python3
"""Phase-1.1 semantic validation for the local ToN-IoT project.

This module is deliberately audit-only.  It keeps platform/source files
separate and writes evidence reports; it does not concatenate data, impute
values, fit transforms, create a split, or train a model.

Run from the project root:
    python -m research_pipeline.src.validation.semantic_validation
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from ..common.config import CANONICAL_TYPES, PLATFORM_ORDER, canonical_type, load_label_mapping
from ..common.paths import PROJECT_ROOT
from ..common.utils import relative_path as relpath, update_report_section, write_csv_rows as write_csv

LOGGER = logging.getLogger(__name__)


PLATFORMS = PLATFORM_ORDER
ATTACK_TYPES = CANONICAL_TYPES
CONFIDENCES = {"HIGH", "MEDIUM", "LOW", "UNKNOWN", "UNRESOLVED"}


def display_time(value: Any) -> str:
    if value is None or value == "" or (isinstance(value, float) and math.isnan(value)):
        return "UNKNOWN"
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        return timestamp.isoformat()
    return timestamp.isoformat()


def merge_range(current: tuple[pd.Timestamp, pd.Timestamp] | None, values: Iterable[Any]) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    parsed = pd.to_datetime(pd.Series(list(values)), errors="coerce")
    parsed = parsed.dropna()
    if parsed.empty:
        return current
    left, right = parsed.min(), parsed.max()
    if current is None:
        return pd.Timestamp(left), pd.Timestamp(right)
    return min(current[0], pd.Timestamp(left)), max(current[1], pd.Timestamp(right))


def processed_type_ranges(root: Path) -> tuple[dict[tuple[str, str], tuple[pd.Timestamp, pd.Timestamp]], dict[tuple[str, str], list[str]]]:
    """Stream the two semantic columns needed for per-type intervals."""
    processed = root / "Processed_datasets"
    ranges: dict[tuple[str, str], tuple[pd.Timestamp, pd.Timestamp]] = {}
    files_by_type: dict[tuple[str, str], set[str]] = defaultdict(set)
    platform_dirs = {
        "IoT": processed / "Processed_IoT_dataset",
        "Linux": processed / "Processed_Linux_dataset",
        "Network": processed / "Processed_Network_dataset",
        "Windows": processed / "Processed_Windows_dataset",
    }
    for platform, directory in platform_dirs.items():
        for path in sorted(directory.glob("*.csv")):
            try:
                header = pd.read_csv(path, nrows=0).columns.tolist()
            except Exception:
                continue
            if "type" not in header:
                continue
            time_columns = ["date", "time"] if platform == "IoT" else ["ts"]
            if not all(column in header for column in time_columns):
                continue
            usecols = ["type", *time_columns]
            for chunk in pd.read_csv(path, usecols=usecols, chunksize=500_000, low_memory=False):
                chunk = chunk.dropna(subset=["type"])
                if chunk.empty:
                    continue
                if platform == "IoT":
                    values = pd.to_datetime(
                        chunk["date"].astype(str).str.strip() + " " + chunk["time"].astype(str).str.strip(),
                        format="%d-%b-%y %H:%M:%S",
                        errors="coerce",
                    )
                else:
                    numeric = pd.to_numeric(chunk["ts"], errors="coerce")
                    values = pd.to_datetime(numeric, unit="s", errors="coerce")
                valid = values.notna()
                if not valid.any():
                    continue
                small = pd.DataFrame({"type": chunk.loc[valid, "type"].astype(str), "event_time": values.loc[valid]})
                for raw_type, group in small.groupby("type", sort=False):
                    attack = canonical_type(raw_type)
                    if not attack:
                        continue
                    interval = (group["event_time"].min(), group["event_time"].max())
                    key = (platform, attack)
                    ranges[key] = interval if key not in ranges else (
                        min(ranges[key][0], interval[0]),
                        max(ranges[key][1], interval[1]),
                    )
                    files_by_type[key].add(path.name)
    return ranges, {key: sorted(values) for key, values in files_by_type.items()}


def raw_iot_ranges(root: Path) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    """Read IoT JSONL timestamps; unlike processed CSV, the raw timestamp has GMT."""
    base = root / "Raw_datasets" / "telemtry_data"
    ranges: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
    timestamp_pattern = re.compile(r'"Timestamp"\s*:\s*"([^"]+)"')
    for path in sorted(base.glob("IoT_original_normal_attack/*/*.log")):
        folder = path.parent.name
        match = re.match(r"IoT_(.+)_normal$", folder, flags=re.IGNORECASE)
        attack = canonical_type(match.group(1)) if match else ""
        if not attack or attack == "normal":
            continue
        local: list[pd.Timestamp] = []
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                timestamp_match = timestamp_pattern.search(line)
                if not timestamp_match:
                    continue
                parsed = pd.to_datetime(timestamp_match.group(1), utc=True, errors="coerce")
                if not pd.isna(parsed):
                    local.append(parsed)
        if not local:
            continue
        interval = (min(local), max(local))
        ranges[attack] = interval if attack not in ranges else (
            min(ranges[attack][0], interval[0]),
            max(ranges[attack][1], interval[1]),
        )
    return ranges


def raw_windows_ranges(root: Path) -> tuple[dict[str, tuple[pd.Timestamp, pd.Timestamp]], str, int]:
    """Read only the first timestamp column from filtered Windows PDH CSVs."""
    base = root / "Raw_datasets" / "windows_data"
    ranges: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
    timezone_evidence = "UNKNOWN"
    files_seen = 0
    attack_folders = list((base / "Windows_filtered-normal-attack").glob("Windows_filtered_*_normal"))
    paths = [file for folder in attack_folders for file in folder.glob("*.csv")]
    paths += list((base / "Windows_filtered-normal").glob("win*_normal_*.csv"))
    for path in sorted(paths):
        files_seen += 1
        try:
            with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
                reader = csv.reader(handle)
                first = next(reader, [])
                if first:
                    header_match = re.search(r"\(([^()]*(?:Standard|Daylight)[^()]*)\)\(([-+]\d+)\)", first[0])
                    if header_match:
                        timezone_evidence = f"{header_match.group(1)} ({header_match.group(2)})"
                values: list[pd.Timestamp] = []
                for row in reader:
                    if not row or not row[0].strip():
                        continue
                    parsed = pd.to_datetime(row[0].strip(), format="%m/%d/%Y %H:%M:%S.%f", errors="coerce")
                    if not pd.isna(parsed):
                        values.append(parsed)
        except OSError:
            continue
        if not values:
            continue
        if "Windows_filtered_" in path.parent.name:
            raw = path.parent.name[len("Windows_filtered_") : -len("_normal")]
            attack = canonical_type(raw)
        else:
            attack = "normal"
        if not attack:
            continue
        interval = (min(values), max(values))
        ranges[attack] = interval if attack not in ranges else (
            min(ranges[attack][0], interval[0]),
            max(ranges[attack][1], interval[1]),
        )
    return ranges, timezone_evidence, files_seen


def raw_scenario_folders(root: Path) -> dict[str, list[dict[str, Any]]]:
    """Enumerate source folders without reading the large Network/Linux logs."""
    base = root / "Raw_datasets"
    scenarios: dict[str, list[dict[str, Any]]] = defaultdict(list)

    iot_attack = base / "telemtry_data" / "IoT_original_normal_attack"
    for folder in sorted(iot_attack.glob("IoT_*_normal")):
        match = re.match(r"IoT_(.+)_normal$", folder.name, flags=re.IGNORECASE)
        attack = canonical_type(match.group(1)) if match else ""
        if attack:
            scenarios["IoT"].append({"attack": attack, "path": folder, "evidence": f"{relpath(root, folder)}; {len(list(folder.glob('*.log')))} raw JSONL files"})
    iot_normal = base / "telemtry_data" / "IoT_original_normal"
    scenarios["IoT"].append({"attack": "normal", "path": iot_normal, "evidence": f"{relpath(root, iot_normal)}; {len(list(iot_normal.glob('*.log')))} raw JSONL files"})

    win_attack = base / "windows_data" / "Windows_filtered-normal-attack"
    for folder in sorted(win_attack.glob("Windows_filtered_*_normal")):
        raw = folder.name[len("Windows_filtered_") : -len("_normal")]
        attack = canonical_type(raw)
        if attack:
            scenarios["Windows"].append({"attack": attack, "path": folder, "evidence": f"{relpath(root, folder)}; filtered PDH CSV folder"})
    win_normal = base / "windows_data" / "Windows_filtered-normal"
    scenarios["Windows"].append({"attack": "normal", "path": win_normal, "evidence": f"{relpath(root, win_normal)}; filtered PDH CSV files"})

    net_attack = base / "network_data" / "Network_dataset_Bro" / "normal_attack_Bro"
    for folder in sorted(net_attack.glob("normal_*")):
        raw = folder.name[len("normal_") :]
        attack = canonical_type(raw)
        if attack:
            count = sum(1 for item in folder.rglob("*") if item.is_file())
            scenarios["Network"].append({"attack": attack, "path": folder, "evidence": f"{relpath(root, folder)}; {count} Bro/raw files; #types time and #unset_field - observed"})
    net_normal = base / "network_data" / "Network_dataset_Bro" / "normal_Bro"
    scenarios["Network"].append({"attack": "normal", "path": net_normal, "evidence": f"{relpath(root, net_normal)}; normal Bro/raw files"})

    linux_attack = base / "Linux_data" / "Original_Linux_dataset" / "normal_attack"
    for os_folder in sorted(linux_attack.glob("atop_ubuntu*")):
        for folder in sorted(os_folder.glob("atop_*")):
            raw = folder.name[len("atop_") :]
            attack = canonical_type(raw)
            if attack:
                count = sum(1 for item in folder.rglob("*") if item.is_file())
                scenarios["Linux"].append({"attack": attack, "path": folder, "os": os_folder.name, "evidence": f"{relpath(root, folder)}; {count} atop text files; raw row timestamp not evidenced"})
    linux_normal = base / "Linux_data" / "Original_Linux_dataset" / "normal"
    for folder in sorted(linux_normal.glob("atop_ubuntu*_normal")):
        os_name = re.search(r"ubuntu\d+", folder.name, flags=re.IGNORECASE)
        scenarios["Linux"].append({"attack": "normal", "path": folder, "os": os_name.group(0) if os_name else "UNKNOWN", "evidence": f"{relpath(root, folder)}; normal atop source; raw row timestamp not evidenced"})
    return scenarios


def interval_text(ranges: dict[tuple[str, str], tuple[pd.Timestamp, pd.Timestamp]], key: tuple[str, str]) -> tuple[str, str]:
    interval = ranges.get(key)
    return (display_time(interval[0]), display_time(interval[1])) if interval else ("UNKNOWN", "UNKNOWN")


def build_time_report(root: Path, processed_ranges: dict[tuple[str, str], tuple[pd.Timestamp, pd.Timestamp]], windows_tz: str) -> list[dict[str, str]]:
    return [
        {
            "platform": "IoT",
            "source": "Raw_datasets/telemtry_data/IoT_original_normal_attack/*.log + Processed_datasets/Processed_IoT_dataset/*.csv",
            "timestamp_field": "raw JSON Timestamp; processed date + time",
            "timestamp_unit": "Calendar timestamp, 1-second resolution",
            "timezone": "GMT (raw only; processed CSV drops timezone)",
            "timezone_evidence": 'Raw JSON contains values such as "Thu, 25 Apr 2019 14:42:33 GMT"; processed columns are date/time only.',
            "clock_source": "IoT telemetry logger",
            "confidence": "MEDIUM",
            "notes": "Raw timezone is explicit, but the processed representation no longer carries the offset; do not silently treat processed naive time as UTC for cross-platform alignment.",
        },
        {
            "platform": "Network",
            "source": "Raw_datasets/network_data/Network_dataset_Bro + Processed_datasets/Processed_Network_dataset/*.csv",
            "timestamp_field": "ts",
            "timestamp_unit": "Unix epoch seconds with fractional precision",
            "timezone": "UNKNOWN",
            "timezone_evidence": "Raw Bro metadata exposes #types time and numeric fractional timestamps; neither the raw header nor processed ts encodes a timezone.",
            "clock_source": "Bro/Zeek capture host clock (host timezone not evidenced)",
            "confidence": "LOW",
            "notes": "Numeric epoch conversion is supported, but the host/UTC convention is not proven by the local source files.",
        },
        {
            "platform": "Linux",
            "source": "Raw_datasets/Linux_data + Processed_datasets/Processed_Linux_dataset/*.csv",
            "timestamp_field": "processed ts; raw atop row timestamp not evidenced",
            "timestamp_unit": "Unix epoch seconds in processed CSV (raw unit unresolved)",
            "timezone": "UNKNOWN",
            "timezone_evidence": "Processed ts parses as epoch-like seconds; inspected raw atop text and filtered feature CSVs do not provide explicit timezone metadata.",
            "clock_source": "atop/system monitor host clock",
            "confidence": "UNKNOWN",
            "notes": "Do not align Linux to another platform until the raw-to-processed timestamp construction is documented.",
        },
        {
            "platform": "Windows",
            "source": "Raw_datasets/windows_data/Windows_filtered-normal(-attack)/*.csv + Processed_datasets/Processed_Windows_dataset/*.csv",
            "timestamp_field": "filtered PDH first field; processed ts",
            "timestamp_unit": "Local wall-clock in filtered PDH CSV; processed ts is numeric epoch-like seconds",
            "timezone": windows_tz if windows_tz != "UNKNOWN" else "UNKNOWN",
            "timezone_evidence": f"PDH CSV header contains (PDH-CSV 4.0) ({windows_tz}); original BLG files are binary and were not treated as timezone evidence." if windows_tz != "UNKNOWN" else "No explicit timezone header recovered.",
            "clock_source": "Windows Performance Data Helper (PDH)/BLG host clock",
            "confidence": "MEDIUM",
            "notes": "The local PDH timezone is explicit, but conversion from local filtered time to processed ts and DST behavior remain to be verified.",
        },
    ]


def build_scenario_mapping(
    root: Path,
    processed_ranges: dict[tuple[str, str], tuple[pd.Timestamp, pd.Timestamp]],
    processed_files: dict[tuple[str, str], list[str]],
    raw_scenarios: dict[str, list[dict[str, Any]]],
    iot_raw: dict[str, tuple[pd.Timestamp, pd.Timestamp]],
    windows_raw: dict[str, tuple[pd.Timestamp, pd.Timestamp]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for platform in PLATFORMS:
        for index, item in enumerate(raw_scenarios.get(platform, []), start=1):
            attack = item["attack"]
            key = (platform, attack)
            processed_start, processed_end = interval_text(processed_ranges, key)
            source_path = item["path"]
            source_file = relpath(root, source_path)
            files = processed_files.get(key, [])
            if files:
                source_file += "; processed=" + ",".join(files)
            raw_interval: tuple[pd.Timestamp, pd.Timestamp] | None = None
            if platform == "IoT" and attack in iot_raw:
                raw_interval = iot_raw[attack]
            elif platform == "Windows" and attack in windows_raw:
                raw_interval = windows_raw[attack]
            if raw_interval:
                start_time, end_time = display_time(raw_interval[0]), display_time(raw_interval[1])
            elif platform in {"Network", "IoT", "Windows"} and processed_start != "UNKNOWN":
                start_time, end_time = processed_start, processed_end
            else:
                start_time, end_time = "UNKNOWN", "UNKNOWN"
            if platform == "Linux":
                confidence = "UNRESOLVED"
                notes = "Folder/OS/attack naming is evidence, but raw atop rows do not expose a timestamp or stable key that maps this folder to processed windows."
            elif platform == "Network":
                confidence = "MEDIUM" if attack != "normal" and files else "LOW"
                notes = "Attack folder name matches canonical type and processed type vocabulary; interval is processed-derived and normal records are interleaved across files. No shared run ID."
            elif platform == "Windows":
                confidence = "MEDIUM" if attack != "normal" and files else "LOW"
                notes = "Filtered PDH folder name and raw timestamp range support the mapping; timezone conversion to processed ts is not yet proven."
            else:
                confidence = "MEDIUM" if attack != "normal" and files else "LOW"
                notes = "Raw IoT JSONL folder name, explicit raw timestamp, and processed type vocabulary agree; processed files are service profiles, not one isolated scenario file."
            rows.append(
                {
                    "scenario_id": f"{platform}::{attack}::source_{index}",
                    "attack_type": attack,
                    "platform": platform,
                    "source_file": source_file,
                    "start_time": start_time,
                    "end_time": end_time,
                    "source_evidence": item["evidence"],
                    "mapping_confidence": confidence,
                    "notes": notes,
                }
            )
    return rows


def build_alignment_report(
    processed_ranges: dict[tuple[str, str], tuple[pd.Timestamp, pd.Timestamp]],
    mapping_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    available = {(row["platform"], row["attack_type"]) for row in mapping_rows}
    rows: list[dict[str, Any]] = []
    for attack in ATTACK_TYPES:
        intervals: dict[str, tuple[str, str]] = {}
        present: dict[str, str] = {}
        parsed: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
        for platform in PLATFORMS:
            key = (platform, attack)
            interval = processed_ranges.get(key)
            present[platform] = "yes" if key in available else "no"
            if interval:
                parsed[platform] = interval
                intervals[platform] = (display_time(interval[0]), display_time(interval[1]))
            else:
                intervals[platform] = ("UNKNOWN", "UNKNOWN")
        if len(parsed) == len(PLATFORMS):
            common_start = max(interval[0] for interval in parsed.values())
            common_end = min(interval[1] for interval in parsed.values())
            overlap = max(0.0, (common_end - common_start).total_seconds())
            common_start_text = display_time(common_start) if overlap > 0 else "NONE"
            common_end_text = display_time(common_end) if overlap > 0 else "NONE"
        else:
            common_start_text, common_end_text, overlap = "UNKNOWN", "UNKNOWN", "UNKNOWN"
        rows.append(
            {
                "scenario_id": f"UNRESOLVED_ATTACK_COHORT::{attack}",
                "attack_type": attack,
                "network_present": present["Network"],
                "iot_present": present["IoT"],
                "linux_present": present["Linux"],
                "windows_present": present["Windows"],
                "network_start": intervals["Network"][0],
                "network_end": intervals["Network"][1],
                "iot_start": intervals["IoT"][0],
                "iot_end": intervals["IoT"][1],
                "linux_start": intervals["Linux"][0],
                "linux_end": intervals["Linux"][1],
                "windows_start": intervals["Windows"][0],
                "windows_end": intervals["Windows"][1],
                "common_overlap_start": common_start_text,
                "common_overlap_end": common_end_text,
                "overlap_seconds": overlap,
                "alignment_confidence": "UNRESOLVED",
                "notes": "Same attack label/folder naming only; no shared scenario_id, host/device identity, or proven cross-platform clock convention. Calendar overlap is not treated as synchronization.",
            }
        )
    return rows


def build_label_mapping() -> dict[str, Any]:
    """Keep the committed mapping as the only source of label aliases."""
    return load_label_mapping()


def classify_missing(feature: str) -> tuple[str, str, str, str]:
    lowered = feature.lower()
    if lowered.startswith(("dns_", "http_", "ssl_", "weird_")):
        return (
            "STRUCTURAL_NOT_APPLICABLE",
            "Bro protocol/event-specific field is unset when the corresponding DNS/HTTP/SSL/violation event is absent; raw Bro uses #unset_field -.",
            "Preserve raw '-' and add an explicit semantic state; do not zero-impute or call it NaN without a field-specific rule.",
            "HIGH",
        )
    if lowered in {"service", "local_orig", "local_resp"}:
        return (
            "STRUCTURAL_NOT_APPLICABLE",
            "Field is a connection/event state and '-' denotes no detected value or not applicable in the Bro record context.",
            "Keep the categorical placeholder as an explicit state or map it to a documented structural category; never numeric zero-fill.",
            "MEDIUM",
        )
    if lowered in {"proto", "conn_state", "history", "orig_state", "resp_state"}:
        return (
            "VALID_CATEGORY",
            "Categorical protocol/connection-state field; '-' may be a valid source category rather than a missing numeric measurement.",
            "Retain as a categorical level until the Bro field definition confirms a separate not-applicable state.",
            "MEDIUM",
        )
    return (
        "UNKNOWN",
        "The literal '-' is present, but this report has no field-specific evidence that it means structural absence, missing capture, or a valid category.",
        "Keep raw token and review against the field definition before any encoding or imputation.",
        "UNKNOWN",
    )


def build_missing_semantics(root: Path) -> list[dict[str, Any]]:
    schema_path = root / "research_pipeline" / "reports" / "schema_by_file.csv"
    grouped: dict[str, dict[str, Any]] = {}
    with schema_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("platform") != "Network" or int(float(row.get("placeholder_count", 0) or 0)) <= 0:
                continue
            feature = row["column_name"]
            item = grouped.setdefault(feature, {"cells": 0, "files": 0})
            item["cells"] += int(float(row["placeholder_count"]))
            item["files"] += 1
    rows: list[dict[str, Any]] = []
    for feature in sorted(grouped):
        semantic_class, reason, handling, confidence = classify_missing(feature)
        rows.append(
            {
                "platform": "Network",
                "feature": feature,
                "raw_missing_token": "-",
                "semantic_class": semantic_class,
                "reason": reason,
                "recommended_handling": handling,
                "confidence": confidence,
                "placeholder_count": grouped[feature]["cells"],
                "files_with_placeholder": grouped[feature]["files"],
            }
        )
    return rows


def feature_classification(feature: str, platform: str, unique_ratio: float) -> tuple[str, str, str]:
    lowered = feature.lower()
    if lowered in {"label", "target", "ground_truth", "class"}:
        return "TARGET_LEAKAGE", "Ground-truth target; never use as an input feature.", "TARGET_LEAKAGE"
    if lowered in {"type", "attack", "attack_type", "category", "attack_category"}:
        return "TARGET_LEAKAGE", "Attack category is label-adjacent and exposes scenario semantics; retain only in target/provenance views.", "TARGET_LEAKAGE"
    if feature in {"__file_name__", "__file_path__"}:
        return "SPLIT_LEAKAGE_RISK", "File/path metadata can encode attack, platform, source, or author split construction.", "SPLIT_LEAKAGE_RISK"
    if lowered in {"ts", "timestamp", "date", "time", "datetime", "event_time"}:
        return "SPLIT_LEAKAGE_RISK", "Timestamp is needed for ordering and alignment but can leak future information if used as an ordinary feature or split after preprocessing.", "SPLIT_LEAKAGE_RISK"
    if lowered.endswith(("_bytes", "_pkts", "_packets")) or lowered in {"bytes", "packets", "ip_bytes"}:
        return "VALID_OBSERVABLE_FEATURE", "Traffic-volume measurement rather than an endpoint identity; still validate whether it is post-event aggregated.", "VALID_OBSERVABLE_FEATURE"
    identifier_terms = ("src_ip", "dst_ip", "ip", "uid", "pid", "process id", "process_id", "device", "host", "port")
    is_identifier = any(term in lowered for term in identifier_terms)
    if is_identifier:
        if unique_ratio >= 0.10 or lowered in {"uid", "pid"} or "process id" in lowered:
            return "HIGH_CARDINALITY_IDENTIFIER", "Identity/key-like field with high or unique-per-record cardinality; it may be useful for grouping or graph edges but risks memorization and split leakage.", "SEMANTIC_GRAPH_KEY"
        return "SEMANTIC_GRAPH_KEY", "IP/port/PID/UID/device/process identity is a semantic key for grouping or graph construction, not automatically target leakage.", "SEMANTIC_GRAPH_KEY"
    if lowered in {"cmd", "command", "process_name"} or "cmd" in lowered:
        return "VALID_OBSERVABLE_FEATURE", "Observed process/command context; retain only after checking environment specificity and sensitive-content policy.", "REVIEW_REQUIRED"
    if any(term in lowered for term in ("proto", "service", "state", "status", "history", "category", "condition", "mode")):
        return "REVIEW_REQUIRED", "Categorical/state feature; semantics and missing-category policy must be fixed before encoding.", "REVIEW_REQUIRED"
    return "VALID_OBSERVABLE_FEATURE", "Observed measurement or telemetry statistic; validate range, missingness, and train-only preprocessing.", "VALID_OBSERVABLE_FEATURE"


def build_feature_risk(root: Path) -> list[dict[str, Any]]:
    schema_path = root / "research_pipeline" / "reports" / "schema_by_file.csv"
    inventory_path = root / "research_pipeline" / "reports" / "dataset_inventory.csv"
    rows_by_file: dict[tuple[str, str], int] = {}
    with inventory_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            rows_by_file[(row["platform"], row["file_name"])] = int(float(row["rows"]))
    output: list[dict[str, Any]] = []
    seen_files: set[tuple[str, str]] = set()
    with schema_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            platform, file_name = row["platform"], row["file_name"]
            seen_files.add((platform, file_name))
            row_count = rows_by_file.get((platform, file_name), 0)
            unique = float(row.get("unique_count", 0) or 0)
            ratio = unique / row_count if row_count else 0.0
            classification, reason, secondary = feature_classification(row["column_name"], platform, ratio)
            output.append(
                {
                    "platform": platform,
                    "file": file_name,
                    "feature": row["column_name"],
                    "classification": classification,
                    "secondary_classification": secondary,
                    "reason": reason,
                    "unique_count": row["unique_count"],
                    "unique_count_method": row.get("unique_count_method", "UNKNOWN"),
                    "unique_ratio": f"{ratio:.8f}",
                    "legacy_classification": "",
                }
            )
    for platform, file_name in sorted(seen_files):
        output.append(
            {
                "platform": platform,
                "file": file_name,
                "feature": "__file_name__",
                "classification": "SPLIT_LEAKAGE_RISK",
                "secondary_classification": "SPLIT_LEAKAGE_RISK",
                "reason": "Synthetic provenance field; keep for audit/grouping but exclude from ML inputs unless a source-held-out protocol explicitly permits it.",
                "unique_count": "1",
                "unique_count_method": "SYNTHETIC",
                "unique_ratio": "0.00000000",
                "legacy_classification": "",
            }
        )
    return output


def build_split_strategy(root: Path) -> str:
    return """# Phase 1.1 split strategy proposal (not executed)

## Recommendation

Use a **hybrid scenario-aware and chronological protocol**, with the split unit kept inside each platform/source rather than creating a unified table:

1. Where raw folder provenance is reliable (especially IoT, Network, and filtered Windows), reserve complete source scenarios or source runs for an **unseen-scenario test**. Do not let rows from the same raw folder appear in train and test.
2. On the remaining scenarios, order rows by a validated event time and create train/validation/test cutoffs chronologically. Keep the original timestamp, source file, and scenario ID in the provenance view.
3. Linux currently has unresolved raw-to-processed scenario/time provenance. Until that is repaired, use a conservative chronological/file-block protocol and report it as a separate limitation; do not pretend that an attack folder name is a synchronized run ID.

This recommendation is a protocol design only. No rows have been split, concatenated, imputed, scaled, encoded, oversampled, or modeled.

## Candidate strategies

| Strategy | Strength | Leakage risk | Assessment |
|---|---|---|---|
| Random row split | Simple and often class-balanced | Very high: adjacent telemetry, duplicate rows, same scenario, IP/process identity, and future information can cross the boundary | Reject as the primary result |
| Chronological split | Respects temporal ordering and deployment direction | A single source/scenario can still occur on both sides; repeated entities can cross; timestamps are not yet equally trustworthy across platforms | Good within a validated source/scenario |
| Scenario/source split | Tests generalization to an unseen run/source | Class distribution can be unstable; some attack types may be absent from a fold; scenario mapping is unresolved for Linux | Required for the unseen-scenario test where evidence supports it |
| Hybrid (recommended) | Combines unseen-scenario generalization with forward-in-time evaluation | Requires provenance, timestamp validation, and explicit fold feasibility checks | Best fit for this audit state |

## Leakage controls

- Keep `label` and `type` out of input features; preserve both in target/semantic views.
- Do not use filename/path, scenario metadata, or a post-event aggregate as an input feature when that metadata defines the split.
- Keep one raw scenario/source folder in exactly one split. For Network, treat `uid` and endpoint identities as semantic graph keys and report whether IP/port groups cross splits; they are not automatically leakage, but an entity-memorization result is not equivalent to unseen-entity generalization.
- For Linux process views, keep PID/process identity and CMD provenance within a source/group where feasible. PID/CMD are not automatically leakage; they require an explicit process/entity split or an ablation.
- Fit any imputer, encoder, scaler, vocabulary, or threshold using training data only. This is a future protocol requirement, not an action in Phase 1.1.
- Duplicate rows are counted for audit. They should not be allowed to cross a split merely because their source row order differs.

## Class distribution and attack coverage

The current processed counts show strong platform-specific imbalance:

| Platform | Normal | Attack | Normal % | Attack % |
|---|---:|---:|---:|---:|
| IoT | 2,986,455 | 518,816 | 85.20% | 14.80% |
| Linux | 5,041,705 | 813,624 | 86.10% | 13.90% |
| Network | 796,380 | 21,542,641 | 3.56% | 96.44% |
| Windows | 47,258 | 17,084 | 73.45% | 26.55% |

A row-level stratified split could preserve these counts while still leaking scenario context. Therefore every proposed fold must report:

- normal/attack counts and proportions by platform/source;
- canonical attack-type coverage and counts in each fold;
- whether an attack type is absent because it was deliberately held out or because the source has no such scenario;
- per-file/per-scenario row counts before any sampling.

For the unseen-scenario test, do not force every attack type into every fold. Instead report the coverage matrix and mark missing categories explicitly. If a paper requires every type in validation, keep a scenario-aware validation fold with feasible coverage and retain a separate unseen-scenario test.

## Temporal ordering and entity policy

The processed files contain timestamp order violations, and timezone/clock semantics are not fully established. Sort within the chosen source/scenario only after validating the timestamp representation; never infer chronology from current row order. IoT raw timestamps explicitly say GMT, Windows filtered PDH headers explicitly say `AUS Eastern Standard Time (-600)`, while Network and Linux timezone conventions remain unproven.

The first reproducible implementation should produce a split manifest with `platform`, `source_file`, `scenario_id`, `event_time`, `split`, and a reason for assignment. It should then run duplicate, entity/IP/process crossing, label coverage, and temporal-order checks before any preprocessing or model training.
"""


def update_audit_summary(root: Path, report_counts: dict[str, int]) -> None:
    marker = "## Phase 1.1 Semantic Validation"
    section = f"""
## Phase 1.1 Semantic Validation

- Duplicate audit was corrected to count each exact repeated raw CSV row after its first occurrence, including repeats within a single pandas chunk and across chunk boundaries. The fingerprint is a length-delimited BLAKE2b-128 digest of the raw CSV fields; the digest collision risk is negligible but not mathematically zero. High-cardinality unique counts use bounded-memory HLL approximation where marked in `schema_by_file.csv`.
- Time semantics are not uniformly confirmed. IoT raw JSON explicitly carries `GMT`; Windows filtered PDH CSV explicitly carries `AUS Eastern Standard Time (-600)`; Network has numeric fractional Bro/Zeek `time` values without timezone evidence; Linux raw-to-processed timezone/unit provenance remains unresolved. See `time_semantic_validation.csv`.
- Scenario mapping is platform/source-specific. Folder names and processed type values provide useful evidence, but they do not establish one shared cross-platform run. See `scenario_mapping.csv`.
- Calendar intervals overlap for some attack-type cohorts, but no shared run ID, host/device identity, or proven clock convention exists. `cross_platform_alignment.csv` therefore marks alignment as `UNRESOLVED` and never equates label-time overlap with synchronization.
- The canonical category map is versioned as `research_pipeline/config/label_mapping.json`; raw type values must be preserved alongside `canonical_type`. Binary evidence is `label=0` normal and `label=1` attack.
- Network `-` placeholders were classified field-by-field in `missing_semantics.csv`. The observed DNS/HTTP/SSL/violation placeholders are structural/not-applicable candidates, not numeric zeros or automatic NaNs.
- `feature_risk.csv` now distinguishes target leakage, split leakage risk, high-cardinality identifiers, semantic graph keys, valid observable features, and review-required fields. IP/port/PID/UID/CMD are not automatically dropped; their use depends on grouping, entity split, and graph/ML view.
- The recommended future protocol is hybrid: unseen-scenario/source holdout where provenance is reliable, followed by chronological train/validation cuts within retained scenarios. No actual split has been created; details are in `split_strategy.md`.

Generated reports: `{report_counts['time']} time-semantic rows`, `{report_counts['scenario']} scenario rows`, `{report_counts['alignment']} alignment cohorts`, `{report_counts['missing']} missing-semantic rows`, and `{report_counts['risk']} feature-risk rows`.
"""
    update_report_section(root, marker, section)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    root = args.project_root.resolve()
    reports = root / "research_pipeline" / "reports"
    config = root / "research_pipeline" / "config"
    reports.mkdir(parents=True, exist_ok=True)
    config.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Streaming processed type/timestamp ranges...")
    processed_ranges, processed_files = processed_type_ranges(root)
    LOGGER.info("Reading IoT raw GMT timestamps...")
    iot_raw = raw_iot_ranges(root)
    LOGGER.info("Reading Windows filtered PDH timestamps...")
    windows_raw, windows_tz, windows_files = raw_windows_ranges(root)
    LOGGER.info("Windows filtered files inspected: %s", windows_files)
    raw_scenarios = raw_scenario_folders(root)

    time_rows = build_time_report(root, processed_ranges, windows_tz)
    write_csv(
        reports / "time_semantic_validation.csv",
        time_rows,
        ["platform", "source", "timestamp_field", "timestamp_unit", "timezone", "timezone_evidence", "clock_source", "confidence", "notes"],
    )

    mapping_rows = build_scenario_mapping(root, processed_ranges, processed_files, raw_scenarios, iot_raw, windows_raw)
    write_csv(
        reports / "scenario_mapping.csv",
        mapping_rows,
        ["scenario_id", "attack_type", "platform", "source_file", "start_time", "end_time", "source_evidence", "mapping_confidence", "notes"],
    )

    alignment_rows = build_alignment_report(processed_ranges, mapping_rows)
    write_csv(
        reports / "cross_platform_alignment.csv",
        alignment_rows,
        ["scenario_id", "attack_type", "network_present", "iot_present", "linux_present", "windows_present", "network_start", "network_end", "iot_start", "iot_end", "linux_start", "linux_end", "windows_start", "windows_end", "common_overlap_start", "common_overlap_end", "overlap_seconds", "alignment_confidence", "notes"],
    )

    (config / "label_mapping.json").write_text(json.dumps(build_label_mapping(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    missing_rows = build_missing_semantics(root)
    write_csv(
        reports / "missing_semantics.csv",
        missing_rows,
        ["platform", "feature", "raw_missing_token", "semantic_class", "reason", "recommended_handling", "confidence", "placeholder_count", "files_with_placeholder"],
    )

    risk_rows = build_feature_risk(root)
    write_csv(
        reports / "feature_risk.csv",
        risk_rows,
        ["platform", "file", "feature", "classification", "secondary_classification", "reason", "unique_count", "unique_count_method", "unique_ratio", "legacy_classification"],
    )

    (reports / "split_strategy.md").write_text(build_split_strategy(root), encoding="utf-8")
    update_audit_summary(
        root,
        {
            "time": len(time_rows),
            "scenario": len(mapping_rows),
            "alignment": len(alignment_rows),
            "missing": len(missing_rows),
            "risk": len(risk_rows),
        },
    )
    LOGGER.info("Semantic validation reports written.")


if __name__ == "__main__":
    main()
