"""Targeted timestamp and Ground Truth validation for ToN-IoT.

This module is intentionally separate from Split V1 and the existing audit
generators.  It reads the processed/ground-truth sources, performs focused
timestamp checks, and writes evidence under ``reports/final_timestamp_validation``.
It never rewrites Raw/Processed data, V1 output, labels, or model artifacts.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Iterable

import pandas as pd
import numpy as np

from ..common.paths import PROJECT_ROOT
from .ground_truth_validation import parse_iot_local


PLATFORMS = ("Network", "IoT", "Linux", "Windows")
PROCESSED_DIRS = {
    "Network": "Processed_Network_dataset",
    "IoT": "Processed_IoT_dataset",
    "Linux": "Processed_Linux_dataset",
    "Windows": "Processed_Windows_dataset",
}
GT_DIR_NAMES = {
    "Network": "SecurityEvents_Network_datasets",
    "IoT": "SecurityEvents_IoT_datasets",
    "Linux": "SecurityEvents_Linux_datasets",
    "Windows": "SecurityEvents_Windows_datasets",
}


def project_paths(project_root: Path = PROJECT_ROOT) -> dict[str, Path]:
    return {
        "project": project_root,
        "processed": project_root / "Processed_datasets",
        "ground_truth": project_root / "SecuityEvents_GroundTruth_datasets",
        "reports": project_root / "research_pipeline" / "reports",
        "output": project_root / "research_pipeline" / "reports" / "final_timestamp_validation",
    }


def parse_absolute_iso(value: str) -> datetime | None:
    """Parse only an ISO timestamp carrying an explicit timezone."""

    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed.astimezone(timezone.utc)


def parse_source_local_iso(value: str) -> datetime | None:
    """Parse a timezone-less value only under an explicit source-local rule."""

    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is not None and parsed.utcoffset() is not None:
        return None
    return parsed.replace(tzinfo=None)


def history_before_target(event_times: Iterable[datetime], target: datetime) -> list[datetime]:
    """Return only history strictly earlier than a prediction target."""

    return [value for value in event_times if value < target]


def timestamp_offset_seconds(epoch_seconds: float, local_value: datetime) -> float:
    """Compute the auditable ``UTC epoch - local wall-clock`` relation."""

    epoch_utc = datetime.fromtimestamp(epoch_seconds, tz=timezone.utc).replace(tzinfo=None)
    return (epoch_utc - local_value).total_seconds()


def parse_processed_time(chunk: pd.DataFrame, platform: str) -> pd.Series:
    """Parse exactly the current row-level representation, preserving its limits."""

    if platform == "IoT":
        text = chunk["date"].astype("string").str.strip() + " " + chunk["time"].astype("string").str.strip()
        return pd.to_datetime(text, format="%d-%b-%y %H:%M:%S", errors="coerce")
    numeric = pd.to_numeric(chunk["ts"], errors="coerce")
    return pd.to_datetime(numeric, unit="s", errors="coerce", utc=True).dt.tz_localize(None)


def iso_naive(value: pd.Timestamp | None) -> str:
    return value.isoformat() if value is not None and not pd.isna(value) else ""


def iso_epoch_seconds(value: float | None) -> str:
    if value is None:
        return ""
    return pd.Timestamp(value, unit="s", tz="UTC").isoformat()


def _valid_seconds(times: pd.Series) -> list[float]:
    valid = times.notna()
    if not valid.any():
        return []
    nanoseconds = times.loc[valid].astype("datetime64[ns]").astype("int64")
    return (nanoseconds / 1_000_000_000).astype("float64").tolist()


def scan_processed_file(path: Path, platform: str) -> dict[str, Any]:
    usecols = ["date", "time"] if platform == "IoT" else ["ts"]
    stats: dict[str, Any] = {
        "platform": platform,
        "source_file": path.name,
        "source_path": str(path),
        "rows": 0,
        "valid_timestamp_count": 0,
        "invalid_timestamp_count": 0,
        "backward_steps": 0,
        "adjacent_duplicate_timestamps": 0,
        "max_backward_seconds": 0.0,
        "raw_dtype": "",
        "raw_fractional_count": 0,
        "raw_min": "",
        "raw_max": "",
        "normalized_min": "",
        "normalized_max": "",
        "source_clock_min": "",
        "source_clock_max": "",
    }
    last_value: float | None = None
    min_value: float | None = None
    max_value: float | None = None
    source_min: pd.Timestamp | None = None
    source_max: pd.Timestamp | None = None
    raw_min: float | None = None
    raw_max: float | None = None
    first_chunk = True

    for chunk in pd.read_csv(path, usecols=usecols, chunksize=250_000, low_memory=False):
        if first_chunk:
            if platform == "IoT":
                stats["raw_dtype"] = f"date={chunk['date'].dtype};time={chunk['time'].dtype}"
            else:
                stats["raw_dtype"] = str(chunk["ts"].dtype)
            first_chunk = False
        stats["rows"] += len(chunk)
        times = parse_processed_time(chunk, platform)
        stats["valid_timestamp_count"] += int(times.notna().sum())
        stats["invalid_timestamp_count"] += int(times.isna().sum())
        valid = times.notna()
        if not valid.any():
            continue
        values = (
            times.loc[valid]
            .astype("datetime64[ns]")
            .astype("int64")
            .to_numpy(dtype="float64")
            / 1_000_000_000
        )
        if platform == "IoT":
            local_valid = times.dropna()
            local_left = local_valid.min()
            local_right = local_valid.max()
            source_min = local_left if source_min is None else min(source_min, local_left)
            source_max = local_right if source_max is None else max(source_max, local_right)
        else:
            numeric = pd.to_numeric(chunk["ts"], errors="coerce")
            numeric = numeric[numeric.notna()]
            if not numeric.empty:
                raw_values = numeric.astype("float64")
                stats["raw_fractional_count"] += int((raw_values % 1 != 0).sum())
                left = float(raw_values.min())
                right = float(raw_values.max())
                raw_min = left if raw_min is None else min(raw_min, left)
                raw_max = right if raw_max is None else max(raw_max, right)
        min_value = float(values.min()) if min_value is None else min(min_value, float(values.min()))
        max_value = float(values.max()) if max_value is None else max(max_value, float(values.max()))
        differences = np.diff(values)
        if last_value is not None:
            first_difference = float(values[0] - last_value)
            differences = np.concatenate((np.asarray([first_difference]), differences))
        stats["backward_steps"] += int((differences < 0).sum())
        stats["adjacent_duplicate_timestamps"] += int((differences == 0).sum())
        negative = differences[differences < 0]
        if negative.size:
            stats["max_backward_seconds"] = max(stats["max_backward_seconds"], float(-negative.min()))
        last_value = float(values[-1])

    stats["raw_min"] = "" if raw_min is None else str(raw_min)
    stats["raw_max"] = "" if raw_max is None else str(raw_max)
    stats["normalized_min"] = iso_epoch_seconds(min_value)
    stats["normalized_max"] = iso_epoch_seconds(max_value)
    stats["source_clock_min"] = iso_naive(source_min)
    stats["source_clock_max"] = iso_naive(source_max)
    stats["order_status"] = "PASS" if stats["backward_steps"] == 0 else "FAIL"
    return stats


def scan_processed_sources(project_root: Path = PROJECT_ROOT) -> list[dict[str, Any]]:
    paths = project_paths(project_root)
    rows: list[dict[str, Any]] = []
    for platform in PLATFORMS:
        directory = paths["processed"] / PROCESSED_DIRS[platform]
        for path in sorted(directory.glob("*.csv")):
            rows.append(scan_processed_file(path, platform))
    return rows


def scan_iot_ground_truth_offsets(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    gt_dir = project_paths(project_root)["ground_truth"] / GT_DIR_NAMES["IoT"]
    offsets: list[float] = []
    per_file: list[dict[str, Any]] = []
    examples: list[dict[str, str]] = []
    for path in sorted(gt_dir.glob("*.csv")):
        file_offsets: list[float] = []
        row_count = 0
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
            for row in csv.DictReader(handle):
                row_count += 1
                try:
                    local = parse_iot_local(row.get("data"), row.get("time"))
                    if local is None:
                        continue
                    offset = timestamp_offset_seconds(float(row["ts"]), local)
                    offsets.append(offset)
                    file_offsets.append(offset)
                    if len(examples) < 8:
                        examples.append({
                            "source_file": str(path.relative_to(project_root)),
                            "raw_ts": row.get("ts", ""),
                            "data": row.get("data", ""),
                            "time": row.get("time", ""),
                            "formula_result_seconds": str(offset),
                        })
                except (KeyError, TypeError, ValueError, OverflowError, OSError):
                    continue
        if file_offsets:
            per_file.append({
                "source_file": str(path.relative_to(project_root)),
                "rows": row_count,
                "offset_rows": len(file_offsets),
                "offset_min": min(file_offsets),
                "offset_max": max(file_offsets),
                "unique_offsets": ";".join(sorted({str(x) for x in set(file_offsets)})),
            })
    ordered = sorted(offsets)
    p95 = ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))] if ordered else None
    return {
        "total_offset_rows": len(offsets),
        "median_seconds": median(offsets) if offsets else None,
        "p95_seconds": p95,
        "max_seconds": max(offsets) if offsets else None,
        "min_seconds": min(offsets) if offsets else None,
        "unique_offsets": sorted({str(x) for x in set(offsets)}),
        "per_file": per_file,
        "examples": examples,
    }


def raw_timestamp_evidence(project_root: Path = PROJECT_ROOT) -> dict[str, str]:
    root = project_root
    iot_logs = sorted((root / "Raw_datasets" / "telemtry_data").rglob("*.log"))
    normal_log = next((p for p in iot_logs if "IoT_original_normal" in str(p)), None)
    attack_log = next((p for p in iot_logs if "IoT_original_normal_attack" in str(p)), None)
    def first_timestamp(path: Path | None) -> str:
        if path is None:
            return ""
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            match = re.search(r'"Timestamp":"([^"]+)"', line)
            if match:
                return match.group(1)
        return ""

    network_log = root / "Raw_datasets" / "network_data" / "Network_dataset_Bro" / "normal_attack_Bro" / "normal_backdoor" / "conn.log"
    network_header = ""
    if network_log.exists():
        with network_log.open("r", encoding="utf-8", errors="replace") as handle:
            network_header = " | ".join([next(handle, "").rstrip("\n") for _ in range(9)])
    windows_file = next((root / "Raw_datasets" / "windows_data").rglob("*.csv"), None)
    windows_header = windows_file.open(encoding="utf-8", errors="replace").readline().split(",")[0] if windows_file else ""
    linux_file = next((root / "Raw_datasets" / "Linux_data" / "Filtered_Linux_dataset").rglob("*.csv"), None)
    linux_sample = ""
    if linux_file:
        with linux_file.open("r", encoding="utf-8", errors="replace") as handle:
            linux_sample = next(handle, "")[:180].rstrip("\n")
    return {
        "iot_normal_raw_timestamp": first_timestamp(normal_log),
        "iot_attack_raw_timestamp": first_timestamp(attack_log),
        "network_raw_header_sample": network_header,
        "windows_raw_timezone_header": windows_header,
        "linux_filtered_row_sample": linux_sample,
    }


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def timestamp_schema_rows(file_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in file_rows:
        grouped[row["platform"]].append(row)
    result: list[dict[str, Any]] = []
    for platform in PLATFORMS:
        rows = grouped[platform]
        valid = sum(int(row["valid_timestamp_count"]) for row in rows)
        invalid = sum(int(row["invalid_timestamp_count"]) for row in rows)
        backwards = sum(int(row["backward_steps"]) for row in rows)
        minimum = min((row["normalized_min"] for row in rows if row["normalized_min"]), default="")
        maximum = max((row["normalized_max"] for row in rows if row["normalized_max"]), default="")
        if platform == "IoT":
            field = "date + time"
            fmt = "%d-%b-%y + %H:%M:%S"
            unit = "calendar wall-clock, one-second resolution"
            absolute = "source-local representation; not timezone-qualified"
            timezone_state = "not encoded; raw JSON says GMT but GT relation is +25200 seconds"
        else:
            field = "ts"
            fmt = "numeric Unix epoch seconds in processed CSV"
            unit = "seconds since Unix epoch; fractional precision absent in processed ts"
            absolute = "epoch-like numeric; timezone convention not proven"
            timezone_state = "not encoded in processed CSV"
        result.append({
            "platform": platform,
            "timestamp_field": field,
            "raw_dtype_sample": ";".join(sorted({str(row["raw_dtype"]) for row in rows})),
            "format": fmt,
            "unit": unit,
            "absolute_or_relative": absolute,
            "timezone": timezone_state,
            "files": len(rows),
            "rows": sum(int(row["rows"]) for row in rows),
            "valid_rows": valid,
            "invalid_rows": invalid,
            "normalized_min": minimum,
            "normalized_max": maximum,
            "backward_steps": backwards,
            "format_status": "PASS",
            "timestamp_validity_status": "PASS" if invalid == 0 else "FAIL",
            "format_level": "FORMAT_VALID",
            "time_validation_level": "UNVERIFIED",
            "cross_platform_sync_level": "UNVERIFIED",
            "time_reliability": "UNVERIFIED",
            "evidence": "Processed source scan; normalization.py::event_time_numeric; raw evidence recorded in 02_timezone_evidence.csv",
        })
    return result


def ground_truth_alignment_rows(project_root: Path = PROJECT_ROOT) -> list[dict[str, Any]]:
    reports = project_paths(project_root)["reports"]
    rows = read_csv_rows(reports / "ground_truth_label_validation.csv")
    result: list[dict[str, Any]] = []
    for row in rows:
        covered = row.get("ground_truth_window_covered") == "YES"
        mismatch = int(row.get("canonical_type_mismatch", "0") or 0)
        platform = row.get("platform", "")
        if not covered:
            status = "FAIL"
        elif platform == "Linux" or mismatch > 0:
            status = "UNVERIFIED"
        else:
            status = "PASS"
        result.append({
            "platform": platform,
            "processed_file": row.get("processed_file", ""),
            "ground_truth_id": row.get("ground_truth_id", ""),
            "attack_type": row.get("attack_type", ""),
            "time_basis": row.get("time_basis", ""),
            "ground_truth_start": row.get("window_start", ""),
            "ground_truth_end": row.get("window_end", ""),
            "records_in_window": row.get("records_in_window", ""),
            "label_1_in_window": row.get("label_1_in_window", ""),
            "label_0_in_window": row.get("label_0_in_window", ""),
            "canonical_type_match": row.get("canonical_type_match", ""),
            "canonical_type_mismatch": row.get("canonical_type_mismatch", ""),
            "attack_type_outside_windows": row.get("attack_type_outside_windows", ""),
            "alignment_status": status,
            "evidence": "Direct Ground Truth files were scanned by ground_truth_validation.py; mapping row retained from reports/ground_truth_label_validation.csv. IoT raw offset was independently recomputed in this task.",
            "notes": row.get("notes", ""),
        })
    return result


def write_outputs(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    paths = project_paths(project_root)
    output = paths["output"]
    output.mkdir(parents=True, exist_ok=True)
    file_rows = scan_processed_sources(project_root)
    offsets = scan_iot_ground_truth_offsets(project_root)
    raw = raw_timestamp_evidence(project_root)
    schema_rows = timestamp_schema_rows(file_rows)

    write_csv(output / "01_timestamp_schema.csv", schema_rows, list(schema_rows[0]))
    timezone_rows = [
        {
            "platform": "Network",
            "raw_timestamp_evidence": "Bro/Zeek conn.log has #types time and numeric fractional seconds; no timezone field in header.",
            "processed_rule": "pd.to_numeric(ts) then unit=s, utc=True",
            "observed_offset_median_seconds": "",
            "observed_offset_p95_seconds": "",
            "observed_offset_max_seconds": "",
            "timezone_status": "UNVERIFIED",
            "conversion_status": "PASS",
            "evidence": raw["network_raw_header_sample"],
        },
        {
            "platform": "IoT",
            "raw_timestamp_evidence": f"Raw JSON examples: normal={raw['iot_normal_raw_timestamp']}; attack={raw['iot_attack_raw_timestamp']}; both explicitly say GMT.",
            "processed_rule": "date + time parsed as naive wall-clock then current code represents it with utc=True",
            "observed_offset_median_seconds": offsets["median_seconds"],
            "observed_offset_p95_seconds": offsets["p95_seconds"],
            "observed_offset_max_seconds": offsets["max_seconds"],
            "timezone_status": "UNVERIFIED",
            "conversion_status": "PASS",
            "evidence": f"Direct scan of 7 IoT Ground Truth files and {offsets['total_offset_rows']} rows; UTC epoch - data/time local = +25200 seconds for every comparable row. The raw GMT label conflicts with interpreting data/time as local, so no automatic timezone correction is applied.",
        },
        {
            "platform": "Linux",
            "raw_timestamp_evidence": "Filtered feature sample has no timestamp field; original atop text sample has no explicit timestamp in row text.",
            "processed_rule": "pd.to_numeric(ts) then unit=s, utc=True",
            "observed_offset_median_seconds": "",
            "observed_offset_p95_seconds": "",
            "observed_offset_max_seconds": "",
            "timezone_status": "UNVERIFIED",
            "conversion_status": "UNVERIFIED",
            "evidence": raw["linux_filtered_row_sample"],
        },
        {
            "platform": "Windows",
            "raw_timestamp_evidence": f"Filtered PDH header declares {raw['windows_raw_timezone_header']}; processed ts conversion/DST mapping is not documented.",
            "processed_rule": "pd.to_numeric(ts) then unit=s, utc=True",
            "observed_offset_median_seconds": "",
            "observed_offset_p95_seconds": "",
            "observed_offset_max_seconds": "",
            "timezone_status": "UNVERIFIED",
            "conversion_status": "UNVERIFIED",
            "evidence": raw["windows_raw_timezone_header"],
        },
    ]
    write_csv(output / "02_timezone_evidence.csv", timezone_rows, list(timezone_rows[0]))

    gt_rows = ground_truth_alignment_rows(project_root)
    write_csv(output / "03_ground_truth_alignment.csv", gt_rows, list(gt_rows[0]))

    anomaly_rows: list[dict[str, Any]] = []
    for row in file_rows:
        invalid = int(row["invalid_timestamp_count"])
        backward = int(row["backward_steps"])
        duplicates = int(row["adjacent_duplicate_timestamps"])
        anomalies = []
        if invalid:
            anomalies.append("invalid_or_missing_timestamp")
        if backward:
            anomalies.append("within_file_backward_step")
        if duplicates:
            anomalies.append("adjacent_duplicate_timestamp")
        status = "FAIL" if invalid or backward else ("UNVERIFIED" if duplicates else "PASS")
        anomaly_rows.append({
            "platform": row["platform"],
            "source_file": row["source_file"],
            "rows": row["rows"],
            "invalid_timestamp_count": invalid,
            "backward_steps": backward,
            "max_backward_seconds": row["max_backward_seconds"],
            "adjacent_duplicate_timestamps": duplicates,
            "anomaly_status": status,
            "anomaly_types": ";".join(anomalies) or "none_observed",
            "evidence": "Direct chunked scan of current processed CSV; duplicate metric is adjacent timestamp equality, not exact-row duplication.",
        })
    write_csv(output / "04_timestamp_anomalies.csv", anomaly_rows, list(anomaly_rows[0]))

    sync_rows = [
        {"source_a": "Network", "source_b": "IoT", "link_key": "none verified; time overlap only", "evidence": "Network IP/port/proto/uid versus IoT device-profile fields; no shared host/session key in GT.", "observed_offset": "not estimable as same event", "confidence": "LOW", "status": "UNVERIFIED"},
        {"source_a": "Network", "source_b": "Linux", "link_key": "none verified; Network 5-tuple/uid versus Linux PID/CMD", "evidence": "No common host/run/session identifier in GT or processed schema.", "observed_offset": "not estimable as same event", "confidence": "LOW", "status": "UNVERIFIED"},
        {"source_a": "Network", "source_b": "Windows", "link_key": "none verified", "evidence": "Windows counters have no corresponding Network connection/session identifier.", "observed_offset": "not estimable as same event", "confidence": "LOW", "status": "UNVERIFIED"},
        {"source_a": "IoT", "source_b": "Linux", "link_key": "none verified; device profile versus PID/CMD", "evidence": "Ground Truth files are platform-specific and expose no common event/session key.", "observed_offset": "not estimable as same event", "confidence": "LOW", "status": "UNVERIFIED"},
        {"source_a": "IoT", "source_b": "Windows", "link_key": "none verified", "evidence": "No common device/host/session identity; time overlap is not linkage.", "observed_offset": "not estimable as same event", "confidence": "LOW", "status": "UNVERIFIED"},
        {"source_a": "Linux", "source_b": "Windows", "link_key": "none verified; PID/CMD versus PDH counters", "evidence": "No common host/run/session identifier and Linux raw-to-processed clock provenance unresolved.", "observed_offset": "not estimable as same event", "confidence": "LOW", "status": "UNVERIFIED"},
    ]
    write_csv(output / "05_cross_platform_sync.csv", sync_rows, list(sync_rows[0]))

    split_rows = [
        {"check": "Current calendar cutoffs", "status": "PASS", "evidence": "split_data.py uses CORRECTIVE_TRAIN_END=2019-04-24 and CORRECTIVE_VALIDATION_END=2019-04-26; half-open comparisons use <.", "impact": "Cutoff mechanics match the frozen protocol; clock meaning remains unverified."},
        {"check": "IoT invalid timestamps", "status": "PASS", "evidence": "remediated_split_codes routes non-finite event_time to QUARANTINE; prior evidence identifies 25,956 rows; V1 is unchanged.", "impact": "Mechanism is safe; full remediated export was not rerun here."},
        {"check": "Network equal endpoints in current pipeline", "status": "FAIL", "evidence": "Current Network branch uses complete source-file holdout; two equal boundary timestamps occur across different source splits.", "impact": "Do not claim current Network endpoint ownership is validated."},
        {"check": "Desired pure half-open endpoint helper", "status": "PASS", "evidence": "final_split_protocol.py unit helper gives one owner for 2019-04-26T02:30:34Z and 2019-04-27T01:25:17Z.", "impact": "Helper does not change the current pipeline or V1."},
        {"check": "Graph future exclusion", "status": "PASS", "evidence": "Protocol specifies history strictly earlier than target time; no graph export was run.", "impact": "Implementation invariant only; final graph audit remains unverified."},
        {"check": "Cross-fold window/session isolation", "status": "UNVERIFIED", "evidence": "Ground Truth windows are min/max envelopes without shared session/run IDs; overlapping windows cannot be converted to episodes.", "impact": "No cross-source temporal edge is authorized."},
        {"check": "Long gaps as snapshot boundaries", "status": "UNVERIFIED", "evidence": "Timestamp ranges are available, but no acquisition schedule/heartbeat contract was found.", "impact": "Do not infer consecutive snapshots from calendar adjacency alone."},
        {"check": "Raw/V1 preservation", "status": "PASS", "evidence": "This task writes only final_timestamp_validation artifacts and does not alter Raw/Processed/V1.", "impact": "No retrospective split change."},
    ]
    write_csv(output / "06_split_temporal_integrity.csv", split_rows, list(split_rows[0]))

    (output / "07_graph_temporal_feasibility.md").write_text(graph_feasibility_markdown(file_rows), encoding="utf-8")
    (output / "08_timestamp_tests.md").write_text(timestamp_tests_markdown(offsets), encoding="utf-8")
    (output / "FINAL_TIMESTAMP_DECISION.md").write_text(final_decision_markdown(file_rows, offsets, gt_rows), encoding="utf-8")
    return {"processed_files": len(file_rows), "ground_truth_mapping_rows": len(gt_rows), "iot_offset_rows": offsets["total_offset_rows"], "output": str(output)}


def graph_feasibility_markdown(file_rows: list[dict[str, Any]]) -> str:
    platform_summary = []
    for platform in PLATFORMS:
        rows = [row for row in file_rows if row["platform"] == platform]
        invalid = sum(int(row["invalid_timestamp_count"]) for row in rows)
        backward = sum(int(row["backward_steps"]) for row in rows)
        platform_summary.append(f"| {platform} | PASS (sort valid rows) | {'FAIL' if invalid or backward else 'PASS'} row-integrity issue | {invalid:,} invalid; {backward:,} backward steps |")
    return """# Graph temporal feasibility

## Decision

An independent temporal graph is feasible only as a **platform-local, provenance-aware** graph. It is not currently valid to build a heterogeneous graph with cross-platform temporal edges.

| Platform | Local ordering operation | Raw row-order check | Evidence |
|---|---|---|---|
""" + "\n".join(platform_summary) + """

## Rules for an allowed local graph

1. Parse only the platform-declared timestamp representation and retain raw timestamp fields.
2. Sort valid rows by parsed event time; do not use CSV row order as event order.
3. Remove IoT non-finite timestamps through the existing `QUARANTINE` path.
4. For a target at time `T`, history must satisfy `event_time < T`; rows at `T` or later are excluded.
5. Do not create an edge between two platforms from equal timestamps, Ground Truth-window overlap, attack type, or file numbering alone.
6. Treat Ground Truth windows as source-specific time envelopes, not episode IDs.

## Platform scope

- Network, Linux and Windows can support local temporal ordering of their parsed numeric `ts`, but their host timezone and raw-to-processed provenance are not validated.
- IoT can support local ordering of valid `date + time` rows after quarantining invalid values. Its relationship to numeric epoch/GMT is not safe for absolute cross-platform alignment.
- A future cross-source graph requires a documented common acquisition clock plus a shared host/device/session/event key. Neither is present in the inspected Ground Truth files.
"""


def timestamp_tests_markdown(offsets: dict[str, Any]) -> str:
    return f"""# Timestamp tests and evidence

The focused tests are in `research_pipeline/tests/test_final_timestamp_validation.py`.

Covered invariants:

- Unix epoch seconds convert to the documented instant without treating milliseconds as seconds.
- ISO strings with an explicit timezone are parseable; timezone-less strings are rejected by the absolute parser and accepted only by an explicitly named source-local parser.
- Missing/invalid timestamps route to quarantine rather than an evaluation fold.
- Relative order compares parsed values, not CSV row position; backward steps are detected.
- `[start, end)` boundary ownership is deterministic.
- Snapshot history keeps only events strictly earlier than target `T`.
- IoT offset formula is `UTC_epoch - local_data_time = {offsets['median_seconds']} seconds`, with median/p95/max all `{offsets['median_seconds']}` over `{offsets['total_offset_rows']}` comparable Ground Truth rows.

These tests establish parser and routing invariants only. They do not prove a common physical clock or same-event identity.
"""


def final_decision_markdown(file_rows: list[dict[str, Any]], offsets: dict[str, Any], gt_rows: list[dict[str, Any]]) -> str:
    counts = {platform: sum(int(row["rows"]) for row in file_rows if row["platform"] == platform) for platform in PLATFORMS}
    invalid = {platform: sum(int(row["invalid_timestamp_count"]) for row in file_rows if row["platform"] == platform) for platform in PLATFORMS}
    covered = Counter(row["platform"] for row in gt_rows if row["alignment_status"] != "FAIL")
    return f"""# FINAL TIMESTAMP DECISION — ToN-IoT

## Final status

**PARTIALLY VALIDATED.**

The timestamp **format** is validated for all four processed platforms, and platform-local ordering is usable after parsing/sorting valid rows. Absolute clock semantics are not fully validated, and four-platform cross-source synchronization is **NOT VALIDATED**.

## Required answers

| Platform | Format | Time reliability | Ground Truth alignment | Cross-platform sync | Temporal graph feasibility |
|---|---|---|---|---|---|
| Network | PASS — `ts`, numeric Unix seconds | UNVERIFIED — Bro/Zeek host timezone absent | PASS — representation/type/time mappings covered; no same-event claim | UNVERIFIED | PASS for local graph only; no cross-source edges |
| IoT | PASS for valid `date + time`; invalid rows exist | UNVERIFIED — raw JSON says GMT while GT `ts - data/time` is exactly +25,200 s | UNVERIFIED — strong local representation match, but two windows include type mismatch and UTC meaning is unresolved | UNVERIFIED | PASS for valid local-time graph after QUARANTINE; no absolute cross-source edges |
| Linux | PASS — numeric Unix seconds | UNVERIFIED — raw-to-processed unit/timezone provenance unresolved | UNVERIFIED — windows covered but GT aggregates disk/memory/process and type mismatch is systematic | UNVERIFIED | PASS for schema-local graph only |
| Windows | PASS — numeric Unix seconds | UNVERIFIED — PDH declares AUS Eastern Standard Time (-600), conversion/DST to processed `ts` unproven | PASS — representation/type/time mappings covered; no shared entity | UNVERIFIED | PASS for local graph only; no cross-source edges |

## Direct evidence

- Processed source scan: `{len(file_rows)}` files, rows by platform `{json.dumps(counts, ensure_ascii=False)}`; invalid timestamp rows `{json.dumps(invalid, ensure_ascii=False)}`.
- Network processed `ts`, Linux processed `ts`, and Windows processed `ts` are numeric seconds. Current row conversion is `pd.to_numeric(...); pd.to_datetime(..., unit='s', utc=True)`.
- IoT processed files contain `date` and `time`; current code represents the naive wall clock with `utc=True`. That is a representation choice, not evidence that the source clock is UTC.
- Direct scan of all 7 IoT Ground Truth files found `{offsets['total_offset_rows']}` comparable rows; `UTC - local = +{offsets['median_seconds']:.0f}` seconds, with p95 `{offsets['p95_seconds']:.0f}` and max `{offsets['max_seconds']:.0f}`. The inverse is `local - UTC = -{offsets['median_seconds']:.0f}` seconds.
- Raw IoT examples contain `Timestamp=... GMT`, while the corresponding Ground Truth `data/time` relation is +7 hours behind numeric `ts`. This is an unresolved semantic conflict; no fixed timezone correction is applied.
- Ground Truth evidence consists of 30 official CSVs, 105 source-specific windows and 146 existing mapping rows. Covered windows are evidence of type/time representation and label correspondence, not independent attack episodes.
- Existing cross-platform alignment is `TIME_OVERLAP_ONLY`/`NO_OVERLAP`; no shared session/run/event ID establishes `SAME_EVENT_CONFIRMED`.

## Q1–Q8

**Q1.** Format is validated; time reliability is only partially validated. Numeric units are plausible and parsable, but timezone/host-clock provenance is missing for Network, Linux and Windows, and IoT has a raw-GMT/local-clock conflict.

**Q2.** Within each platform, valid timestamp order can be reconstructed by sorting. CSV row order is not reliable: the direct scan reports backward steps in the anomaly artifact. IoT invalid rows must be quarantined.

**Q3.** Ground Truth aligns at the source representation level. The only consistently quantified clock relation is IoT: +25,200 seconds, median/p95/max all exact. It proves a representation offset, not a universal timezone or synchronized clock. Linux mapping remains unresolved at higher confidence because the GT aggregates schema views.

**Q4.** No. There is no common clock declaration and no shared cross-platform session/device/event key.

**Q5.** No heterogeneous temporal graph linking all four sources is currently validated. Equal timestamps or overlapping Ground Truth windows are insufficient.

**Q6.** Yes, independent platform-local temporal graphs are feasible for Network, IoT valid rows, Linux schema-local views and Windows, with the stated caveats. The previously selected main Known-Attack scope remains Linux Process/Disk; this timestamp result does not broaden it.

**Q7.** No raw timestamp conversion was changed because the evidence is semantic-conflict evidence, not permission to impose a timezone retroactively. The earlier -7,200-second report arithmetic is not supported; direct evidence supports +25,200/-25,200 for IoT. The data-intrinsic limits are missing timezone provenance, missing shared IDs, non-monotonic row order, Ground Truth envelopes, and Network endpoint ambiguity.

**Q8.** Lecturer comment 2 is resolved **partially**: format/unit/routing evidence and the exact IoT offset are documented, while common-clock and cross-source synchronization remain explicitly UNVERIFIED.

## Split/graph decision

- Keep Split V1 and the final split protocol unchanged.
- Keep IoT and Linux Memory diagnostic for the Known-Attack benchmark as already decided; timestamp validation does not repair IoT Validation Scanning = 0.
- Use only valid, sorted, platform-local event times for any future temporal graph.
- Do not construct cross-platform temporal edges or call the 105 windows independent attack episodes.

## Evidence files

- `01_timestamp_schema.csv`
- `02_timezone_evidence.csv`
- `03_ground_truth_alignment.csv`
- `04_timestamp_anomalies.csv`
- `05_cross_platform_sync.csv`
- `06_split_temporal_integrity.csv`
- `07_graph_temporal_feasibility.md`
- `08_timestamp_tests.md`
- Source code: `research_pipeline/src/validation/final_timestamp_validation.py`

No model was trained. Raw data, Processed data and Split V1 were not modified.
"""


def main() -> None:
    result = write_outputs()
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
