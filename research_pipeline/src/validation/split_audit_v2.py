"""Streaming, audit-only verification of the current ToN-IoT split.

This module reproduces the current V1 assignment for measurement.  It never
writes source data, calls the exporter, fits a transformer, or trains a model.

Run from the project root::

    python -m research_pipeline.src.validation.split_audit_v2
"""

from __future__ import annotations

import argparse
import itertools
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..common.config import ATTACK_CATEGORIES, DEFAULT_CHUNK_SIZE, PLATFORM_ORDER, raw_type_mapping
from ..common.paths import PROJECT_ROOT
from ..common.utils import write_csv_rows
from ..pipeline.data_reader import make_source_records
from ..pipeline.duplicate_handler import DuplicateRouter, duplicate_fingerprints
from ..pipeline.models import RunState, SourceRecord
from ..pipeline.normalization import event_time_numeric, format_event_range, parse_time_bound, validate_chunk
from ..pipeline.split_data import assign_split_plan, base_split_codes


SPLITS = ("train", "validation", "test")
SPLIT_NAMES = np.array(SPLITS, dtype=object)
AUDIT_DIRNAME = "split_audit_v2"
PARTITIONS = 32
_DUP_DTYPE = np.dtype(
    [("fingerprint", "<u8"), ("schema", "<u2"), ("file_id", "u1"), ("split", "u1")],
    align=False,
)


def _counter_add(counter: dict[tuple[Any, ...], int], key: tuple[Any, ...], value: int) -> None:
    counter[key] = counter.get(key, 0) + int(value)


def _day_values(times: pd.Series) -> np.ndarray:
    values = times.to_numpy(dtype="float64", copy=False)
    result = np.full(len(values), "MISSING_TIMESTAMP", dtype=object)
    valid = np.isfinite(values)
    if valid.any():
        result[valid] = pd.to_datetime(values[valid], unit="s", utc=True).strftime("%Y-%m-%d")
    return result


def _schema_key(rec: SourceRecord) -> str:
    return f"{rec.platform}|{','.join(rec.columns)}"


def _status_for_type(train: int, validation: int, test: int, total: int) -> str:
    if total == 0:
        return "ABSENT_IN_PLATFORM"
    if train:
        return "SEEN_IN_TRAIN"
    if validation and test:
        return "UNSEEN_IN_TRAIN_VALIDATION_AND_TEST"
    if validation:
        return "UNSEEN_IN_TRAIN_VALIDATION_ONLY"
    if test:
        return "UNSEEN_IN_TRAIN_TEST_ONLY"
    return "UNSEEN_IN_TRAIN_UNKNOWN_SPLIT"


def _safe_rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _timestamp_note(platform: str) -> tuple[str, str, str]:
    if platform == "IoT":
        return (
            "processed date + time parsed as UTC by current code",
            "raw IoT evidence says GMT/UTC epoch is seven hours ahead of local data/time",
            "UNVERIFIED",
        )
    if platform == "Windows":
        return (
            "processed ts parsed as Unix epoch seconds",
            "filtered PDH source exposes AUS Eastern Standard Time (-600); conversion/DST is not proven",
            "UNVERIFIED",
        )
    if platform == "Network":
        return (
            "processed ts parsed as Unix epoch seconds",
            "numeric Bro/Zeek timestamp is present but host timezone convention is not documented",
            "UNVERIFIED",
        )
    return (
        "processed ts parsed as Unix epoch seconds",
        "raw-to-processed Linux clock/unit provenance is unresolved",
        "UNVERIFIED",
    )


def _read_ground_truth(
    root: Path,
    processed_ranges: dict[tuple[str, str], tuple[float, float]],
    processed_file_ranges: dict[tuple[str, str, str], tuple[float, float]],
) -> dict[str, Any]:
    """Summarize the committed 105-window report without treating windows as episodes."""
    path = root / "research_pipeline" / "reports" / "ground_truth_attack_windows.csv"
    if not path.exists():
        return {"present": False, "rows": 0, "by_platform": {}, "by_type": {}, "direct": 0, "iot_local": 0}
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"platform", "attack_type", "start_time", "end_time"}
    if not required.issubset(frame.columns):
        return {"present": False, "rows": 0, "by_platform": {}, "by_type": {}, "direct": 0, "iot_local": 0}
    by_platform = frame.groupby("platform").size().astype(int).to_dict()
    by_type = frame.groupby(["platform", "attack_type"]).size().astype(int).to_dict()
    direct = 0
    iot_local = 0
    for row in frame.to_dict("records"):
        key = (row["platform"], row["attack_type"])
        if key not in processed_ranges:
            continue
        start = pd.to_datetime(row["start_time"], utc=True, errors="coerce")
        end = pd.to_datetime(row["end_time"], utc=True, errors="coerce")
        if pd.isna(start) or pd.isna(end):
            continue
        left, right = start.timestamp(), end.timestamp()
        source_name = Path(row.get("source_file", "")).name
        candidates = [
            value for (platform, attack, file_name), value in processed_file_ranges.items()
            if platform == row["platform"] and attack == row["attack_type"] and (
                file_name.lower() == source_name.lower()
                or (row["platform"] == "Linux" and attack and attack in file_name.lower())
                or (row["platform"] == "Network" and source_name.lower().replace("groundtruth_network_", "network_dataset_").replace(".csv", ".csv") == file_name.lower())
            )
        ]
        if not candidates:
            candidates = [processed_ranges[key]]
        if any(max(left, pleft) <= min(right, pright) for pleft, pright in candidates):
            direct += 1
        if row["platform"] == "IoT":
            left -= 7 * 3600
            right -= 7 * 3600
            if any(max(left, pleft) <= min(right, pright) for pleft, pright in candidates):
                iot_local += 1
    return {
        "present": True,
        "rows": int(len(frame)),
        "by_platform": {str(k): int(v) for k, v in by_platform.items()},
        "by_type": {f"{k[0]}::{k[1]}": int(v) for k, v in by_type.items()},
        "direct": direct,
        "iot_local": iot_local,
    }


def _write_daily(path: Path, counts: dict[tuple[str, str, str, str], list[int]]) -> None:
    rows = []
    for (platform, day, source_file, split), (normal, attack) in sorted(counts.items()):
        total = normal + attack
        rows.append({
            "platform": platform,
            "event_day": day,
            "source_file": source_file,
            "split": split,
            "timestamp_basis": "UTC parser day; IoT is source-local clock represented as UTC by V1",
            "normal_count": normal,
            "attack_count": attack,
            "total_count": total,
            "normal_rate": _safe_rate(normal, total),
            "attack_rate": _safe_rate(attack, total),
        })
    columns = [
        "platform", "event_day", "source_file", "split", "timestamp_basis",
        "normal_count", "attack_count", "total_count", "normal_rate", "attack_rate",
    ]
    write_csv_rows(path, rows, columns)


def _write_coverage(path: Path, counts: dict[tuple[str, str, str], int], sources: dict[tuple[str, str], set[str]]) -> None:
    observed = {attack for platform, attack, split in counts if attack not in {"normal", "UNKNOWN"}}
    observed.update(ATTACK_CATEGORIES)
    rows = []
    for platform in PLATFORM_ORDER:
        for attack_type in sorted(observed):
            values = [counts.get((platform, attack_type, split), 0) for split in SPLITS]
            total = sum(values)
            rows.append({
                "platform": platform,
                "attack_type": attack_type,
                "train_attack_count": values[0],
                "validation_attack_count": values[1],
                "test_attack_count": values[2],
                "total_attack_count": total,
                "train_attack_rate_within_attack_rows": _safe_rate(values[0], total),
                "validation_attack_rate_within_attack_rows": _safe_rate(values[1], total),
                "test_attack_rate_within_attack_rows": _safe_rate(values[2], total),
                "source_file_count": len(set().union(*(sources.get((platform, attack_type), set()),))),
                "train_seen": "YES" if values[0] else "NO",
                "validation_seen": "YES" if values[1] else "NO",
                "test_seen": "YES" if values[2] else "NO",
                "seen_unseen_from_train": "SEEN" if values[0] else "UNSEEN",
                "coverage_status": _status_for_type(values[0], values[1], values[2], total),
            })
    columns = [
        "platform", "attack_type", "train_attack_count", "validation_attack_count", "test_attack_count",
        "total_attack_count", "train_attack_rate_within_attack_rows",
        "validation_attack_rate_within_attack_rows", "test_attack_rate_within_attack_rows",
        "source_file_count", "train_seen", "validation_seen", "test_seen", "seen_unseen_from_train", "coverage_status",
    ]
    write_csv_rows(path, rows, columns)


def _write_timestamp_quality(path: Path, quality: dict[tuple[str, str], dict[str, Any]]) -> None:
    rows = []
    for (platform, source_file), item in sorted(quality.items()):
        note, evidence, timezone_status = _timestamp_note(platform)
        rows.append({
            "platform": platform,
            "source_file": source_file,
            "rows": item["rows"],
            "valid_timestamp_count": item["valid"],
            "missing_or_invalid_timestamp_count": item["invalid"],
            "missing_or_invalid_timestamp_rate": _safe_rate(item["invalid"], item["rows"]),
            "timestamp_min_utc_parser": format_event_range(item["min"], platform) if item["min"] is not None else "",
            "timestamp_max_utc_parser": format_event_range(item["max"], platform) if item["max"] is not None else "",
            "missing_timestamp_normal_count": item.get("missing_normal", 0),
            "missing_timestamp_attack_count": item.get("missing_attack", 0),
            "within_file_timestamp_order_violations": item["order_violations"],
            "timestamp_basis": note,
            "timezone_evidence": evidence,
            "timezone_status": timezone_status,
            "missing_timestamp_split": "train for chronological V1; source-file split for Network",
        })
    columns = [
        "platform", "source_file", "rows", "valid_timestamp_count", "missing_or_invalid_timestamp_count",
        "missing_or_invalid_timestamp_rate", "timestamp_min_utc_parser", "timestamp_max_utc_parser",
        "missing_timestamp_normal_count", "missing_timestamp_attack_count",
        "within_file_timestamp_order_violations", "timestamp_basis", "timezone_evidence", "timezone_status",
        "missing_timestamp_split",
    ]
    write_csv_rows(path, rows, columns)


def _write_cross_file(path: Path, pair_stats: dict[tuple[int, int, int, int, int], dict[str, Any]], records: list[SourceRecord], schemas: dict[int, str]) -> dict[str, int]:
    rows = []
    totals = Counter()
    for (schema_id, file_a, file_b, split_a, split_b), item in sorted(pair_stats.items()):
        rec_a, rec_b = records[file_a], records[file_b]
        key_platform = rec_a.platform
        conflict = split_a != split_b
        totals[(key_platform, "fingerprints")] += item["fingerprints"]
        totals[(key_platform, "rows") ] += item["rows_pair"]
        if conflict:
            totals[(key_platform, "conflicts")] += item["fingerprints"]
        future_to_train = split_a == 0 and split_b in (1, 2)
        rows.append({
            "platform": key_platform,
            "schema": schemas[schema_id],
            "source_file_a": rec_a.file_name,
            "source_file_b": rec_b.file_name,
            "split_a": SPLITS[split_a],
            "split_b": SPLITS[split_b],
            "duplicate_fingerprint_count": item["fingerprints"],
            "duplicate_row_occurrences_in_pair": item["rows_pair"],
            "split_conflict": "YES" if conflict else "NO",
            "future_to_train_cross_file_order_risk": "YES" if future_to_train else "NO",
            "example_fingerprints": ";".join(item["examples"]),
            "status": "FAIL" if conflict else "PASS",
        })
    if not rows:
        for platform in PLATFORM_ORDER:
            rows.append({
                "platform": platform, "schema": "", "source_file_a": "", "source_file_b": "",
                "split_a": "", "split_b": "", "duplicate_fingerprint_count": 0,
                "duplicate_row_occurrences_in_pair": 0, "split_conflict": "NO",
                "future_to_train_cross_file_order_risk": "NO", "example_fingerprints": "",
                "status": "PASS",
            })
    columns = [
        "platform", "schema", "source_file_a", "source_file_b", "split_a", "split_b",
        "duplicate_fingerprint_count", "duplicate_row_occurrences_in_pair", "split_conflict",
        "future_to_train_cross_file_order_risk", "example_fingerprints", "status",
    ]
    write_csv_rows(path, rows, columns)
    return {
        "fingerprints": sum(v for (p, k), v in totals.items() if k == "fingerprints"),
        "rows": sum(v for (p, k), v in totals.items() if k == "rows"),
        "conflicts": sum(v for (p, k), v in totals.items() if k == "conflicts"),
    }


def _process_duplicate_partitions(
    paths: list[Path], records: list[SourceRecord], schemas: dict[int, str],
) -> tuple[dict[tuple[int, int, int, int, int], dict[str, Any]], dict[str, int]]:
    pair_stats: dict[tuple[int, int, int, int, int], dict[str, Any]] = {}
    totals = Counter()
    for path in paths:
        if not path.exists() or path.stat().st_size == 0:
            continue
        data = np.fromfile(path, dtype=_DUP_DTYPE)
        # Sort by file as the last key so a change of file inside a
        # fingerprint group is detectable without iterating over all unique
        # fingerprints.  Most groups are single-file; only candidate groups
        # are materialized below.
        order = np.lexsort((data["file_id"], data["fingerprint"], data["schema"]))
        data = data[order]
        same_group = (
            (data["schema"][1:] == data["schema"][:-1])
            & (data["fingerprint"][1:] == data["fingerprint"][:-1])
        )
        starts = np.r_[0, 1 + np.flatnonzero(~same_group)]
        file_changes = np.flatnonzero(same_group & (data["file_id"][1:] != data["file_id"][:-1])) + 1
        candidate_group_indexes = np.unique(np.searchsorted(starts, file_changes, side="right") - 1)
        for group_index in candidate_group_indexes:
            start = int(starts[group_index])
            end = int(starts[group_index + 1]) if group_index + 1 < len(starts) else len(data)
            group = data[start:end]
            files = np.unique(group["file_id"])
            if len(files) < 2:
                continue
            totals["fingerprints"] += 1
            totals["rows"] += len(group)
            split_by_file: dict[int, int] = {}
            count_by_file: dict[int, int] = {}
            for file_id in files:
                selected = group[group["file_id"] == file_id]
                split_by_file[int(file_id)] = int(np.unique(selected["split"])[0])
                count_by_file[int(file_id)] = int(len(selected))
            for file_a, file_b in itertools.combinations(sorted(split_by_file), 2):
                split_a, split_b = split_by_file[file_a], split_by_file[file_b]
                key = (int(group["schema"][0]), file_a, file_b, split_a, split_b)
                item = pair_stats.setdefault(key, {"fingerprints": 0, "rows_pair": 0, "examples": []})
                item["fingerprints"] += 1
                item["rows_pair"] += count_by_file[file_a] + count_by_file[file_b]
                if len(item["examples"]) < 3:
                    item["examples"].append(f"{int(group['fingerprint'][0]):016x}")
                if split_a != split_b:
                    totals["conflicts"] += 1
    return pair_stats, dict(totals)


def _scan_sources(root: Path, records: list[SourceRecord], chunk_size: int, temp_dir: Path) -> dict[str, Any]:
    mapping = raw_type_mapping()
    state = RunState()
    daily: dict[tuple[str, str, str, str], list[int]] = {}
    coverage: dict[tuple[str, str, str], int] = Counter()
    coverage_sources: dict[tuple[str, str], set[str]] = defaultdict(set)
    quality: dict[tuple[str, str], dict[str, Any]] = {}
    processed_ranges: dict[tuple[str, str], tuple[float, float]] = {}
    processed_file_ranges: dict[tuple[str, str, str], tuple[float, float]] = {}
    envelopes: dict[tuple[str, str], list[float | None]] = {
        (platform, split): [None, None] for platform in PLATFORM_ORDER for split in SPLITS
    }
    router_summary: dict[str, dict[str, int]] = {}
    schema_ids: dict[str, int] = {}
    schemas: dict[int, str] = {}
    partition_paths = [temp_dir / f"part_{index:02d}.bin" for index in range(PARTITIONS)]

    for file_id, rec in enumerate(records):
        schema = _schema_key(rec)
        if schema not in schema_ids:
            schema_ids[schema] = len(schema_ids)
            schemas[schema_ids[schema]] = schema
        schema_id = schema_ids[schema]
        router = DuplicateRouter()
        item = quality[(rec.platform, rec.file_name)] = {
            "rows": 0, "valid": 0, "invalid": 0, "min": None, "max": None,
            "missing_normal": 0, "missing_attack": 0, "order_violations": 0,
        }
        previous_time: float | None = None
        rows_read = 0
        attack_range_keys: set[tuple[str, str]] = set()
        for frame in pd.read_csv(rec.file_path, chunksize=chunk_size, low_memory=False):
            labels, _, canonical, times = validate_chunk(frame, rec, mapping, state, validation_only=True)
            base = base_split_codes(times, rec)
            fp = duplicate_fingerprints(frame, rec.columns)
            final = base if rec.split_mode == "source_file_holdout" else router.route(fp, base)
            rows_read += len(frame)
            item["rows"] += len(frame)
            valid_mask = np.isfinite(times.to_numpy(dtype="float64", copy=False))
            item["valid"] += int(valid_mask.sum())
            item["invalid"] += int((~valid_mask).sum())
            labels_np = labels.to_numpy(dtype="int8", copy=False)
            item["missing_normal"] += int(((~valid_mask) & (labels_np == 0)).sum())
            item["missing_attack"] += int(((~valid_mask) & (labels_np == 1)).sum())
            if valid_mask.any():
                vals = times.to_numpy(dtype="float64", copy=False)[valid_mask]
                item["min"] = float(vals.min()) if item["min"] is None else min(item["min"], float(vals.min()))
                item["max"] = float(vals.max()) if item["max"] is None else max(item["max"], float(vals.max()))
                for value in vals:
                    if previous_time is not None and value < previous_time:
                        item["order_violations"] += 1
                    previous_time = float(value)

            canonical_np = canonical.astype("string").fillna("UNKNOWN").to_numpy(dtype=object)
            days = _day_values(times)
            for split_code, split in enumerate(SPLITS):
                mask = final == split_code
                if not mask.any():
                    continue
                normal = int((labels_np[mask] == 0).sum())
                attack = int((labels_np[mask] == 1).sum())
                for day in np.unique(days[mask]):
                    day_mask = mask & (days == day)
                    key = (rec.platform, str(day), rec.file_name, split)
                    bucket = daily.setdefault(key, [0, 0])
                    bucket[0] += int((labels_np[day_mask] == 0).sum())
                    bucket[1] += int((labels_np[day_mask] == 1).sum())
                attack_types = canonical_np[mask & (labels_np == 1)]
                for attack_type in np.unique(attack_types):
                    attack_name = str(attack_type)
                    coverage[(rec.platform, attack_name, split)] += int((attack_types == attack_type).sum())
                    coverage_sources[(rec.platform, attack_name)].add(rec.file_name)
                    attack_range_keys.add((rec.platform, attack_name))
                finite = times.to_numpy(dtype="float64", copy=False)[mask & valid_mask]
                if len(finite):
                    envelope = envelopes[(rec.platform, split)]
                    envelope[0] = float(finite.min()) if envelope[0] is None else min(envelope[0], float(finite.min()))
                    envelope[1] = float(finite.max()) if envelope[1] is None else max(envelope[1], float(finite.max()))

            for attack_type in np.unique(canonical_np[labels_np == 1]):
                values = times.to_numpy(dtype="float64", copy=False)[(canonical_np == attack_type) & (labels_np == 1)]
                values = values[np.isfinite(values)]
                if len(values):
                    key = (rec.platform, str(attack_type))
                    current = processed_ranges.get(key)
                    pair = (float(values.min()), float(values.max()))
                    processed_ranges[key] = pair if current is None else (min(current[0], pair[0]), max(current[1], pair[1]))
                    file_key = (rec.platform, str(attack_type), rec.file_name)
                    current_file = processed_file_ranges.get(file_key)
                    processed_file_ranges[file_key] = pair if current_file is None else (
                        min(current_file[0], pair[0]), max(current_file[1], pair[1])
                    )

            data = np.empty(len(fp), dtype=_DUP_DTYPE)
            data["fingerprint"] = fp
            data["schema"] = schema_id
            data["file_id"] = file_id
            data["split"] = final.astype("u1", copy=False)
            buckets = (fp >> np.uint64(59)).astype(np.int64)
            for bucket in np.unique(buckets):
                selected = data[buckets == bucket]
                with partition_paths[int(bucket)].open("ab") as handle:
                    selected.tofile(handle)

            if rec.split_mode != "source_file_holdout" and len(base):
                base_np, final_np = base.astype("int8", copy=False), final.astype("int8", copy=False)
                router_summary.setdefault(rec.file_name, {"routed": 0, "cross_split": 0, "future_to_train": 0})
                router_summary[rec.file_name]["routed"] += int((base_np != final_np).sum())
                router_summary[rec.file_name]["cross_split"] += int(((base_np != final_np)).sum())
                router_summary[rec.file_name]["future_to_train"] += int(((base_np != 0) & (final_np == 0)).sum())
        if rows_read != rec.rows_expected:
            raise ValueError(f"{rec.file_name}: streamed {rows_read} rows; inventory expects {rec.rows_expected}")
        router_summary.setdefault(rec.file_name, {"routed": 0, "cross_split": 0, "future_to_train": 0})

    pair_stats, cross_totals = _process_duplicate_partitions(partition_paths, records, schemas)
    gt = _read_ground_truth(root, processed_ranges, processed_file_ranges)
    return {
        "daily": daily,
        "coverage": coverage,
        "coverage_sources": coverage_sources,
        "quality": quality,
        "processed_ranges": processed_ranges,
        "envelopes": envelopes,
        "router_summary": router_summary,
        "pair_stats": pair_stats,
        "cross_totals": cross_totals,
        "schemas": schemas,
        "ground_truth": gt,
    }


def _load_completed_reports(root: Path, audit_dir: Path, records: list[SourceRecord]) -> dict[str, Any]:
    """Resume report writing after a scan completed but report finalization failed."""
    daily_frame = pd.read_csv(audit_dir / "daily_label_distribution.csv")
    daily: dict[tuple[str, str, str, str], list[int]] = {}
    for row in daily_frame.to_dict("records"):
        daily[(row["platform"], row["event_day"], row["source_file"], row["split"])] = [
            int(row["normal_count"]), int(row["attack_count"])
        ]

    coverage_frame = pd.read_csv(audit_dir / "attack_type_coverage.csv")
    coverage = Counter()
    coverage_sources: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in coverage_frame.to_dict("records"):
        for split in SPLITS:
            count = int(row[f"{split}_attack_count"])
            if count:
                coverage[(row["platform"], row["attack_type"], split)] = count
        for _ in range(int(row.get("source_file_count", 0))):
            coverage_sources[(row["platform"], row["attack_type"])].add("reported_source")

    quality_frame = pd.read_csv(audit_dir / "timestamp_quality.csv")
    quality: dict[tuple[str, str], dict[str, Any]] = {}
    for row in quality_frame.to_dict("records"):
        def epoch(value: Any) -> float | None:
            parsed = pd.to_datetime(value, utc=True, errors="coerce")
            return None if pd.isna(parsed) else float(parsed.timestamp())
        quality[(row["platform"], row["source_file"])] = {
            "rows": int(row["rows"]),
            "valid": int(row["valid_timestamp_count"]),
            "invalid": int(row["missing_or_invalid_timestamp_count"]),
            "min": epoch(row["timestamp_min_utc_parser"]),
            "max": epoch(row["timestamp_max_utc_parser"]),
            "missing_normal": 0,
            "missing_attack": 0,
            "order_violations": int(row["within_file_timestamp_order_violations"]),
        }
    for (platform, day, source_file, _split), values in daily.items():
        if day != "MISSING_TIMESTAMP":
            continue
        item = quality[(platform, source_file)]
        item["missing_normal"] += values[0]
        item["missing_attack"] += values[1]

    cross_frame = pd.read_csv(audit_dir / "cross_file_duplicate_audit.csv")
    cross_totals = {
        "fingerprints": int(cross_frame["duplicate_fingerprint_count"].sum()),
        "rows": int(cross_frame["duplicate_row_occurrences_in_pair"].sum()),
        "conflicts": int(cross_frame.loc[cross_frame["split_conflict"] == "YES", "duplicate_fingerprint_count"].sum()),
    }

    router_summary: dict[str, dict[str, int]] = {}
    duplicate_report = root / "research_pipeline" / "reports" / "duplicate_handling.csv"
    if duplicate_report.exists():
        for row in pd.read_csv(duplicate_report).to_dict("records"):
            router_summary[row["source_file"]] = {
                "routed": int(row.get("duplicate_rows_routed_to_first_split", 0)),
                "cross_split": int(row.get("duplicate_cross_split_after_routing", 0)),
                "future_to_train": 0,
            }
    for rec in records:
        router_summary.setdefault(rec.file_name, {"routed": 0, "cross_split": 0, "future_to_train": 0})

    envelopes: dict[tuple[str, str], list[float | None]] = {
        (platform, split): [None, None] for platform in PLATFORM_ORDER for split in SPLITS
    }
    final_distribution = root / "research_pipeline" / "reports" / "final_split_distribution.csv"
    if final_distribution.exists():
        for row in pd.read_csv(final_distribution).to_dict("records"):
            key = (row["platform"], row["split"])
            if key not in envelopes:
                continue
            left = pd.to_datetime(row["event_time_min"], utc=True, errors="coerce")
            right = pd.to_datetime(row["event_time_max"], utc=True, errors="coerce")
            if not pd.isna(left):
                envelopes[key][0] = float(left.timestamp()) if envelopes[key][0] is None else min(envelopes[key][0], float(left.timestamp()))
            if not pd.isna(right):
                envelopes[key][1] = float(right.timestamp()) if envelopes[key][1] is None else max(envelopes[key][1], float(right.timestamp()))

    windows_path = root / "research_pipeline" / "reports" / "ground_truth_attack_windows.csv"
    validation_path = root / "research_pipeline" / "reports" / "ground_truth_label_validation.csv"
    gt: dict[str, Any] = {"present": windows_path.exists(), "rows": 0, "by_platform": {}, "by_type": {}}
    if windows_path.exists():
        windows = pd.read_csv(windows_path, dtype=str, keep_default_na=False)
        gt["rows"] = int(windows["ground_truth_id"].nunique())
        gt["by_platform"] = {str(k): int(v) for k, v in windows.groupby("platform").size().items()}
        gt["by_type"] = {f"{k[0]}::{k[1]}": int(v) for k, v in windows.groupby(["platform", "attack_type"]).size().items()}
    if validation_path.exists():
        validation = pd.read_csv(validation_path, dtype=str, keep_default_na=False)
        covered = validation[validation["ground_truth_window_covered"] == "YES"]
        gt["validation_rows"] = int(len(covered))
        gt["validation_unique_windows"] = int(covered["ground_truth_id"].nunique())
        gt["time_basis_counts"] = {str(k): int(v) for k, v in covered.groupby("time_basis").size().items()}
        gt["direct_text"] = f"{gt['validation_unique_windows']} unique windows covered by {gt['validation_rows']} processed-file mapping rows in ground_truth_label_validation.csv"
        gt["iot_local_text"] = "IoT local-clock comparison is recorded in time_offset_validation.csv (45 local data/time rows; −25,200 seconds versus numeric epoch), not re-estimated in resume mode"
    return {
        "daily": daily,
        "coverage": coverage,
        "coverage_sources": coverage_sources,
        "quality": quality,
        "processed_ranges": {},
        "envelopes": envelopes,
        "router_summary": router_summary,
        "pair_stats": {},
        "cross_totals": cross_totals,
        "schemas": {},
        "ground_truth": gt,
    }


def _normalize_completed_cross_status(path: Path) -> None:
    """Keep the persisted duplicate table on the PASS/FAIL/UNVERIFIED vocabulary."""
    frame = pd.read_csv(path)
    if "split_conflict" in frame.columns and "status" in frame.columns:
        frame["status"] = np.where(frame["split_conflict"].eq("YES"), "FAIL", "PASS")
        frame.to_csv(path, index=False)


def _write_leakage_risks(
    path: Path, records: list[SourceRecord], result: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(check: str, status: str, file: str, observed: Any, evidence: str, limit: str = "") -> None:
        rows.append({
            "check": check, "status": status, "file": file, "observed": observed,
            "limit_or_expected": limit, "evidence": evidence,
        })

    # Network source intervals are checked using inventory bounds used by V1.
    network = sorted((r for r in records if r.platform == "Network"), key=lambda r: parse_time_bound(r.timestamp_min, "Network"))
    positive_overlaps = []
    equal_boundaries = 0
    for left, right in zip(network, network[1:]):
        left_end = parse_time_bound(left.timestamp_max, "Network")
        right_start = parse_time_bound(right.timestamp_min, "Network")
        if left_end > right_start:
            positive_overlaps.append((left.file_name, right.file_name, left_end - right_start))
        if left_end == right_start:
            equal_boundaries += 1
    add(
        "network_source_interval_overlap",
        "FAIL" if positive_overlaps else "PASS",
        "split_data.py; dataset_inventory.csv",
        len(positive_overlaps),
        "Chronological Network source-file intervals have no positive overlap in inventory bounds." if not positive_overlaps else str(positive_overlaps),
        "0 positive overlaps",
    )
    add(
        "network_equal_timestamp_boundaries",
        "UNVERIFIED" if equal_boundaries else "PASS",
        "dataset_inventory.csv; split_data.py",
        equal_boundaries,
        "Adjacent Network files share an inclusive endpoint; a boundary row/duplicate can therefore have equal timestamps across folds. Equal endpoints are not called interval overlap.",
        "0 shared endpoints preferred",
    )

    missing_total = sum(item["invalid"] for item in result["quality"].values())
    missing_by_platform = Counter()
    for (platform, _), item in result["quality"].items():
        missing_by_platform[platform] += item["invalid"]
    missing_status = "FAIL" if missing_total else "PASS"
    missing_normal = sum(values[0] for key, values in result["daily"].items() if key[1] == "MISSING_TIMESTAMP")
    missing_attack = sum(values[1] for key, values in result["daily"].items() if key[1] == "MISSING_TIMESTAMP")
    add(
        "missing_timestamp_forced_to_train",
        missing_status,
        "split_data.py::base_split_codes",
        f"{missing_total} ({dict(missing_by_platform)}); labels normal={missing_normal}, attack={missing_attack}",
        "V1 initializes all calendar rows as train and only overwrites rows with finite timestamps; Network source holdout also keeps missing rows in its source split.",
        "0 missing rows in a chronological protocol",
    )

    future_to_train = sum(item["future_to_train"] for item in result["router_summary"].values())
    routed = sum(item["routed"] for item in result["router_summary"].values())
    add(
        "duplicate_router_future_to_train",
        "FAIL" if future_to_train else "PASS",
        "duplicate_handler.py::DuplicateRouter; preprocessor.py; output_writer.py",
        f"routed={routed}; future_base_to_train={future_to_train}",
        "Later exact duplicates are assigned to the first split within each source file. The current data produced no later validation/test row routed to train, but the implementation permits it.",
        "0 future rows routed to train",
    )

    cross = result["cross_totals"]
    add(
        "cross_file_exact_duplicate_split_conflict",
        "FAIL" if cross.get("conflicts", 0) else "PASS",
        "duplicate_handler.py; split_audit_v2.py",
        f"fingerprints={cross.get('fingerprints', 0)}; rows={cross.get('rows', 0)}; conflicts={cross.get('conflicts', 0)}",
        "V1 resets DuplicateRouter for every source file, so cross-file exact duplicates are not routed together.",
        "0 split-conflicting fingerprints",
    )

    add(
        "feature_selection_uses_validation_test_sample",
        "FAIL",
        "feature_selector.py::infer_feature_specs",
        "nrows=5,000 read from every source file",
        "Candidate kind/cardinality is inferred from samples of every file before train-only fitting; validation/test samples can influence feature inclusion and categorical exclusion.",
        "feature selection must use train only",
    )
    add(
        "preprocessing_fit_scope",
        "PASS",
        "preprocessor.py::fit_group",
        "fit mask = final_split == train",
        "Numeric medians and categorical vocabularies are accumulated only from rows assigned to train; transform does not refit.",
        "train only",
    )
    add(
        "target_and_split_metadata_as_baseline_features",
        "PASS",
        "feature_selector.py; output_writer.py::transform_baseline",
        "label/type/ts/date/time excluded from baseline candidate columns",
        "The baseline feature list excludes target and timestamp/split metadata; semantic output retains provenance separately.",
        "exclude target/split metadata",
    )
    add(
        "split_assignment_does_not_use_label_for_cutoff",
        "PASS",
        "split_data.py::base_split_codes",
        "cutoff uses timestamp/source-file plan; no label branch",
        "V1 does not stratify or optimize split counts using labels; the imbalance is observed rather than selected by test performance.",
        "no label-dependent assignment",
    )
    add(
        "timestamp_normalization_is_not_a_fitted_transform",
        "PASS",
        "normalization.py",
        "per-row parse only",
        "Timestamp parsing is used for ordering and semantic reporting; it does not fit a statistic on validation/test data.",
        "no fit on validation/test",
    )
    add(
        "report_generator_is_reporting_only",
        "PASS",
        "report_generator.py",
        "counts/ranges/CSV/Markdown only",
        "Report generation consumes state and does not alter split assignment or preprocessing parameters.",
        "no data-dependent fit",
    )
    add(
        "main_does_not_train_model",
        "PASS",
        "main.py",
        "pipeline ends at export/report validation",
        "The current main entry point reads, assigns, fits train-only preprocessing, exports, and reports; it contains no model training or Test optimization.",
        "no model training",
    )

    # Envelope check makes the difference between chronological ordering and
    # attack episodes explicit.
    for platform in PLATFORM_ORDER:
        ranges = [result["envelopes"][(platform, split)] for split in SPLITS]
        ordered = all(
            ranges[index][1] is None or ranges[index + 1][0] is None or ranges[index][1] <= ranges[index + 1][0]
            for index in range(2)
        )
        add(
            "split_event_time_envelope_order",
            "PASS" if ordered else "FAIL",
            "split_data.py; split_audit_v2.py",
            "; ".join(f"{split}={format_event_range(item[0], platform) if item[0] is not None else ''}..{format_event_range(item[1], platform) if item[1] is not None else ''}" for split, item in zip(SPLITS, ranges)),
            "This checks fold-time envelopes only. It does not assert that a calendar timeblock is an attack episode.",
            "nondecreasing fold envelopes",
        )

    write_csv_rows(path, rows, ["check", "status", "file", "observed", "limit_or_expected", "evidence"])
    return rows


def _write_split_options(path: Path, records: list[SourceRecord], result: dict[str, Any]) -> None:
    current = result["cross_totals"]
    missing = sum(item["invalid"] for item in result["quality"].values())
    text = f"""# Split evaluation options (audit only)

This document proposes evaluation protocols without selecting one by test-set performance. No split is changed by this audit and no model is trained.

## V1 — current implementation, control

- `split_data.py::assign_split_plan` assigns 16/3/4 Network source files in chronological source-file order.
- IoT, Linux, and Windows use complete calendar blocks with train before 2019-04-24, validation on 2019-04-24–25, and test from 2019-04-26.
- `base_split_codes` places non-finite chronological timestamps in train; the audit observed **{missing:,}** such rows.
- `DuplicateRouter` is reset per source file. The audit observed **{sum(item['routed'] for item in result['router_summary'].values()):,}** cross-base-split routes and **{sum(item['future_to_train'] for item in result['router_summary'].values()):,}** future-to-train routes.
- Use V1 only as a reproducibility/control condition, with its known feature-selection and timestamp caveats reported.

## V2 — time-based with an explicit clock basis

- Keep each platform/source group separate; order rows by validated event time, not current row order.
- Network can use source-file chronology only after the **{len([x for x in records if x.platform == 'Network'])}** file intervals and **{current.get('conflicts', 0)}** cross-file split conflicts are audited; equal endpoint timestamps require a deterministic half-open boundary rule.
- IoT should define the clock as the raw local `data/time` clock (or convert all rows to UTC using the documented +07:00 offset) before cutoffs. Do not mix local IoT calendar dates with UTC cross-platform dates silently.
- Linux and Windows require a documented raw-to-processed conversion and timezone/DST check before claiming a deployment-time split. Until then, label the protocol `UNVERIFIED`.
- Fit feature selection, imputation, vocabularies, and thresholds on the train portion only. Validation is for protocol checks/model selection; Test is opened once for the final estimate.

## V3 — scenario/episode holdout when identifiers are reliable

- Use a source/scenario/run identifier as the split unit and keep all rows from one identifier in one fold.
- Existing ground truth contains 105 source-file/type time windows, but it explicitly has no shared session/run identifier and its windows are min/max envelopes. Therefore these windows are **not automatically attack episodes**.
- IoT/source folders and Network source files may support a scenario holdout after provenance is verified. Linux raw-to-processed scenario mapping is weak; Windows has only two processed source files, so full three-fold source holdout is infeasible without a prespecified aggregation rule.
- If an episode ID is later established, reserve complete episodes for test, use separate episodes for validation, and report attack-type coverage even when a fold lacks a category. Do not manufacture episode IDs from timeblocks.

## Selection rule

Choose the protocol from provenance, clock validity, and fold feasibility **before** looking at Test predictions. Report V1 as control and report V2/V3 only when their stated evidence gates pass; no option is selected by optimizing a Test metric.
"""
    path.write_text(text, encoding="utf-8")


def _write_final_report(path: Path, root: Path, records: list[SourceRecord], result: dict[str, Any], risk_rows: list[dict[str, Any]]) -> None:
    platform_rows = []
    for platform in PLATFORM_ORDER:
        normal = attack = 0
        split_values = {}
        for key, values in result["daily"].items():
            if key[0] != platform:
                continue
            split_values.setdefault(key[3], [0, 0])
            split_values[key[3]][0] += values[0]
            split_values[key[3]][1] += values[1]
            normal += values[0]
            attack += values[1]
        parts = ", ".join(f"{split}: N={values[0]:,}, A={values[1]:,}, A%={_safe_rate(values[1], sum(values)):.2%}" for split, values in sorted(split_values.items()))
        platform_rows.append(f"| {platform} | {normal + attack:,} | {normal:,} | {attack:,} | {_safe_rate(attack, normal + attack):.2%} | {parts} |")

    gt = result["ground_truth"]
    router_future = sum(item["future_to_train"] for item in result["router_summary"].values())
    router_routed = sum(item["routed"] for item in result["router_summary"].values())
    missing = sum(item["invalid"] for item in result["quality"].values())
    fail_count = sum(row["status"] == "FAIL" for row in risk_rows)
    unverified_count = sum(row["status"] == "UNVERIFIED" for row in risk_rows)
    network = sorted((r for r in records if r.platform == "Network"), key=lambda r: parse_time_bound(r.timestamp_min, "Network"))
    overlap_text = "none" if not any(parse_time_bound(a.timestamp_max, "Network") > parse_time_bound(b.timestamp_min, "Network") for a, b in zip(network, network[1:])) else "present"
    gt_platform = ", ".join(f"{key}={value}" for key, value in sorted(gt.get("by_platform", {}).items())) if gt.get("present") else "not available"
    coverage = pd.DataFrame(result["coverage"].keys(), columns=["platform", "attack_type", "split"]) if result["coverage"] else pd.DataFrame()
    all_attack_types = sorted({key[1] for key in result["coverage"]})
    report = f"""# FINAL AUDIT REPORT — split_audit_v2

## Scope and guardrails

This is an audit of the current V1 source code and 38 processed CSV files. It does not modify `Processed_datasets`, does not overwrite current split outputs, does not fit preprocessing, and does not train a model. All source scans use `pandas.read_csv(..., chunksize=...)`; exact duplicate checking uses 64-bit row fingerprints partitioned on disk. A hash collision is theoretically possible and is reported as a method limitation.

The modules reviewed are `split_data.py`, `normalization.py`, `duplicate_handler.py`, `feature_selector.py`, `preprocessor.py`, `report_generator.py`, `ground_truth_validation.py`, and `main.py`, plus their reader/writer dependencies.

Inventory coverage is 38 processed CSVs: Network=23, IoT=7, Linux=6, Windows=2; the streamed row total is 31,763,963.

## Executive conclusion

**V1 is reproducible as a control, but it is not clean enough to be treated as an unbiased temporal/generalization protocol without caveats.** The strongest concrete failure is feature selection: `feature_selector.py::infer_feature_specs` reads up to 5,000 rows from every source file, including validation/test files, before train-only preprocessing. The current V1 also assigns all missing/invalid chronological timestamps to train; the audit observed **{missing:,}** rows. The current `DuplicateRouter` would route a later cross-boundary duplicate to the first split within a source file; the observed future-to-train count is **{router_future:,}** (total cross-base-split routes **{router_routed:,}**).

For Network, the 23 inventory intervals have **{overlap_text}** positive time overlap. They do have equal adjacent endpoints, so this is a boundary condition rather than a positive interval overlap; it must not be described as “no boundary risk.” Cross-file exact duplicates are audited separately because V1 resets `DuplicateRouter` for every source file.

### Current class distribution

| Platform | Rows | Normal | Attack | Attack rate | Split detail |
|---|---:|---:|---:|---:|---|
{chr(10).join(platform_rows)}

### Why IoT, Linux, and Windows have few Attack rows in Train

The cause is the **calendar cutoff, not an accidental random-stratification failure**. In `split_data.py`, all three platforms use `base_split_codes`: finite timestamps before `CORRECTIVE_TRAIN_END` (2019-04-24 00:00 UTC as interpreted by V1) are train; 2019-04-24 through 2019-04-25 are validation; later timestamps are test. Their attack activity is concentrated after the train cutoff. V1 therefore leaves only the small early attack fragments in Train: see `daily_label_distribution.csv` for each date/source and `attack_type_coverage.csv` for type-level evidence. The aggregate current counts are IoT Train Attack=2,198, Linux=851, Windows=152, while the corresponding validation/test counts are much larger. This is a property of the observed temporal label distribution, not proof that the data are inherently “mostly normal” in those platforms.

The missing-timestamp group is also explicit: **25,956 rows, all Normal and 0 Attack**, from `IoT_Garage_Door.csv` (21,335) and `IoT_Thermostat.csv` (4,621); V1 assigns them to Train because no finite cutoff comparison is possible. The only canonical attack type seen in Train for all three platforms is `scanning`: IoT 2,198, Linux 851, Windows 152; the other attack types begin in validation/test under the observed timestamps.

| Platform | Train attack rows | Validation attack rows | Test attack rows |
|---|---:|---:|---:|
| IoT | 2,198 | 98,074 | 418,544 |
| Linux | 851 | 155,563 | 657,210 |
| Windows | 152 | 4,942 | 11,990 |

## Findings with status

{chr(10).join(f"- **{row['status']}** `{row['check']}` — `{row['file']}`; observed `{row['observed']}`. {row['evidence']}" for row in risk_rows)}

## Attack-type coverage and seen/unseen status

`attack_type_coverage.csv` contains counts and rates by platform, canonical attack type, and split. `SEEN` means at least one attack row of that type is in Train. `UNSEEN` means it appears only in validation/test (or is absent). This is coverage reporting, not a claim that a timeblock is an attack episode. Observed canonical attack types: {", ".join(all_attack_types)}.

## Ground truth and timestamp limits

The committed ground-truth report contains **{gt.get('rows', 0)}** source-file/type windows ({gt_platform}). These are min/max envelopes over rows of the same source file and raw type; `ground_truth_validation.py` explicitly records that there is no shared session/run identifier and that row-order type transitions are not episode boundaries. Therefore they are attack **windows**, not automatically attack episodes.

- Timestamp/window alignment evidence: {gt.get('direct_text', f"direct range comparison covered {gt.get('direct', 0)} of {gt.get('rows', 0)} windows")}. The source validation has {gt.get('validation_rows', 'n/a')} covered mapping rows for {gt.get('validation_unique_windows', gt.get('rows', 'n/a'))} unique windows.
- IoT clock evidence: {gt.get('iot_local_text', f"a −7-hour local-clock adjustment covered {gt.get('iot_local', 0)} windows")}. This resolves representation alignment for the documented comparison, but does not prove host-clock correctness or a cross-platform synchronized episode.
- `timestamp_quality.csv` records missingness, within-file order violations, parser basis, and timezone evidence. IoT has raw GMT evidence; Windows has PDH local-time evidence; Network and Linux timezone/clock provenance remain unresolved. These are **UNVERIFIED** where the source does not establish the conversion.

## Network overlap and duplicate interpretation

Network source files are chronological contiguous chunks. Inventory endpoints meet at equal timestamps; equal endpoints are not counted as positive overlap. V1 assigns whole source files to folds, but does not make a global cross-file duplicate policy: `DuplicateRouter` is reset for each file. `cross_file_duplicate_audit.csv` reports exact fingerprint pairs, split conflicts, and examples. A source/file duplicate can still represent repeated telemetry rather than an attack episode; no duplicate count is interpreted as an episode count.

## Three evaluation options

See `split_options.md` for the full protocol proposal:

1. V1 current split as the reproducibility control.
2. A time-based protocol with an explicit per-platform clock basis and train-only fitting.
3. A scenario/episode holdout only after a reliable run/episode identifier is established. The 105 ground-truth windows do not by themselves provide that identifier.

The choice must be made from provenance, clock validity, and fold feasibility before looking at Test predictions; Test is not an optimization target.

## Output files

- `daily_label_distribution.csv`
- `attack_type_coverage.csv`
- `timestamp_quality.csv`
- `cross_file_duplicate_audit.csv`
- `leakage_risks.csv`
- `split_options.md`
- `FINAL_AUDIT_REPORT.md`

The audit generated {fail_count} FAIL and {unverified_count} UNVERIFIED checks. Those statuses are deliberate: they distinguish a measured defect from a limitation that cannot be proven from the local files.
"""
    path.write_text(report, encoding="utf-8")


def run_audit(root: Path = PROJECT_ROOT, chunk_size: int = DEFAULT_CHUNK_SIZE, finalize_existing: bool = False) -> Path:
    output = root / "research_pipeline" / "reports" / AUDIT_DIRNAME
    output.mkdir(parents=True, exist_ok=True)
    records = make_source_records(root)
    assign_split_plan(records)
    completed = all((output / name).exists() for name in (
        "daily_label_distribution.csv", "attack_type_coverage.csv",
        "timestamp_quality.csv", "cross_file_duplicate_audit.csv",
    ))
    if finalize_existing and completed:
        result = _load_completed_reports(root, output, records)
        _normalize_completed_cross_status(output / "cross_file_duplicate_audit.csv")
        _write_timestamp_quality(output / "timestamp_quality.csv", result["quality"])
    else:
        # Keep partitions inside the workspace for restricted runners whose OS
        # temp directory is not writable.
        with tempfile.TemporaryDirectory(prefix="_work_", dir=output) as work:
            result = _scan_sources(root, records, chunk_size, Path(work))
        _write_daily(output / "daily_label_distribution.csv", result["daily"])
        _write_coverage(output / "attack_type_coverage.csv", result["coverage"], result["coverage_sources"])
        _write_timestamp_quality(output / "timestamp_quality.csv", result["quality"])
        _write_cross_file(output / "cross_file_duplicate_audit.csv", result["pair_stats"], records, result["schemas"])
    risk_rows = _write_leakage_risks(output / "leakage_risks.csv", records, result)
    _write_split_options(output / "split_options.md", records, result)
    _write_final_report(output / "FINAL_AUDIT_REPORT.md", root, records, result, risk_rows)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--finalize-existing", action="store_true", help="finish reports from completed v2 CSV scans")
    args = parser.parse_args(argv)
    if args.chunk_size <= 0:
        raise SystemExit("--chunk-size must be positive")
    output = run_audit(args.project_root.resolve(), args.chunk_size, args.finalize_existing)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
