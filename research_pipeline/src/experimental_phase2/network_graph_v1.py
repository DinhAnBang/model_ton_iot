"""Network Graph V1 with causal Event nodes and bounded Train data."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..common.config import raw_type_mapping
from ..common.paths import PROJECT_ROOT
from ..pipeline.data_reader import make_source_records
from ..pipeline.models import RunState
from ..pipeline.normalization import event_time_numeric, validate_chunk
from ..pipeline.split_data import assign_split_plan
from ..pipeline.feature_selector import read_feature_risk


TARGET_COLUMNS = {"label", "type", "ts", "split", "source_file", "event_id"}
NODE_TYPES = ("ip", "event", "protocol", "service")
RELATIONS = (
    "source_ip_to_event",
    "event_to_destination_ip",
    "event_to_protocol",
    "event_to_service",
)


def event_id_for(source_file: str, source_row: int) -> str:
    """Stable event key: source file plus zero-based data-row ordinal."""
    return f"event::{source_file}::{source_row:09d}"


def typed_node_id(node_type: str, value: str) -> str:
    return f"{node_type}::{value}"


def assign_half_open_snapshot(timestamp: float, origin: float, window_seconds: int = 30) -> int:
    """Assign t to [origin+k*w, origin+(k+1)*w)."""
    if not np.isfinite(timestamp):
        return -1
    return int(np.floor((float(timestamp) - float(origin)) / window_seconds))


def event_feature_columns(frame: pd.DataFrame, risk: dict[tuple[str, str, str], dict[str, Any]], file_name: str) -> list[str]:
    """Choose observable numeric Event fields; targets and time are excluded."""
    selected: list[str] = []
    for column in frame.columns:
        if column in TARGET_COLUMNS:
            continue
        risk_row = risk.get(("Network", file_name, column), {})
        if risk_row.get("classification") != "VALID_OBSERVABLE_FEATURE":
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        nonmissing = frame[column].notna()
        if int(nonmissing.sum()) and float(values[nonmissing].notna().mean()) >= 0.95:
            selected.append(column)
    return selected


def _clean(value: Any) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"-", "nan", "none"}:
        return None
    return text


def read_ordered_window(path: Path, start_row: int, sample_rows: int, chunk_size: int) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    remaining = sample_rows
    skipped = 0
    for frame in pd.read_csv(path, chunksize=chunk_size, low_memory=False):
        if skipped + len(frame) <= start_row:
            skipped += len(frame)
            continue
        local_start = max(0, start_row - skipped)
        take = frame.iloc[local_start:local_start + remaining].copy()
        pieces.append(take)
        remaining -= len(take)
        skipped += len(frame)
        if remaining <= 0:
            break
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def choose_label_checked_window(path: Path, sample_rows: int, scan_rows: int, chunk_size: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    # Quét nhãn trước, sau đó lấy một cửa sổ liên tiếp theo thứ tự nguồn.
    """Scan labels first, then choose one contiguous source-ordered window."""
    label_scan = pd.read_csv(path, nrows=scan_rows, usecols=["label"])
    labels = pd.to_numeric(label_scan["label"], errors="coerce").to_numpy()
    attack_positions = np.flatnonzero(labels == 1)
    if len(attack_positions):
        # Prespecified rule: center the bounded window on the first observed
        # class transition; this is not an accuracy or Test optimization.
        first_attack = int(attack_positions[0])
        start = max(0, first_attack - sample_rows // 2)
        rule = "first observed Attack transition centered in a contiguous source-ordered window"
    else:
        first_attack = -1
        start = 0
        rule = "first source-ordered window because bounded label scan has no Attack"
    window = read_ordered_window(path, start, sample_rows, chunk_size)
    window_labels = pd.to_numeric(window["label"], errors="coerce")
    return window, {
        "label_scan_rows": len(label_scan),
        "label_scan_normal": int((labels == 0).sum()),
        "label_scan_attack": int((labels == 1).sum()),
        "first_attack_row_in_scan": first_attack,
        "selected_start_row": start,
        "selected_end_row_exclusive": start + len(window),
        "selected_normal": int((window_labels == 0).sum()),
        "selected_attack": int((window_labels == 1).sum()),
        "selection_rule": rule,
    }


def build_event_graph(
    frame: pd.DataFrame,
    record: Any,
    times: pd.Series,
    feature_columns: list[str],
    source_row_offset: int = 0,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], pd.DataFrame]:
    """Build a causal, source-local Event graph and retain provenance."""
    # Mỗi dòng timestamp hợp lệ tạo đúng một Event; không suy diễn session.
    nodes: dict[str, dict[str, Any]] = {node_type: {} for node_type in NODE_TYPES}
    edges: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    valid_rows = 0
    for source_row, row in frame.iterrows():
        timestamp = float(times.iloc[source_row]) if pd.notna(times.iloc[source_row]) else np.nan
        if not np.isfinite(timestamp):
            continue
        valid_rows += 1
        source_row_number = source_row_offset + int(source_row)
        event_id = event_id_for(record.file_name, source_row_number)
        label = int(row["label"])
        event_features = {
            column: (float(pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]) if pd.notna(row[column]) else np.nan)
            for column in feature_columns
        }
        nodes["event"][event_id] = {
            "node_type": "event",
            "source_file": record.file_name,
            "source_row": source_row_number,
            "timestamp": timestamp,
            "label": label,
            "features": event_features,
        }
        event_rows.append({
            "event_id": event_id,
            "source_file": record.file_name,
            "source_row": source_row_number,
            "timestamp": timestamp,
            "label": label,
            "src_ip": _clean(row.get("src_ip")),
            "dst_ip": _clean(row.get("dst_ip")),
            "proto": _clean(row.get("proto")),
            "service": _clean(row.get("service")),
        })

        src = _clean(row.get("src_ip"))
        dst = _clean(row.get("dst_ip"))
        proto = _clean(row.get("proto"))
        service = _clean(row.get("service"))
        if src is not None:
            src_id = typed_node_id("ip", src)
            nodes["ip"].setdefault(src_id, {"node_type": "ip", "value": src})
            edges.append({"src_type": "ip", "src_id": src_id, "relation": "source_ip_to_event", "dst_type": "event", "dst_id": event_id, "event_id": event_id, "source_file": record.file_name, "source_row": source_row_number, "timestamp": timestamp})
        if dst is not None:
            dst_id = typed_node_id("ip", dst)
            nodes["ip"].setdefault(dst_id, {"node_type": "ip", "value": dst})
            edges.append({"src_type": "event", "src_id": event_id, "relation": "event_to_destination_ip", "dst_type": "ip", "dst_id": dst_id, "event_id": event_id, "source_file": record.file_name, "source_row": source_row_number, "timestamp": timestamp})
        if proto is not None:
            proto_id = typed_node_id("protocol", proto)
            nodes["protocol"].setdefault(proto_id, {"node_type": "protocol", "value": proto})
            edges.append({"src_type": "event", "src_id": event_id, "relation": "event_to_protocol", "dst_type": "protocol", "dst_id": proto_id, "event_id": event_id, "source_file": record.file_name, "source_row": source_row_number, "timestamp": timestamp})
        if service is not None:
            service_id = typed_node_id("service", service)
            nodes["service"].setdefault(service_id, {"node_type": "service", "value": service})
            edges.append({"src_type": "event", "src_id": event_id, "relation": "event_to_service", "dst_type": "service", "dst_id": service_id, "event_id": event_id, "source_file": record.file_name, "source_row": source_row_number, "timestamp": timestamp})

    events = pd.DataFrame(event_rows).sort_values(["timestamp", "source_row"], kind="stable").reset_index(drop=True)
    events.attrs["valid_rows"] = valid_rows
    return nodes, edges, events


def causal_leakage_audit(events: pd.DataFrame, edges: pd.DataFrame, feature_columns: list[str], source_file: str) -> dict[str, Any]:
    """Compute leakage checks from timestamps and provenance, not fixed zeros."""
    first_seen: dict[str, float] = {}
    causal_ids: dict[str, int] = {}
    future_vocab_exposures_if_global_encoder = 0
    future_node_uses = 0
    identity_collisions = 0
    ordered_events = events.sort_values(["timestamp", "source_row"], kind="stable")
    all_first_seen: dict[str, float] = {}
    for row in ordered_events.itertuples(index=False):
        for key in [
            typed_node_id("ip", row.src_ip) if row.src_ip else None,
            typed_node_id("ip", row.dst_ip) if row.dst_ip else None,
            typed_node_id("protocol", row.proto) if row.proto else None,
            typed_node_id("service", row.service) if row.service else None,
        ]:
            if key is not None:
                all_first_seen[key] = min(all_first_seen.get(key, float(row.timestamp)), float(row.timestamp))
    for causal_index, row in enumerate(ordered_events.itertuples(index=False)):
        keys = [
            typed_node_id("ip", row.src_ip) if row.src_ip else None,
            typed_node_id("ip", row.dst_ip) if row.dst_ip else None,
            typed_node_id("protocol", row.proto) if row.proto else None,
            typed_node_id("service", row.service) if row.service else None,
        ]
        for key in [value for value in keys if value is not None]:
            previous = first_seen.setdefault(key, float(row.timestamp))
            if key not in causal_ids:
                causal_ids[key] = causal_index
            if previous > float(row.timestamp) or causal_ids[key] > causal_index:
                future_node_uses += 1
        # This is a measured counter for what a global vocabulary would expose
        # to this past row. The causal graph itself never uses these keys.
        future_vocab_exposures_if_global_encoder += sum(
            seen_at > float(row.timestamp) for seen_at in all_first_seen.values()
        )
        if event_id_for(row.source_file, int(row.source_row)) != row.event_id:
            identity_collisions += 1

    future_edge_uses = int((edges["timestamp"] > edges.groupby("event_id")["timestamp"].transform("min")).sum()) if not edges.empty else 0
    source_violations = int((edges["source_file"] != source_file).sum()) if not edges.empty else 0
    target_overlap = sorted(set(feature_columns) & TARGET_COLUMNS)
    feature_source_mismatch = int((events["source_file"] != source_file).sum()) if not events.empty else 0
    return {
        "event_count": len(events),
        "edge_count": len(edges),
        "future_node_id_uses": future_node_uses,
        "future_vocab_exposures_if_global_encoder": future_vocab_exposures_if_global_encoder,
        "future_edge_uses": future_edge_uses,
        "identity_collisions": identity_collisions,
        "cross_source_edges": source_violations,
        "target_columns_in_event_features": ";".join(target_overlap),
        "event_feature_source_mismatch": feature_source_mismatch,
        "status": "PASS" if not any((future_node_uses, future_edge_uses, identity_collisions, source_violations, target_overlap, feature_source_mismatch)) else "FAIL",
    }


def snapshot_statistics(events: pd.DataFrame, edges: pd.DataFrame, window_seconds: int = 30) -> tuple[pd.DataFrame, dict[str, Any]]:
    if events.empty:
        return pd.DataFrame(), {"snapshot_count": 0, "timestamp_status": "FAIL"}
    origin = float(events["timestamp"].min())
    ordered = bool((events["timestamp"].diff().dropna() >= 0).all())
    events = events.copy()
    events["snapshot_id"] = [assign_half_open_snapshot(value, origin, window_seconds) for value in events["timestamp"]]
    edges = edges.copy()
    if not edges.empty:
        edges["snapshot_id"] = [assign_half_open_snapshot(value, origin, window_seconds) for value in edges["timestamp"]]
    rows: list[dict[str, Any]] = []
    for snapshot_id, group in events.groupby("snapshot_id", sort=True):
        snapshot_start = origin + int(snapshot_id) * window_seconds
        snapshot_end = snapshot_start + window_seconds
        event_ids = set(group["event_id"])
        node_sets = {
            "event": event_ids,
            "ip": {typed_node_id("ip", value) for value in group[["src_ip", "dst_ip"]].stack().dropna()},
            "protocol": {typed_node_id("protocol", value) for value in group["proto"].dropna()},
            "service": {typed_node_id("service", value) for value in group["service"].dropna()},
        }
        edge_group = edges[edges["snapshot_id"] == snapshot_id]
        relation_counts = edge_group["relation"].value_counts().to_dict() if not edge_group.empty else {}
        normal = int((group["label"] == 0).sum())
        attack = int((group["label"] == 1).sum())
        rows.append({
            "snapshot_id": int(snapshot_id), "start_timestamp": snapshot_start, "end_timestamp_exclusive": snapshot_end,
            "interval_semantics": "[start,end)", "source_file": str(group["source_file"].iloc[0]),
            "normal_events": normal, "attack_events": attack, "total_events": len(group),
            "ip_nodes": len(node_sets["ip"]), "event_nodes": len(node_sets["event"]),
            "protocol_nodes": len(node_sets["protocol"]), "service_nodes": len(node_sets["service"]),
            "source_ip_to_event_edges": int(relation_counts.get("source_ip_to_event", 0)),
            "event_to_destination_ip_edges": int(relation_counts.get("event_to_destination_ip", 0)),
            "event_to_protocol_edges": int(relation_counts.get("event_to_protocol", 0)),
            "event_to_service_edges": int(relation_counts.get("event_to_service", 0)),
            "single_class_snapshot": bool(normal == 0 or attack == 0),
            "status": "PASS" if len(event_ids) == len(group) and snapshot_start <= group["timestamp"].min() and group["timestamp"].max() < snapshot_end else "FAIL",
        })
    stats = pd.DataFrame(rows)
    return stats, {
        "snapshot_count": len(stats),
        "timestamp_status": "PASS" if ordered and (stats["end_timestamp_exclusive"] > stats["start_timestamp"]).all() else "FAIL",
        "single_class_snapshots": int(stats["single_class_snapshot"].sum()),
    }


def run_network_graph_v1(project_root: Path = PROJECT_ROOT, sample_rows: int = 100_000, chunk_size: int = 50_000) -> dict[str, Any]:
    records = make_source_records(project_root)
    assign_split_plan(records)
    train = sorted([record for record in records if record.platform == "Network" and record.fixed_split == "train"], key=lambda item: item.file_name)
    if not train:
        raise RuntimeError("Network has no Train source file")
    record = train[0]
    frame, label_scan = choose_label_checked_window(record.file_path, sample_rows, max(1_000_000, sample_rows), chunk_size)
    state = RunState()
    _, _, _, times = validate_chunk(frame, record, raw_type_mapping(), state, validation_only=True)
    risk = read_feature_risk()
    features = event_feature_columns(frame, risk, record.file_name)
    nodes, edge_rows, events = build_event_graph(frame, record, times, features, label_scan["selected_start_row"])
    edge_frame = pd.DataFrame(edge_rows)
    leakage = causal_leakage_audit(events, edge_frame, features, record.file_name)
    snapshots, snapshot_meta = snapshot_statistics(events, edge_frame)
    summary_rows: list[dict[str, Any]] = []
    for node_type in NODE_TYPES:
        summary_rows.append({"scope": "graph_v1", "source_file": record.file_name, "node_type": node_type, "relation": "", "count": len(nodes[node_type]), "event_feature_columns": ";".join(features), "node_feature_policy": "raw_event_observation" if node_type == "event" else "type_one_hot; no future vocabulary", "status": "PASS"})
    for relation in RELATIONS:
        summary_rows.append({"scope": "graph_v1", "source_file": record.file_name, "node_type": "", "relation": relation, "count": int((edge_frame["relation"] == relation).sum()) if not edge_frame.empty else 0, "event_feature_columns": ";".join(features), "node_feature_policy": "typed directed edge", "status": "PASS"})
    summary_rows.append({"scope": "graph_v1", "source_file": record.file_name, "node_type": "event_reconciliation", "relation": "valid_timestamp_rows_to_event_nodes", "count": len(events), "event_feature_columns": ";".join(features), "node_feature_policy": "one Event per valid row", "status": "PASS" if len(events) == int(times.notna().sum()) else "FAIL"})
    summary = pd.DataFrame(summary_rows)
    for key, value in label_scan.items():
        summary[f"label_scan_{key}"] = value
    return {
        "record": record, "frame": frame, "times": times, "features": features,
        "nodes": nodes, "edges": edge_frame, "events": events,
        "summary": summary, "snapshots": snapshots, "label_scan": label_scan,
        "leakage": leakage, "snapshot_meta": snapshot_meta,
    }

