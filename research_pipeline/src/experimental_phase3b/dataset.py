"""Bounded source-order Network sampling and causal graph construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..common.config import raw_type_mapping
from ..common.paths import PROJECT_ROOT
from ..pipeline.data_reader import make_source_records
from ..pipeline.feature_selector import read_feature_risk
from ..pipeline.models import RunState
from ..pipeline.normalization import event_time_numeric, validate_chunk
from ..pipeline.split_data import assign_split_plan
from ..experimental_phase2.network_graph_v1 import build_event_graph, event_feature_columns
from ..experimental_phase3a.conversion import (
    NODE_TYPES,
    _causal_maps,
    _edge_index,
    _numeric_event_matrix,
    _snapshot_sequences,
)


@dataclass
class NetworkPart:
    record: Any
    frame: pd.DataFrame
    events: pd.DataFrame
    edges: pd.DataFrame
    nodes: dict[str, dict[str, Any]]
    times: pd.Series
    split: str


@dataclass
class ProtocolDataset:
    train_parts: list[NetworkPart]
    validation_parts: list[NetworkPart]
    train_graph: Any
    validation_graphs: dict[str, Any]
    target_event_ids: dict[str, list[str]]
    cutoffs: dict[str, float]
    feature_columns: list[str]
    duplicate_audit: dict[str, Any]
    source_records: list[Any]


def read_prefix(path: Path, rows: int, chunk_size: int = 250_000) -> pd.DataFrame:
    """Read a deterministic source-order prefix without loading a full CSV."""
    pieces: list[pd.DataFrame] = []
    remaining = rows
    for chunk in pd.read_csv(path, chunksize=min(chunk_size, rows), low_memory=False):
        take = chunk.iloc[:remaining].copy()
        pieces.append(take)
        remaining -= len(take)
        if remaining <= 0:
            break
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def network_records(project_root: Path = PROJECT_ROOT) -> tuple[list[Any], list[Any], list[Any]]:
    records = make_source_records(project_root)
    assign_split_plan(records)
    network = sorted(
        [record for record in records if record.platform == "Network"],
        key=lambda record: (pd.to_datetime(record.timestamp_min, utc=True), record.file_name),
    )
    return (
        [record for record in network if record.fixed_split == "train"],
        [record for record in network if record.fixed_split == "validation"],
        [record for record in network if record.fixed_split == "test"],
    )


def phase2_feature_schema(train_records: list[Any], project_root: Path = PROJECT_ROOT) -> list[str]:
    """Reuse the approved Phase-2 Train-observable Network feature list."""
    stats_path = project_root / "research_pipeline" / "reports" / "experimental_phase2" / "network_graph_statistics.csv"
    if stats_path.exists():
        stats = pd.read_csv(stats_path, dtype=str, keep_default_na=False)
        rows = stats[stats["node_type"].eq("event")]
        if not rows.empty:
            columns = [value for value in rows.iloc[0]["event_feature_columns"].split(";") if value]
            missing = [name for name in columns if any(name not in record.columns for record in train_records)]
            if not missing:
                return columns
    risk = read_feature_risk()
    first = read_prefix(train_records[0].file_path, 2_000)
    columns = event_feature_columns(first, risk, train_records[0].file_name)
    if not columns:
        raise RuntimeError("No approved Network Event feature columns available")
    return columns


def _make_part(record: Any, frame: pd.DataFrame, split: str, feature_columns: list[str]) -> NetworkPart:
    state = RunState()
    _, raw_type, canonical_type, times = validate_chunk(frame, record, raw_type_mapping(), state, validation_only=True)
    nodes, edge_rows, events = build_event_graph(frame, record, times, feature_columns, source_row_offset=0)
    type_by_row = dict(zip(frame.index[times.notna()], canonical_type[times.notna()].astype(str)))
    events["attack_type"] = events["source_row"].map(type_by_row).fillna("UNKNOWN")
    events["split"] = split
    return NetworkPart(record, frame, events, pd.DataFrame(edge_rows), nodes, times, split)


def _combine_parts(parts: list[NetworkPart], feature_columns: list[str], cutoff: float | None = None):
    if not parts:
        raise RuntimeError("Cannot build a graph from no source parts")
    events = pd.concat([part.events for part in parts], ignore_index=True)
    edges = pd.concat([part.edges for part in parts if not part.edges.empty], ignore_index=True) if any(not part.edges.empty for part in parts) else pd.DataFrame()
    if cutoff is not None:
        events = events[events["timestamp"] < cutoff].copy()
        allowed = set(events["event_id"])
        if not edges.empty:
            edges = edges[edges["event_id"].isin(allowed)].copy()
    events = events.sort_values(["timestamp", "source_file", "source_row"], kind="stable").reset_index(drop=True)
    if not edges.empty:
        edges = edges[edges["event_id"].isin(set(events["event_id"]))].copy()
    nodes = {node_type: {} for node_type in NODE_TYPES}
    for part in parts:
        for node_type, values in part.nodes.items():
            nodes[node_type].update(values)
    event_ids = set(events["event_id"])
    nodes["event"] = {key: value for key, value in nodes["event"].items() if key in event_ids}
    event_x = _numeric_event_matrix(events, nodes, feature_columns)
    node_maps, node_values = _causal_maps(events)
    edge_index = _edge_index(edges, node_maps)
    from ..experimental_phase3a.conversion import CausalGraph, _future_removal_test
    snapshots, snapshot_meta = _snapshot_sequences(events, edges, window_seconds=30)
    mapping = events[["event_id", "source_file", "source_row", "timestamp", "label", "attack_type", "split"]].copy()
    mapping["snapshot_id"] = [
        int(np.floor((value - float(events["timestamp"].min())) / 30)) for value in mapping["timestamp"]
    ] if not events.empty else []
    sequence_lookup = snapshots.set_index("snapshot_id")["sequence_id"].to_dict() if not snapshots.empty else {}
    mapping["sequence_id"] = mapping["snapshot_id"].map(sequence_lookup).astype("Int64")
    mapping["tensor_index"] = mapping["event_id"].map(node_maps["event"]).astype("int64")
    mapping = mapping.sort_values("tensor_index", kind="stable").reset_index(drop=True)
    result = {
        "record": parts[0].record,
        "frame": pd.concat([part.frame for part in parts], ignore_index=True),
        "features": feature_columns,
        "label_scan": {"selection_rule": "fixed source-order prefixes; no label-based selection"},
        "nodes": nodes,
        "edges": edges,
        "events": events,
        "snapshot_meta_phase3b": snapshot_meta,
    }
    return CausalGraph(
        result=result,
        events=events,
        edges=edges,
        feature_columns=feature_columns,
        event_x=event_x,
        node_maps=node_maps,
        node_values=node_values,
        edge_index=edge_index,
        event_mapping=mapping,
        snapshots=snapshots,
        future_test=_future_removal_test(events, edges),
    )


def _duplicate_audit(train_parts: list[NetworkPart], validation_parts: list[NetworkPart]) -> dict[str, Any]:
    columns = train_parts[0].record.columns
    train_exact: set[int] = set()
    validation_exact: set[int] = set()
    train_keys: set[tuple[str, ...]] = set()
    validation_keys: set[tuple[str, ...]] = set()
    key_columns = ["src_ip", "src_port", "dst_ip", "dst_port", "proto", "service"]
    for part, destination, key_set in [(part, train_exact, train_keys) for part in train_parts] + [(part, validation_exact, validation_keys) for part in validation_parts]:
        fingerprints = pd.util.hash_pandas_object(part.frame[columns], index=False).to_numpy(dtype="uint64")
        destination.update(int(value) for value in fingerprints)
        available = [column for column in key_columns if column in part.frame.columns]
        if len(available) == len(key_columns):
            key_set.update(tuple("" if pd.isna(value) else str(value) for value in row) for row in part.frame[available].itertuples(index=False, name=None))
    return {
        "train_rows": sum(len(part.frame) for part in train_parts),
        "validation_rows": sum(len(part.frame) for part in validation_parts),
        "exact_fingerprint_intersection": len(train_exact & validation_exact),
        "connection_key_intersection_proxy": len(train_keys & validation_keys),
        "connection_key_definition": ",".join(key_columns),
        "status": "PASS" if not (train_exact & validation_exact) else "FAIL",
        "semantic_session_identity": "UNVERIFIED; Network schema has no session/run identifier",
    }


def load_protocol_dataset(
    project_root: Path = PROJECT_ROOT,
    train_prefix_rows: int = 2_000,
    validation_prefix_rows: int = 8_000,
    validation_target_rows: int = 4_000,
    chunk_size: int = 250_000,
) -> ProtocolDataset:
    train_records, validation_records, test_records = network_records(project_root)
    if not train_records or not validation_records:
        raise RuntimeError("Network source-file holdout does not contain Train and Validation records")
    feature_columns = phase2_feature_schema(train_records, project_root)
    train_parts = [
        _make_part(record, read_prefix(record.file_path, train_prefix_rows, chunk_size), "train", feature_columns)
        for record in train_records
    ]
    validation_parts = [
        _make_part(record, read_prefix(record.file_path, validation_prefix_rows, chunk_size), "validation", feature_columns)
        for record in validation_records
    ]
    train_graph = _combine_parts(train_parts, feature_columns)
    validation_graphs: dict[str, Any] = {}
    target_event_ids: dict[str, list[str]] = {}
    cutoffs: dict[str, float] = {}
    origin = float(train_graph.events["timestamp"].min())
    for part in validation_parts:
        probe = part.events[part.events["source_row"] < validation_target_rows]
        if probe.empty:
            raise RuntimeError(f"No validation probe rows for {part.record.file_name}")
        target_snapshot = int(np.floor((float(probe["timestamp"].max()) - origin) / 30))
        target_start = origin + target_snapshot * 30
        cutoff = target_start + 30
        graph = _combine_parts(train_parts + [part], feature_columns, cutoff=cutoff)
        targets = graph.events[
            (graph.events["source_file"] == part.record.file_name)
            & (graph.events["timestamp"] >= target_start)
            & (graph.events["timestamp"] < cutoff)
        ]["event_id"].tolist()
        if not targets:
            raise RuntimeError(f"No target events in causal validation window for {part.record.file_name}")
        validation_graphs[part.record.file_name] = graph
        target_event_ids[part.record.file_name] = targets
        cutoffs[part.record.file_name] = cutoff
    return ProtocolDataset(
        train_parts=train_parts,
        validation_parts=validation_parts,
        train_graph=train_graph,
        validation_graphs=validation_graphs,
        target_event_ids=target_event_ids,
        cutoffs=cutoffs,
        feature_columns=feature_columns,
        duplicate_audit=_duplicate_audit(train_parts, validation_parts),
        source_records=train_records + validation_records + test_records,
    )

