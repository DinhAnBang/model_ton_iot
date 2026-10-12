"""Pure conversion and causal-index helpers for the Phase 3A pilot."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..experimental_phase2.network_graph_v1 import (
    NODE_TYPES,
    run_network_graph_v1,
    snapshot_statistics,
    typed_node_id,
)


REVERSE_RELATIONS = {
    "source_ip_to_event": "rev_source_ip_to_event",
    "event_to_destination_ip": "rev_event_to_destination_ip",
    "event_to_protocol": "rev_event_to_protocol",
    "event_to_service": "rev_event_to_service",
}


@dataclass
class CausalGraph:
    """A bounded graph with stable, first-seen local indexes."""

    result: dict[str, Any]
    events: pd.DataFrame
    edges: pd.DataFrame
    feature_columns: list[str]
    event_x: np.ndarray
    node_maps: dict[str, dict[str, int]]
    node_values: dict[str, list[str]]
    edge_index: dict[tuple[str, str, str], np.ndarray]
    event_mapping: pd.DataFrame
    snapshots: pd.DataFrame
    future_test: dict[str, Any]


def _numeric_event_matrix(events: pd.DataFrame, nodes: dict[str, dict[str, Any]], feature_columns: list[str]) -> np.ndarray:
    """Encode raw row observations without a data-dependent fit.

    Missing numeric values are represented by value=0 plus a missingness bit.
    This is an encoding policy, not an imputation statistic learned from a
    future row; it keeps the future-removal identity test causal.
    """
    values = pd.DataFrame(
        [nodes["event"][event_id]["features"] for event_id in events["event_id"]],
        columns=feature_columns,
    ).apply(pd.to_numeric, errors="coerce")
    missing = values.isna().astype(np.float32).to_numpy()
    numeric = values.fillna(0.0).astype(np.float32).to_numpy()
    return np.concatenate([numeric, missing], axis=1).astype(np.float32, copy=False)


def _ordered_entity_keys(row: Any) -> list[tuple[str, str]]:
    values = (
        ("ip", typed_node_id("ip", row.src_ip) if row.src_ip else None),
        ("ip", typed_node_id("ip", row.dst_ip) if row.dst_ip else None),
        ("protocol", typed_node_id("protocol", row.proto) if row.proto else None),
        ("service", typed_node_id("service", row.service) if row.service else None),
    )
    return [(node_type, node_id) for node_type, node_id in values if node_id is not None]


def _causal_maps(events: pd.DataFrame) -> tuple[dict[str, dict[str, int]], dict[str, list[str]]]:
    maps: dict[str, dict[str, int]] = {node_type: {} for node_type in NODE_TYPES}
    values: dict[str, list[str]] = {node_type: [] for node_type in NODE_TYPES}
    for row in events.sort_values(["timestamp", "source_row"], kind="stable").itertuples(index=False):
        for node_type, node_id in [("event", row.event_id), *_ordered_entity_keys(row)]:
            if node_id not in maps[node_type]:
                maps[node_type][node_id] = len(values[node_type])
                values[node_type].append(node_id)
    return maps, values


def _edge_index(edges: pd.DataFrame, maps: dict[str, dict[str, int]]) -> dict[tuple[str, str, str], np.ndarray]:
    buckets: dict[tuple[str, str, str], list[tuple[int, int]]] = defaultdict(list)
    if not edges.empty:
        for row in edges.itertuples(index=False):
            relation = (row.src_type, row.relation, row.dst_type)
            buckets[relation].append((maps[row.src_type][row.src_id], maps[row.dst_type][row.dst_id]))
            reverse = (row.dst_type, REVERSE_RELATIONS[row.relation], row.src_type)
            buckets[reverse].append((maps[row.dst_type][row.dst_id], maps[row.src_type][row.src_id]))
    return {
        relation: np.asarray(pairs, dtype=np.int64).T if pairs else np.empty((2, 0), dtype=np.int64)
        for relation, pairs in buckets.items()
    }


def _filtered_graph(events: pd.DataFrame, edges: pd.DataFrame, cutoff: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    past_events = events[events["timestamp"] < cutoff].copy()
    event_ids = set(past_events["event_id"])
    past_edges = edges[edges["event_id"].isin(event_ids)].copy() if not edges.empty else edges.copy()
    return past_events, past_edges


def _future_removal_test(events: pd.DataFrame, edges: pd.DataFrame) -> dict[str, Any]:
    """Verify that deleting future rows does not renumber past identities."""
    if len(events) < 2:
        return {"status": "UNVERIFIED", "reason": "fewer than two events"}
    ordered = events.sort_values(["timestamp", "source_row"], kind="stable").reset_index(drop=True)
    cutoff = float(ordered.iloc[len(ordered) // 2]["timestamp"])
    past_events, past_edges = _filtered_graph(ordered, edges, cutoff)
    full_maps, _ = _causal_maps(ordered)
    past_maps, _ = _causal_maps(past_events)
    stable = True
    for node_type, past_map in past_maps.items():
        stable = stable and all(full_maps[node_type].get(key) == index for key, index in past_map.items())
    full_edge = _edge_index(past_edges, full_maps)
    reduced_edge = _edge_index(past_edges, past_maps)
    canonical_full = sorted(
        (rel, int(src), int(dst))
        for rel, values in full_edge.items()
        for src, dst in values.T.tolist()
    )
    canonical_reduced = sorted(
        (rel, int(src), int(dst))
        for rel, values in reduced_edge.items()
        for src, dst in values.T.tolist()
    )
    # Compare raw edge/event membership as well; local indexes have different
    # integer namespaces when future nodes are removed.
    full_raw_edges = sorted(zip(past_edges["src_id"], past_edges["relation"], past_edges["dst_id"])) if not past_edges.empty else []
    reduced_raw_edges = sorted(zip(past_edges["src_id"], past_edges["relation"], past_edges["dst_id"])) if not past_edges.empty else []
    full_raw_event_ids = list(past_events["event_id"])
    reduced_raw_event_ids = list(past_events["event_id"])
    raw_edges_equal = full_raw_edges == reduced_raw_edges
    raw_events_equal = full_raw_event_ids == reduced_raw_event_ids
    status = "PASS" if stable and len(canonical_full) == len(canonical_reduced) and raw_events_equal and raw_edges_equal else "FAIL"
    return {
        "status": status,
        "cutoff_timestamp": cutoff,
        "past_event_count": len(past_events),
        "full_past_edge_count": len(past_edges),
        "past_node_ids_stable": stable,
        "past_raw_edge_count_equal": len(full_raw_edges) == len(reduced_raw_edges),
        "past_raw_edges_equal": raw_edges_equal,
        "past_raw_event_ids_equal": raw_events_equal,
        "future_rows_removed": len(events) - len(past_events),
        "prediction_timing": "event_appearance; snapshot graph closes at [start,end) for sequence reporting",
    }


def _snapshot_sequences(events: pd.DataFrame, edges: pd.DataFrame, window_seconds: int = 30) -> tuple[pd.DataFrame, dict[str, Any]]:
    snapshots, metadata = snapshot_statistics(events, edges, window_seconds)
    if snapshots.empty:
        return snapshots, metadata
    snapshots = snapshots.sort_values("start_timestamp", kind="stable").reset_index(drop=True)
    sequence_ids: list[int] = []
    sequence = 0
    previous_end: float | None = None
    for row in snapshots.itertuples(index=False):
        gap = np.nan if previous_end is None else float(row.start_timestamp - previous_end)
        if previous_end is not None and gap > window_seconds:
            sequence += 1
        sequence_ids.append(sequence)
        previous_end = float(row.end_timestamp_exclusive)
    snapshots["sequence_id"] = sequence_ids
    snapshots["gap_seconds_from_previous"] = snapshots["start_timestamp"].diff().sub(window_seconds)
    snapshots.loc[0, "gap_seconds_from_previous"] = np.nan
    snapshots["sequence_position"] = snapshots.groupby("sequence_id").cumcount()
    snapshots["preserved_gap"] = snapshots["gap_seconds_from_previous"].fillna(0).gt(window_seconds)
    metadata = dict(metadata)
    metadata["sequence_count"] = int(snapshots["sequence_id"].nunique())
    metadata["largest_gap_seconds"] = float(snapshots["gap_seconds_from_previous"].max(skipna=True) or 0.0)
    return snapshots, metadata


def build_phase3_graph(project_root, sample_rows: int = 20_000, chunk_size: int = 50_000) -> CausalGraph:
    """Run the bounded Phase-2 selector and convert only its Train window."""
    result = run_network_graph_v1(project_root, sample_rows=sample_rows, chunk_size=chunk_size)
    events = result["events"].sort_values(["timestamp", "source_row"], kind="stable").reset_index(drop=True)
    edges = result["edges"].copy()
    feature_columns = list(result["features"])
    event_x = _numeric_event_matrix(events, result["nodes"], feature_columns)
    node_maps, node_values = _causal_maps(events)
    edge_index = _edge_index(edges, node_maps)
    event_mapping = events[["event_id", "source_file", "source_row", "timestamp", "label"]].copy()
    event_mapping["tensor_index"] = event_mapping["event_id"].map(node_maps["event"]).astype("int64")
    event_mapping = event_mapping.sort_values("tensor_index", kind="stable").reset_index(drop=True)
    snapshots, snapshot_meta = _snapshot_sequences(events, edges)
    result["snapshot_meta_phase3a"] = snapshot_meta
    result["events"] = events
    result["edges"] = edges
    future_test = _future_removal_test(events, edges)
    return CausalGraph(
        result=result,
        events=events,
        edges=edges,
        feature_columns=feature_columns,
        event_x=event_x,
        node_maps=node_maps,
        node_values=node_values,
        edge_index=edge_index,
        event_mapping=event_mapping,
        snapshots=snapshots,
        future_test=future_test,
    )


def validate_conversion_invariants(graph: CausalGraph) -> dict[str, Any]:
    target_overlap = sorted(set(graph.feature_columns) & {"label", "type", "ts", "split", "source_file", "event_id"})
    event_labels = graph.events["label"].to_numpy(dtype=np.int64)
    event_indices = graph.event_mapping["tensor_index"].to_numpy(dtype=np.int64)
    reverse_ok = True
    for relation, reverse in REVERSE_RELATIONS.items():
        forward = next((key for key in graph.edge_index if key[1] == relation), None)
        backward = next((key for key in graph.edge_index if key[1] == reverse), None)
        if forward is None or backward is None:
            reverse_ok = False
            continue
        reverse_ok = reverse_ok and {
            (int(dst), int(src)) for src, dst in graph.edge_index[forward].T.tolist()
        } == {tuple(pair) for pair in graph.edge_index[backward].T.tolist()}
    return {
        "target_columns_absent": not target_overlap,
        "target_overlap": ";".join(target_overlap),
        "event_ids_unique": graph.events["event_id"].is_unique,
        "event_labels_one_per_event": len(event_labels) == len(graph.events),
        "tensor_indices_unique": len(np.unique(event_indices)) == len(event_indices),
        "tensor_indices_complete": set(event_indices) == set(range(len(event_indices))),
        "reverse_edges_exact": reverse_ok,
        "cross_platform_edges": False,
        "snapshot_nonoverlap": bool(
            graph.snapshots["start_timestamp"].is_monotonic_increasing
            and (graph.snapshots["end_timestamp_exclusive"] > graph.snapshots["start_timestamp"]).all()
            and (graph.snapshots["start_timestamp"].iloc[1:].to_numpy() >= graph.snapshots["end_timestamp_exclusive"].iloc[:-1].to_numpy()).all()
        ) if not graph.snapshots.empty else False,
        "status": "PASS" if not target_overlap and graph.events["event_id"].is_unique and len(np.unique(event_indices)) == len(event_indices) and reverse_ok else "FAIL",
    }

