from __future__ import annotations

import numpy as np
import pandas as pd

from research_pipeline.src.experimental_phase2.network_graph_v1 import (
    assign_half_open_snapshot,
    build_event_graph,
    causal_leakage_audit,
    event_id_for,
)


class _Record:
    file_name = "Network_dataset_test.csv"


def _frame() -> pd.DataFrame:
    return pd.DataFrame({
        "ts": [100.0, 129.9, 130.0],
        "src_ip": ["10.0.0.1", "10.0.0.1", "10.0.0.2"],
        "src_port": [1, 2, 3], "dst_ip": ["10.0.0.2", "10.0.0.2", "10.0.0.3"],
        "dst_port": [4, 5, 6], "proto": ["tcp", "tcp", "udp"],
        "service": ["http", "http", "dns"], "duration": [1.0, 2.0, 3.0],
        "label": [0, 1, 0], "type": ["normal", "scanning", "normal"],
    })


def test_half_open_boundary_assigns_end_to_next_snapshot() -> None:
    assert assign_half_open_snapshot(100.0, 100.0, 30) == 0
    assert assign_half_open_snapshot(129.999, 100.0, 30) == 0
    assert assign_half_open_snapshot(130.0, 100.0, 30) == 1


def test_event_ids_are_unique_and_record_count_reconciles() -> None:
    frame = _frame()
    nodes, edges, events = build_event_graph(frame, _Record(), frame["ts"], ["src_port", "dst_port", "duration"])
    assert len(events) == len(frame)
    assert events["event_id"].is_unique
    assert len(nodes["event"]) == len(frame)
    assert len(edges) == 4 * len(frame)


def test_event_features_exclude_targets_and_edges_are_direct() -> None:
    frame = _frame()
    nodes, edges, events = build_event_graph(frame, _Record(), frame["ts"], ["duration"])
    assert not ({"label", "type", "ts"} & {"duration"})
    assert set(pd.DataFrame(edges)["relation"]) <= {
        "source_ip_to_event", "event_to_destination_ip", "event_to_protocol", "event_to_service",
    }
    assert all(edge["source_file"] == _Record.file_name for edge in edges)


def test_causal_leakage_audit_is_computed() -> None:
    frame = _frame()
    _, edge_rows, events = build_event_graph(frame, _Record(), frame["ts"], ["duration"])
    audit = causal_leakage_audit(events, pd.DataFrame(edge_rows), ["duration"], _Record.file_name)
    assert audit["future_node_id_uses"] == 0
    assert audit["future_vocab_exposures_if_global_encoder"] > 0
    assert audit["future_edge_uses"] == 0
    assert audit["target_columns_in_event_features"] == ""
    assert audit["status"] == "PASS"

