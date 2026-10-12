"""Unit tests for the bounded Phase 3A graph conversion invariants."""

from __future__ import annotations

import numpy as np
import pandas as pd

from research_pipeline.src.experimental_phase2.network_graph_v1 import assign_half_open_snapshot
from research_pipeline.src.experimental_phase3a.conversion import (
    _causal_maps,
    _edge_index,
    _future_removal_test,
    _snapshot_sequences,
)


def _events() -> pd.DataFrame:
    return pd.DataFrame([
        {"event_id": "event::f::000000000", "source_file": "f", "source_row": 0, "timestamp": 100.0, "label": 0, "src_ip": "a", "dst_ip": "b", "proto": "tcp", "service": "http"},
        {"event_id": "event::f::000000001", "source_file": "f", "source_row": 1, "timestamp": 130.0, "label": 1, "src_ip": "a", "dst_ip": "c", "proto": "tcp", "service": "http"},
        {"event_id": "event::f::000000002", "source_file": "f", "source_row": 2, "timestamp": 1_000.0, "label": 1, "src_ip": "z", "dst_ip": "c", "proto": "udp", "service": "dns"},
    ])


def _edges() -> pd.DataFrame:
    return pd.DataFrame([
        {"src_type": "ip", "src_id": "ip::a", "relation": "source_ip_to_event", "dst_type": "event", "dst_id": "event::f::000000000", "event_id": "event::f::000000000", "source_file": "f", "source_row": 0, "timestamp": 100.0},
        {"src_type": "event", "src_id": "event::f::000000000", "relation": "event_to_destination_ip", "dst_type": "ip", "dst_id": "ip::b", "event_id": "event::f::000000000", "source_file": "f", "source_row": 0, "timestamp": 100.0},
        {"src_type": "ip", "src_id": "ip::a", "relation": "source_ip_to_event", "dst_type": "event", "dst_id": "event::f::000000001", "event_id": "event::f::000000001", "source_file": "f", "source_row": 1, "timestamp": 130.0},
        {"src_type": "event", "src_id": "event::f::000000001", "relation": "event_to_destination_ip", "dst_type": "ip", "dst_id": "ip::c", "event_id": "event::f::000000001", "source_file": "f", "source_row": 1, "timestamp": 130.0},
    ])


def test_half_open_boundary_is_deterministic():
    assert assign_half_open_snapshot(0.0, 0.0, 30) == 0
    assert assign_half_open_snapshot(30.0, 0.0, 30) == 1
    assert assign_half_open_snapshot(60.0, 0.0, 30) == 2


def test_causal_maps_and_reverse_edges_keep_local_indexes_unique():
    events = _events()
    maps, values = _causal_maps(events)
    assert maps["event"][events.iloc[0].event_id] == 0
    assert len(values["ip"]) == 4
    indexed = _edge_index(_edges(), maps)
    forward = indexed[("ip", "source_ip_to_event", "event")]
    reverse = indexed[("event", "rev_source_ip_to_event", "ip")]
    assert forward.shape == (2, 2)
    assert {(int(dst), int(src)) for src, dst in forward.T.tolist()} == {tuple(pair) for pair in reverse.T.tolist()}


def test_snapshot_gap_is_not_filled_with_fake_steps():
    snapshots, metadata = _snapshot_sequences(_events(), _edges(), window_seconds=30)
    assert metadata["sequence_count"] == 2
    assert int(snapshots["sequence_id"].nunique()) == 2
    assert snapshots["preserved_gap"].any()
    assert snapshots["interval_semantics"].eq("[start,end)").all()


def test_future_removal_preserves_past_identity_and_edges():
    result = _future_removal_test(_events(), _edges())
    assert result["status"] == "PASS"
    assert result["past_node_ids_stable"] is True
    assert result["past_raw_edge_count_equal"] is True

