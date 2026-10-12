"""PyTorch Geometric tensor materialization for the bounded graph."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch_geometric.data import HeteroData

from .conversion import CausalGraph


TYPE_ORDER = ("ip", "event", "protocol", "service")


def build_heterodata(graph: CausalGraph) -> HeteroData:
    data = HeteroData()
    event_count = len(graph.node_values["event"])
    for node_type in TYPE_ORDER:
        if node_type == "event":
            data[node_type].x = torch.from_numpy(graph.event_x)
            data[node_type].y = torch.from_numpy(
                graph.events.set_index("event_id").loc[graph.node_values["event"], "label"].to_numpy(dtype=np.int64, copy=True)
            )
        else:
            # Same-width type one-hot features preserve category identity while
            # keeping non-event node identities outside the feature values.
            type_index = TYPE_ORDER.index(node_type)
            x = np.zeros((len(graph.node_values[node_type]), len(TYPE_ORDER)), dtype=np.float32)
            if len(x):
                x[:, type_index] = 1.0
            data[node_type].x = torch.from_numpy(x)
    for relation, values in graph.edge_index.items():
        src_type, rel, dst_type = relation
        data[src_type, rel, dst_type].edge_index = torch.from_numpy(values).long()
    data["event"].num_nodes = event_count
    return data


def tensor_statistics(data: HeteroData) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for node_type in data.node_types:
        store = data[node_type]
        x = store.x
        y = getattr(store, "y", None)
        rows.append({
            "kind": "node",
            "type": node_type,
            "relation": "",
            "count": int(x.shape[0]),
            "feature_shape": str(tuple(x.shape)),
            "feature_dtype": str(x.dtype),
            "label_count": int(y.shape[0]) if y is not None else 0,
            "edge_shape": "",
            "edge_dtype": "",
            "status": "PASS" if x.dtype == torch.float32 and (y is None or y.dtype == torch.int64) else "FAIL",
        })
    for edge_type in data.edge_types:
        store = data[edge_type]
        edge_index = store.edge_index
        rows.append({
            "kind": "edge",
            "type": f"{edge_type[0]}->{edge_type[2]}",
            "relation": edge_type[1],
            "count": int(edge_index.shape[1]),
            "feature_shape": "",
            "feature_dtype": "",
            "label_count": 0,
            "edge_shape": str(tuple(edge_index.shape)),
            "edge_dtype": str(edge_index.dtype),
            "status": "PASS" if edge_index.dtype == torch.int64 and edge_index.shape[0] == 2 else "FAIL",
        })
    return rows


def save_artifacts(graph: CausalGraph, data: HeteroData, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tensor_path = output_dir / "network_train_phase3a_heterodata.pt"
    torch.save(data, tensor_path)
    graph.event_mapping.to_csv(output_dir / "event_mapping.csv", index=False)
    node_rows = [
        {"node_type": node_type, "raw_node_id": raw_id, "tensor_index": index}
        for node_type, mapping in graph.node_maps.items()
        for raw_id, index in mapping.items()
    ]
    import pandas as pd
    pd.DataFrame(node_rows).to_csv(output_dir / "node_index_mapping.csv", index=False)
    metadata = {
        "feature_columns": graph.feature_columns,
        "event_feature_policy": "raw numeric observation + missingness bit; no fit-dependent normalization",
        "node_types": list(data.node_types),
        "edge_types": [list(value) for value in data.edge_types],
        "event_count": int(data["event"].x.shape[0]),
        "event_feature_dim": int(data["event"].x.shape[1]),
        "source_file": str(graph.result["record"].file_name),
        "split": "train",
        "timestamp_window": "30 seconds, half-open [start,end)",
        "prediction_timing": graph.future_test.get("prediction_timing"),
        "artifact": str(tensor_path),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def reload_checks(data: HeteroData, tensor_path: Path) -> dict[str, Any]:
    try:
        loaded = torch.load(tensor_path, map_location="cpu", weights_only=False)
        node_counts_equal = all(loaded[node].x.shape == data[node].x.shape for node in data.node_types)
        edge_counts_equal = all(loaded[edge].edge_index.shape == data[edge].edge_index.shape for edge in data.edge_types)
        labels_equal = torch.equal(loaded["event"].y, data["event"].y)
        return {
            "save_load_status": "PASS" if node_counts_equal and edge_counts_equal and labels_equal else "FAIL",
            "node_shapes_equal": node_counts_equal,
            "edge_shapes_equal": edge_counts_equal,
            "event_labels_equal": labels_equal,
        }
    except Exception as exc:  # pragma: no cover - runtime compatibility path
        return {"save_load_status": "FAIL", "error": str(exc)}

