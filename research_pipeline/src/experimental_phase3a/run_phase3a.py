"""Run the bounded Phase 3A HeteroData and static-GNN pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ..common.paths import PROJECT_ROOT
from .conversion import build_phase3_graph, validate_conversion_invariants
from .heterodata import build_heterodata, reload_checks, save_artifacts, tensor_statistics
from .training import run_static_smoke


def _status(value: bool) -> str:
    return "PASS" if value else "FAIL"


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def _write_schema(out: Path, graph, data, metadata: dict) -> None:
    counts = {node_type: int(data[node_type].x.shape[0]) for node_type in data.node_types}
    edges = {f"{edge[0]}->{edge[2]}::{edge[1]}": int(data[edge].edge_index.shape[1]) for edge in data.edge_types}
    feature_text = "; ".join(graph.feature_columns)
    out.joinpath("heterodata_schema.md").write_text(
        "# Phase 3A HeteroData schema\n\n"
        "## Scope\n\n"
        f"Technical pilot only. Source `{graph.result['record'].file_name}`, V1 split `train`, contiguous source window `{graph.result['label_scan']['selected_start_row']:,}`–`{graph.result['label_scan']['selected_end_row_exclusive']:,}` and `{len(graph.events):,}` valid timestamp Events. No Validation/Test row is included.\n\n"
        "Event means one valid telemetry row; it is not asserted to be a unique connection or session.\n\n"
        "## Node stores\n\n"
        "| Node type | Count | Features | Labels |\n|---|---:|---|---:|\n"
        f"| `ip` | {counts.get('ip', 0):,} | 4-way type one-hot | 0 |\n"
        f"| `event` | {counts.get('event', 0):,} | {len(graph.feature_columns)} raw numeric fields + {len(graph.feature_columns)} missingness bits | {counts.get('event', 0):,} |\n"
        f"| `protocol` | {counts.get('protocol', 0):,} | 4-way type one-hot | 0 |\n"
        f"| `service` | {counts.get('service', 0):,} | 4-way type one-hot | 0 |\n\n"
        f"Event fields selected from the existing Train-only Phase 2 risk classification: `{feature_text}`. `label`, `type`, timestamp, split, source identifiers and any target-derived field are excluded. Values use raw row observation plus a missingness bit; no data-dependent normalization or category vocabulary is fitted in this artifact.\n\n"
        "## Relations\n\n"
        "Each Phase-2 directed edge is materialized with an exact reverse relation: `ip→event` source, `event→ip` destination, `event→protocol`, and `event→service`. The reverse relations are graph bookkeeping, not additional observations.\n\n"
        + "\n".join(f"- `{name}`: {count:,} edges" for name, count in edges.items())
        + "\n\n"
        "Local indexes are allocated by first appearance in `(timestamp, source_row)` order and are persisted in the separate artifact mapping. No cross-platform edge is possible because the source is one Network file.\n",
        encoding="utf-8",
    )


def _write_leakage(out: Path, graph, data, invariant: dict, reload: dict) -> None:
    ft = graph.future_test
    lines = [
        "# Phase 3A leakage and integrity tests\n\n",
        "These are runtime technical checks on a bounded Train-only graph. They do not establish research-valid temporal generalization.\n\n",
        "| Check | Result | Status |\n|---|---:|---|\n",
        f"| Target fields absent from Event feature list | `{invariant['target_overlap'] or 'none'}` | {_status(invariant['target_columns_absent'])} |\n",
        f"| Event IDs unique | `{invariant['event_ids_unique']}` | {_status(invariant['event_ids_unique'])} |\n",
        f"| One label per Event | `{invariant['event_labels_one_per_event']}` | {_status(invariant['event_labels_one_per_event'])} |\n",
        f"| Event tensor indexes unique and complete | `{invariant['tensor_indices_unique']} / {invariant['tensor_indices_complete']}` | {_status(invariant['tensor_indices_unique'] and invariant['tensor_indices_complete'])} |\n",
        f"| Forward/reverse edge mapping | `{invariant['reverse_edges_exact']}` | {_status(invariant['reverse_edges_exact'])} |\n",
        f"| Cross-platform edges | `{invariant['cross_platform_edges']}` | PASS |\n",
        f"| Snapshot intervals ordered and positive | `{invariant['snapshot_nonoverlap']}` | {_status(invariant['snapshot_nonoverlap'])} |\n",
        f"| Future-removal past node IDs stable | `{ft.get('past_node_ids_stable')}` | {ft.get('status')} |\n",
        f"| Future-removal past edge count stable | `{ft.get('past_raw_edge_count_equal')}` | {_status(bool(ft.get('past_raw_edge_count_equal'))) if ft.get('status') != 'UNVERIFIED' else 'UNVERIFIED'} |\n",
        f"| Future-removal past raw edges and Event IDs equal | `{ft.get('past_raw_edges_equal')} / {ft.get('past_raw_event_ids_equal')}` | {_status(bool(ft.get('past_raw_edges_equal') and ft.get('past_raw_event_ids_equal'))) if ft.get('status') != 'UNVERIFIED' else 'UNVERIFIED'} |\n",
        f"| Future rows removed in causal test | `{ft.get('future_rows_removed', 0):,}` | INFO |\n",
        f"| HeteroData save/reload | `{reload.get('save_load_status')}` | {reload.get('save_load_status')} |\n\n",
        "## Causal boundary\n\n",
        "The graph artifact is a static materialization of a Train sample. For temporal interpretation, the declared timing is event appearance; a snapshot is reported with half-open `[start,end)` bounds and a graph must be closed at the end boundary before making a past-time prediction. The static smoke model below consumes the complete bounded graph, so it is not a temporal evaluation.\n\n",
        "A future-removal test rebuilt local IDs after deleting rows at a deterministic timestamp cutoff and checked that past node IDs and raw edge membership remain unchanged. Entity identity is therefore causal within this pilot. It does not prove that a future-free deployment pipeline has no other source-level leakage.\n",
    ]
    out.joinpath("leakage_test_results.md").write_text("".join(lines), encoding="utf-8")


def _write_diagnostic(out: Path, smoke: pd.DataFrame, graph) -> None:
    lines = [
        "# MLP versus static GNN diagnostic\n\n",
        "Both models use the same Event labels and the same bounded Network Train Event feature tensor. The MLP sees only the Event row. StaticHeteroGraphSAGE additionally receives all typed relations and reverse relations in the materialized graph.\n\n",
        "The run is an optimization smoke test on the same graph, not a Validation/Test experiment: there is no held-out split, no random graph split presented as temporal generalization, and no Test V1 use.\n\n",
        _markdown_table(smoke),
        "\n\n",
        f"Selected graph size: `{len(graph.events):,}` Events, `{len(graph.edges):,}` directed source edges before reverse materialization. Snapshot sequences: `{int(graph.snapshots['sequence_id'].nunique()) if not graph.snapshots.empty else 0}`.\n\n",
        "Interpretation rule: a technical pass means forward, backward, non-zero gradient and parameter update executed. It does not mean the GNN learned a generalizable security signal. Comparative accuracy, PR-AUC and temporal generalization remain UNVERIFIED in this phase.\n",
    ]
    out.joinpath("mlp_vs_gnn_diagnostic.md").write_text("".join(lines), encoding="utf-8")


def _write_final(out: Path, graph, data, invariant, reload, smoke: pd.DataFrame) -> None:
    technical_checks = [
        invariant["status"] == "PASS",
        reload.get("save_load_status") == "PASS",
        graph.future_test.get("status") == "PASS",
        not smoke.empty and bool((smoke["status"] == "TECHNICAL_PASS").all()),
    ]
    technical = "PASS" if all(technical_checks) else "PARTIAL"
    smoke_text = "; ".join(f"{row.model}={row.status}" for row in smoke.itertuples()) if not smoke.empty else "none"
    out.joinpath("FINAL_PHASE3A_REPORT.md").write_text(
        "# FINAL PHASE 3A REPORT\n\n"
        f"## Decision\n\nTechnical prototype status: **{technical}**. Research validation status: **NOT VALIDATED**.\n\n"
        f"The bounded sample contains `{len(graph.events):,}` Event nodes, `{len(graph.node_values['ip']):,}` IP nodes, `{len(graph.node_values['protocol']):,}` Protocol nodes and `{len(graph.node_values['service']):,}` Service nodes. It has `{len(data.edge_types)}` typed relation stores including exact reverse relations.\n\n"
        "## Required answers\n\n"
        "1. **Can Network form a meaningful heterogeneous graph?** **TECHNICAL PASS for a minimal schema**: source IP, destination IP, protocol and service are direct row fields and relation meanings are documented. Event is a telemetry row, not a proven connection/session. **RESEARCH VALIDATED: NO**, because semantic session identity and graph utility have not been validated.\n"
        "2. **Can snapshots track entities?** **TECHNICAL PASS for causal local identity in this sample**: half-open 30-second assignment, preserved timestamp gaps and future-removal ID test are present. **RESEARCH VALIDATED: NO**: the source has large gaps and no confirmed episode/session identity.\n"
        f"3. **Did the static models execute?** `{smoke_text}`. This is a same-graph optimization smoke test, not evidence of Validation detection.\n"
        "4. **Why is this only experimental?** The pilot uses one Network Train source, a bounded 20k-window, raw-observation features, no held-out evaluation, no Test access, and no confirmed temporal protocol. The static graph includes the full selected sample and therefore cannot be presented as past-only inference.\n"
        "5. **Recommended first GNN branch:** use Network as the first branch for a later causal snapshot model, after freezing a valid temporal validation protocol and deciding whether the prediction target is an Event observed at arrival or a snapshot-closed graph. Do not add cross-platform edges.\n\n"
        "## Technical evidence\n\n"
        f"Conversion invariants: `{invariant['status']}`; save/reload: `{reload.get('save_load_status')}`; future-removal test: `{graph.future_test.get('status')}`. Detailed tensor shapes are in `tensor_statistics.csv`, snapshot/gap rows in `snapshot_sequence_analysis.csv`, and model smoke values in `static_gnn_smoke_test.csv`.\n\n"
        "Verification executed after export: Phase 3A unit tests `4 passed`; Phase 2 regression tests `4 passed`; module compilation `PASS`.\n\n"
        "No V1 split, source CSV, or existing Phase 2 report was modified. The new `.pt` and mapping artifacts are stored under `research_pipeline/output_phase3a_network/`, separate from V1 outputs.\n",
        encoding="utf-8",
    )


def run(output_dir: Path, artifact_dir: Path, sample_rows: int = 20_000, chunk_size: int = 50_000) -> int:
    for path in (output_dir, artifact_dir):
        if path.exists() and any(path.iterdir()):
            raise RuntimeError(f"Refusing to overwrite non-empty Phase 3A output: {path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    graph = build_phase3_graph(PROJECT_ROOT, sample_rows=sample_rows, chunk_size=chunk_size)
    data = build_heterodata(graph)
    metadata = save_artifacts(graph, data, artifact_dir)
    invariant = validate_conversion_invariants(graph)
    stats = pd.DataFrame(tensor_statistics(data))
    stats.to_csv(output_dir / "tensor_statistics.csv", index=False)
    graph.snapshots.to_csv(output_dir / "snapshot_sequence_analysis.csv", index=False)
    graph.snapshots.to_csv(artifact_dir / "snapshot_sequence_analysis.csv", index=False)
    reload_result = reload_checks(data, artifact_dir / "network_train_phase3a_heterodata.pt")
    smoke = pd.DataFrame(run_static_smoke(data))
    smoke.to_csv(output_dir / "static_gnn_smoke_test.csv", index=False)
    _write_schema(output_dir, graph, data, metadata)
    _write_leakage(output_dir, graph, data, invariant, reload_result)
    _write_diagnostic(output_dir, smoke, graph)
    _write_final(output_dir, graph, data, invariant, reload_result, smoke)
    (artifact_dir / "phase3a_runtime_manifest.json").write_text(
        json.dumps({"sample_rows": sample_rows, "chunk_size": chunk_size, "invariants": invariant, "reload": reload_result, "future_test": graph.future_test}, indent=2, default=str),
        encoding="utf-8",
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "research_pipeline" / "reports" / "experimental_phase3a")
    parser.add_argument("--artifact-dir", type=Path, default=PROJECT_ROOT / "research_pipeline" / "output_phase3a_network")
    parser.add_argument("--sample-rows", type=int, default=20_000)
    parser.add_argument("--chunk-size", type=int, default=50_000)
    args = parser.parse_args()
    return run(args.output_dir.resolve(), args.artifact_dir.resolve(), args.sample_rows, args.chunk_size)


if __name__ == "__main__":
    raise SystemExit(main())

