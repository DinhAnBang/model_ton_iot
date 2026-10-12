"""Execute the bounded causal Static GNN Phase 3B evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from ..common.paths import PROJECT_ROOT
from ..experimental_phase3a.heterodata import build_heterodata
from ..experimental_phase3a.models import EventMLP, StaticHeteroGraphSAGE
from .dataset import (
    _combine_parts,
    _make_part,
    load_protocol_dataset,
)
from .evaluation import EvaluationRun, majority_baseline, train_and_validate
from .preprocessing import (
    feature_schema_fingerprint,
    fit_train_scaler,
    preprocessing_diagnostics,
    transform_graph_features,
)


def _ensure_edge_schema(data_objects: list[HeteroData]) -> list[tuple[str, str, str]]:
    relations = sorted({edge_type for data in data_objects for edge_type in data.edge_types})
    for data in data_objects:
        for edge_type in relations:
            if edge_type not in data.edge_types:
                data[edge_type].edge_index = torch.empty((2, 0), dtype=torch.long)
    return relations


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(value) for value in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def _graph_bundle(graph, data: HeteroData) -> dict[str, Any]:
    return {"data": data, "mapping": graph.event_mapping, "graph": graph}


def _class_rows(dataset) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    train = dataset.train_graph.events
    rows.append({
        "scope": "train_fit_graph",
        "source_file": "ALL_V1_TRAIN_FILES_PREFIX",
        "split": "train",
        "total": len(train),
        "normal": int((train["label"] == 0).sum()),
        "attack": int((train["label"] == 1).sum()),
        "normal_rate": float((train["label"] == 0).mean()),
        "attack_rate": float((train["label"] == 1).mean()),
        "attack_types": ";".join(f"{key}:{value}" for key, value in train.loc[train["label"] == 1, "attack_type"].value_counts().sort_index().items()),
        "status": "PASS",
    })
    for source_file, ids in dataset.target_event_ids.items():
        events = dataset.validation_graphs[source_file].events.set_index("event_id").loc[ids]
        rows.append({
            "scope": "causal_validation_target_window",
            "source_file": source_file,
            "split": "validation",
            "total": len(events),
            "normal": int((events["label"] == 0).sum()),
            "attack": int((events["label"] == 1).sum()),
            "normal_rate": float((events["label"] == 0).mean()),
            "attack_rate": float((events["label"] == 1).mean()),
            "attack_types": ";".join(f"{key}:{value}" for key, value in events.loc[events["label"] == 1, "attack_type"].value_counts().sort_index().items()),
            "status": "DIAGNOSTIC",
        })
    return pd.DataFrame(rows)


def _write_preprocessing(out: Path, dataset, scaler, diagnostics, fingerprint: str, artifact_dir: Path) -> None:
    pd.DataFrame(diagnostics["missingness_rows"]).to_csv(out / "missingness_label_audit.csv", index=False)
    (artifact_dir / "scaler.json").write_text(json.dumps(scaler.to_dict(), indent=2), encoding="utf-8")
    (artifact_dir / "feature_schema.json").write_text(json.dumps({"feature_columns": dataset.feature_columns, "event_dim_after_missingness": len(dataset.feature_columns) * 2, "fingerprint": fingerprint}, indent=2), encoding="utf-8")
    lines = [
        "# Feature preprocessing\n\n",
        f"Feature schema: `{len(dataset.feature_columns)}` numeric Event fields from the approved Phase-2 Train-observable schema; each field retains a missingness bit. Final Event width is `{len(dataset.feature_columns) * 2}`.\n\n",
        f"Columns: `{'; '.join(dataset.feature_columns)}`. Target, type, timestamp, split, source identifiers and attack type are excluded.\n\n",
        "## Fit boundary\n\n",
        "Median fill, mean and standard deviation were fit only on the bounded Train graph assembled from fixed source-order prefixes of all 16 V1 Network Train files. Validation and all future rows are transformed with the frozen parameters. No label is used to choose a feature, row, source file or time cutoff.\n\n",
        f"Finite raw-value check: **{diagnostics['finite_raw_values']}**; maximum absolute raw value: `{diagnostics['max_abs_raw_value']:.6g}`; target columns in features: `none`; preprocessing status: **{diagnostics['status']}**.\n\n",
        f"Feature/provenance fingerprint: `{fingerprint}`. The scaler and schema are persisted in `{artifact_dir}`.\n\n",
        "## Missingness review\n\n",
        "Missingness bits are derived from source-field presence and are not labels. The audit reports Normal/Attack missingness rates and checks for a perfect missingness-to-label match in `missingness_label_audit.csv`. A statistical association would require domain review; it is not silently treated as proof of leakage.\n\n",
        "The future/validation mutation unit tests fit no new scaler and therefore must leave both schema and scaler values unchanged.\n",
    ]
    (out / "feature_preprocessing.md").write_text("".join(lines), encoding="utf-8")


def _write_protocol(out: Path, dataset, class_distribution: pd.DataFrame) -> None:
    train_files = [part.record.file_name for part in dataset.train_parts]
    val_files = [part.record.file_name for part in dataset.validation_parts]
    train_types = set(dataset.train_graph.events.loc[dataset.train_graph.events["label"] == 1, "attack_type"])
    val_types = set()
    for source_file, ids in dataset.target_event_ids.items():
        target_events = dataset.validation_graphs[source_file].events.set_index("event_id").loc[ids]
        val_types.update(target_events.loc[target_events["label"] == 1, "attack_type"])
    known = sorted(val_types & train_types)
    unseen = sorted(val_types - train_types)
    lines = [
        "# Phase 3B evaluation protocol\n\n",
        "## Decision\n\n",
        "This is a **DIAGNOSTIC causal source-file holdout**, not a research-validated temporal benchmark. It uses the existing V1 chronological Network assignment without changing V1 or using Test.\n\n",
        f"V1 source-file assignment: `{len(train_files)}` Train files (`{', '.join(train_files)}`), `{len(val_files)}` Validation files (`{', '.join(val_files)}`), and `{len([record for record in dataset.source_records if record.fixed_split == 'test'])}` Test files excluded.\n\n",
        "Sampling rule fixed before metric inspection: read the first 2,000 source rows from every Train file; read the first 8,000 source rows from every Validation file; define the target as the final 30-second snapshot touched by the first 4,000 Validation rows of each file. This is source-order sampling, not label balancing or Test optimization.\n\n",
        "Train precedes Validation by source-file timestamp order. Snapshot ownership is half-open `[start,end)` and long gaps create new sequence IDs. The graph used for a Validation target contains Train history and Validation rows strictly before the target window close; future Validation rows are excluded.\n\n",
        f"Train attack types in the bounded fit graph: `{', '.join(sorted(train_types))}`. Validation target types: `{', '.join(sorted(val_types))}`. Seen/known: `{', '.join(known) or 'none'}`. Unseen in Train: `{', '.join(unseen) or 'none'}`.\n\n",
        "The target windows are not balanced. File-level V1 holdout is a split unit, not an asserted attack episode. Ground Truth windows are not interpreted as independent episodes. Network has no verified session/run identifier; repeated IP/connection-key overlap is reported as a limitation.\n\n",
        _markdown_table(class_distribution),
        "\n\nTest V1 is not read for model fitting, early stopping, metric calculation or model selection.\n",
    ]
    (out / "evaluation_protocol.md").write_text("".join(lines), encoding="utf-8")


def _write_causal_report(out: Path, dataset, class_distribution: pd.DataFrame) -> None:
    rows: list[dict[str, Any]] = []
    for source_file, graph in dataset.validation_graphs.items():
        target = graph.events[graph.events["event_id"].isin(dataset.target_event_ids[source_file])]
        rows.append({
            "validation_file": source_file,
            "causal_graph_events": len(graph.events),
            "target_events": len(target),
            "cutoff_epoch": dataset.cutoffs[source_file],
            "cutoff_utc": pd.to_datetime(dataset.cutoffs[source_file], unit="s", utc=True).isoformat(),
            "graph_max_epoch": float(graph.events["timestamp"].max()),
            "future_rows_in_graph": int((graph.events["timestamp"] >= dataset.cutoffs[source_file]).sum()),
            "direct_event_to_event_edges": int(((graph.edges["src_type"] == "event") & (graph.edges["dst_type"] == "event")).sum()) if not graph.edges.empty else 0,
            "event_ids_unique": bool(graph.events["event_id"].is_unique),
            "status": "PASS" if not target.empty and graph.events["timestamp"].max() < dataset.cutoffs[source_file] and graph.events["event_id"].is_unique else "FAIL",
        })
    causal = pd.DataFrame(rows)
    causal.to_csv(out / "causal_graph_window_audit.csv", index=False)
    lines = [
        "# Causal graph validation\n\n",
        "The primary prediction rule is: classify an Event after its 30-second snapshot closes. Each Validation graph is rebuilt with Train history plus only rows with timestamp `< cutoff`; the target is the final snapshot touched by the fixed first-4,000-row probe.\n\n",
        _markdown_table(causal),
        "\n\n",
        f"Exact duplicate intersection across selected Train/Validation prefixes: `{dataset.duplicate_audit['exact_fingerprint_intersection']}`; connection-key intersection proxy: `{dataset.duplicate_audit['connection_key_intersection_proxy']}` using `{dataset.duplicate_audit['connection_key_definition']}`. Exact duplicate status: **{dataset.duplicate_audit['status']}**.\n\n",
        "The connection-key result is not a session proof: the schema has no session/run identifier, so related-flow independence remains **UNVERIFIED**. No Event→Event edge is constructed; entity nodes can legitimately recur from Train history.\n\n",
        "Long timestamp gaps are represented by separate sequence IDs; they are not filled with artificial snapshots. Test V1 is not included.\n",
    ]
    (out / "causal_graph_validation.md").write_text("".join(lines), encoding="utf-8")


def _future_removal(dataset, scaler, train_model: torch.nn.Module | None, gnn_model: torch.nn.Module | None, expected_relations: list[tuple[str, str, str]]) -> pd.DataFrame:
    """Compare causal graph/logits after mutating all rows at/after T."""
    if not dataset.validation_parts:
        return pd.DataFrame([{"status": "UNVERIFIED", "reason": "no Validation part"}])
    part = dataset.validation_parts[0]
    source_file = part.record.file_name
    cutoff = dataset.cutoffs[source_file]
    mutated = part.frame.copy()
    times = validate_for_future(part, mutated)
    future_mask = times.to_numpy(dtype="float64") >= cutoff
    mutated.loc[future_mask, "label"] = 1 - pd.to_numeric(mutated.loc[future_mask, "label"], errors="coerce").astype(int)
    for column in [name for name in dataset.feature_columns if name in mutated.columns]:
        values = pd.to_numeric(mutated.loc[future_mask, column], errors="coerce")
        mutated.loc[future_mask, column] = values.fillna(0) * 997.0 + 13.0
    mutated_part = _make_part(part.record, mutated, "validation", dataset.feature_columns)
    mutated_graph = _combine_parts(dataset.train_parts + [mutated_part], dataset.feature_columns, cutoff=cutoff)
    transform_graph_features(mutated_graph, scaler)
    base_graph = dataset.validation_graphs[source_file]
    base_data = build_heterodata(base_graph)
    changed_data = build_heterodata(mutated_graph)
    _ensure_edge_schema([base_data, changed_data])
    node_equal = all(torch.equal(base_data[node].x, changed_data[node].x) for node in base_data.node_types)
    edge_equal = all(torch.equal(base_data[edge].edge_index, changed_data[edge].edge_index) for edge in base_data.edge_types)
    rows: list[dict[str, Any]] = []
    target_ids = dataset.target_event_ids[source_file]
    for name, model in [("EventMLP", train_model), ("StaticHeteroGraphSAGE", gnn_model)]:
        if model is None:
            rows.append({"model": name, "node_features_equal": node_equal, "edges_equal": edge_equal, "logits_equal": False, "future_labels_changed": True, "future_rows_mutated": int(future_mask.sum()), "status": "UNVERIFIED"})
            continue
        model.eval()
        with torch.no_grad():
            left = _model_logits(model, base_data)
            right = _model_logits(model, changed_data)
        mapping = base_graph.event_mapping.set_index("event_id")
        indices = mapping.loc[target_ids, "tensor_index"].to_numpy(dtype=np.int64)
        logits_equal = bool(torch.allclose(left[indices], right[indices], rtol=1e-5, atol=1e-6))
        rows.append({"model": name, "node_features_equal": node_equal, "edges_equal": edge_equal, "logits_equal": logits_equal, "future_labels_changed": True, "future_rows_mutated": int(future_mask.sum()), "status": "PASS" if node_equal and edge_equal and logits_equal else "FAIL"})
    return pd.DataFrame(rows)


def validate_for_future(part, frame):
    from ..pipeline.normalization import event_time_numeric
    return event_time_numeric(frame, part.record)


def _model_logits(model, data):
    if isinstance(model, EventMLP):
        return model(data["event"].x)
    x_dict = {node_type: data[node_type].x for node_type in data.node_types}
    edge_dict = {edge_type: data[edge_type].edge_index for edge_type in data.edge_types}
    return model(x_dict, edge_dict)


def _write_future_report(out: Path, checks: pd.DataFrame) -> None:
    checks.to_csv(out / "future_removal_checks.csv", index=False)
    lines = [
        "# Future-removal tests\n\n",
        "At a fixed Validation cutoff T, every future Validation row had its label and numeric feature values mutated, then the causal graph was rebuilt with the same frozen Train scaler. Only rows timestamped `< T` were eligible.\n\n",
        _markdown_table(checks),
        "\n\n",
        "PASS means past node features, edge tensors and fixed-model logits for the target Event set were unchanged within `rtol=1e-5`, `atol=1e-6`. This tests the implemented causal cutoff; it does not prove semantic session independence or temporal generalization.\n",
    ]
    (out / "future_removal_tests.md").write_text("".join(lines), encoding="utf-8")


def _write_final(out: Path, dataset, preprocessing_status: str, causal_checks: pd.DataFrame, future_checks: pd.DataFrame, results: pd.DataFrame, diagnostics: pd.DataFrame) -> None:
    technical_pass = preprocessing_status == "PASS" and not causal_checks.empty and bool((causal_checks["status"] == "PASS").all()) and not future_checks.empty and bool((future_checks["status"] == "PASS").all()) and not diagnostics.empty and bool(diagnostics["status"].fillna("").isin(["PASS"]).all())
    rows = results[results["model"].isin(["EventMLP", "StaticHeteroGraphSAGE", "MajorityBaseline"])].copy()
    deltas: list[str] = []
    if not rows.empty and "ALL_VALIDATION_BLOCKS" in set(rows["validation_block"]):
        all_rows = rows[rows["validation_block"] == "ALL_VALIDATION_BLOCKS"].set_index("model")
        if "StaticHeteroGraphSAGE" in all_rows.index and "EventMLP" in all_rows.index:
            deltas.append(f"descriptive Macro-F1 delta GraphSAGE−MLP = {float(all_rows.loc['StaticHeteroGraphSAGE', 'macro_f1'] - all_rows.loc['EventMLP', 'macro_f1']):.4f}")
            deltas.append(f"descriptive PR-AUC delta GraphSAGE−MLP = {float(all_rows.loc['StaticHeteroGraphSAGE', 'pr_auc'] - all_rows.loc['EventMLP', 'pr_auc']):.4f}")
    lines = [
        "# FINAL PHASE 3B REPORT\n\n",
        f"TECHNICAL STATUS: **{'PASS' if technical_pass else 'FAIL'}**\n\n",
        "RESEARCH STATUS: **DIAGNOSTIC ONLY**\n\n",
        "## Direct answers\n\n",
        f"1. **Feature standardization:** **{preprocessing_status}**. The 13 numeric fields were standardized using frozen Train-only median/mean/scale parameters; Validation/future mutations do not refit the schema or scaler. Missingness bits remain explicit.\n",
        f"2. **Causal Validation:** **DIAGNOSTIC ONLY**. Source-file chronology provides a usable engineering Train→Validation order, but session/run identity, related-flow independence and physical clock semantics are not fully verified.\n",
        f"3. **Future-removal:** **{'PASS' if not future_checks.empty and bool((future_checks['status'] == 'PASS').all()) else 'FAIL/UNVERIFIED'}** for the implemented cutoff and fixed-model logits. This is a construction test, not proof of research generalization.\n",
        "4. **Common protocol:** MLP and GraphSAGE use the same causal Validation target Event IDs, same frozen features, same epochs/seed/early-stopping rule and no Test.\n",
        f"5. **GraphSAGE improvement:** `{'; '.join(deltas) if deltas else 'not available'}`. Because the protocol is diagnostic and sampled, this must not be stated as a validated improvement.\n",
        "6. **Before RGCN+GRU:** establish verified temporal/clock semantics, session or scenario identity, broader non-label-selected sampling, independent Validation design, duplicate/related-flow controls, and a valid held-out Test protocol.\n\n",
        "## Limits\n\n",
        f"The pilot uses fixed prefixes from `{len(dataset.train_parts)}` Train files and `{len(dataset.validation_parts)}` Validation files; it does not read Test. Validation target windows include known and/or unseen attack types according to the observed source order, and are not class-balanced. Ground Truth windows are not treated as independent episodes.\n\n",
        "Metrics in `mlp_vs_static_gnn_results.csv` are descriptive Validation diagnostics. A lower loss or higher metric does not upgrade Research Status.\n",
    ]
    (out / "FINAL_PHASE3B_REPORT.md").write_text("".join(lines), encoding="utf-8")


def run(output_dir: Path, artifact_dir: Path, *, train_prefix_rows: int = 2_000, validation_prefix_rows: int = 8_000, validation_target_rows: int = 4_000) -> int:
    for path in (output_dir, artifact_dir):
        if path.exists() and any(path.iterdir()):
            raise RuntimeError(f"Refusing to overwrite non-empty Phase 3B output: {path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_protocol_dataset(PROJECT_ROOT, train_prefix_rows, validation_prefix_rows, validation_target_rows)
    scaler = fit_train_scaler(dataset.train_graph.event_x, dataset.feature_columns)
    preprocessing = preprocessing_diagnostics(dataset.train_graph.event_x, dataset.train_graph.events["label"].to_numpy(dtype=np.int64), dataset.feature_columns)
    transform_graph_features(dataset.train_graph, scaler)
    for graph in dataset.validation_graphs.values():
        transform_graph_features(graph, scaler)
    fingerprint = feature_schema_fingerprint(dataset.feature_columns, dataset.train_graph.event_mapping)
    class_distribution = _class_rows(dataset)
    class_distribution.to_csv(output_dir / "class_distribution.csv", index=False)
    _write_preprocessing(output_dir, dataset, scaler, preprocessing, fingerprint, artifact_dir)
    _write_protocol(output_dir, dataset, class_distribution)
    train_data = build_heterodata(dataset.train_graph)
    validation_data = {name: build_heterodata(graph) for name, graph in dataset.validation_graphs.items()}
    _ensure_edge_schema([train_data, *validation_data.values()])
    validation_bundles = {name: _graph_bundle(dataset.validation_graphs[name], data) for name, data in validation_data.items()}
    runs: list[EvaluationRun] = []
    for model_name in ["EventMLP", "StaticHeteroGraphSAGE"]:
        runs.append(train_and_validate(model_name, train_data, validation_bundles, dataset.target_event_ids, seed=42, epochs=12, patience=3, learning_rate=0.01))
    results = pd.DataFrame([row for run_result in runs for row in run_result.results] + majority_baseline(train_data, validation_bundles, dataset.target_event_ids))
    diagnostics = pd.DataFrame([row for run_result in runs for row in run_result.diagnostics])
    results.to_csv(output_dir / "mlp_vs_static_gnn_results.csv", index=False)
    diagnostics.to_csv(output_dir / "training_diagnostics.csv", index=False)
    _write_causal_report(output_dir, dataset, class_distribution)
    future_checks = _future_removal(dataset, scaler, next((run_result.model for run_result in runs if run_result.model_name == "EventMLP"), None), next((run_result.model for run_result in runs if run_result.model_name == "StaticHeteroGraphSAGE"), None), train_data.edge_types)
    _write_future_report(output_dir, future_checks)
    torch.save({run_result.model_name: run_result.model.state_dict() for run_result in runs}, artifact_dir / "best_checkpoints.pt")
    dataset.train_graph.event_mapping.to_csv(artifact_dir / "train_event_mapping.csv", index=False)
    for name, graph in dataset.validation_graphs.items():
        graph.event_mapping.to_csv(artifact_dir / f"{Path(name).stem}_event_mapping.csv", index=False)
    _write_final(output_dir, dataset, preprocessing["status"], pd.read_csv(output_dir / "causal_graph_window_audit.csv"), future_checks, results, diagnostics)
    (artifact_dir / "phase3b_manifest.json").write_text(json.dumps({"train_prefix_rows": train_prefix_rows, "validation_prefix_rows": validation_prefix_rows, "validation_target_rows": validation_target_rows, "feature_columns": dataset.feature_columns, "fingerprint": fingerprint, "duplicate_audit": dataset.duplicate_audit, "test_used": False}, indent=2, default=str), encoding="utf-8")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "research_pipeline" / "reports" / "experimental_phase3b")
    parser.add_argument("--artifact-dir", type=Path, default=PROJECT_ROOT / "research_pipeline" / "output_phase3b_network")
    parser.add_argument("--train-prefix-rows", type=int, default=2_000)
    parser.add_argument("--validation-prefix-rows", type=int, default=8_000)
    parser.add_argument("--validation-target-rows", type=int, default=4_000)
    args = parser.parse_args()
    return run(args.output_dir.resolve(), args.artifact_dir.resolve(), train_prefix_rows=args.train_prefix_rows, validation_prefix_rows=args.validation_prefix_rows, validation_target_rows=args.validation_target_rows)


if __name__ == "__main__":
    raise SystemExit(main())

