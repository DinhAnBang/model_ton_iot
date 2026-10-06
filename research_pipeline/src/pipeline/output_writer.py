"""Output writer for the Phase-1 research pipeline."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..common.config import CODE_SPLIT
from ..common.errors import PipelineError
from ..common.paths import PROJECT_ROOT
from ..common.utils import log, safe_json_number, slugify
from .duplicate_handler import DuplicateRouter, duplicate_fingerprints
from .feature_selector import final_features
from .models import GroupPlan, RunState, SourceRecord
from .normalization import event_time_text, format_event_range, raw_timestamp_text, validate_chunk
from .preprocessor import transform_baseline
from .split_data import base_split_codes, scenario_id_series


def write_csv_gzip(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = not path.exists()
    frame.to_csv(
        path,
        mode="a" if not header else "w",
        header=header,
        index=False,
        compression="gzip",
        lineterminator="\n",
    )


def output_paths(output_root: Path, rec: SourceRecord, split: str) -> tuple[Path, Path]:
    platform_slug = slugify(rec.platform)
    file_slug = slugify(Path(rec.file_name).stem)
    baseline = output_root / "baseline" / platform_slug / split / f"{file_slug}.csv.gz"
    semantic = output_root / "semantic" / platform_slug / split / f"{file_slug}.csv.gz"
    return baseline, semantic


def build_semantic_frame(
    frame: pd.DataFrame,
    rec: SourceRecord,
    split: str,
    event_text: pd.Series,
    raw_timestamp: pd.Series,
    raw_type: pd.Series,
    canonical: pd.Series,
    scenario_ids: pd.Series,
    row_numbers: np.ndarray,
) -> pd.DataFrame:
    semantic = frame.copy()
    semantic["event_time"] = event_text.astype("string")
    semantic["raw_timestamp"] = raw_timestamp.astype("string")
    semantic["platform"] = rec.platform
    semantic["source_file"] = rec.relative_path
    semantic["scenario_id"] = scenario_ids.astype("string")
    semantic["split"] = split
    semantic["row_identifier"] = [f"{rec.relative_path}::row_{int(n):09d}" for n in row_numbers]
    semantic["raw_type"] = raw_type.astype("string")
    semantic["canonical_type"] = canonical.astype("string")
    semantic["label"] = pd.to_numeric(frame["label"], errors="raise").astype("int8")
    return semantic


def update_counts(
    state: RunState,
    rec: SourceRecord,
    split: str,
    frame: pd.DataFrame,
    canonical: pd.Series,
    times: pd.Series,
    scenario_ids: pd.Series,
) -> None:
    key = (rec.platform, rec.group_id, rec.file_name, split)
    count = len(frame)
    labels = pd.to_numeric(frame["label"], errors="raise").astype("int64")
    state.output_rows[key] += count
    state.output_normal[key] += int((labels == 0).sum())
    state.output_attack[key] += int((labels == 1).sum())
    attack_values = canonical.loc[labels.to_numpy() == 1]
    state.output_attack_types[key].update(attack_values.astype(str).tolist())
    finite = times[np.isfinite(times.to_numpy(dtype="float64"))]
    if len(finite):
        lo = float(finite.min())
        hi = float(finite.max())
        state.time_min[key] = min(lo, state.time_min.get(key, lo))
        state.time_max[key] = max(hi, state.time_max.get(key, hi))
    state.source_split_sets[(rec.platform, rec.file_name)].add(split)
    for scenario_id in scenario_ids.dropna().astype(str).unique():
        state.scenario_split_sets[scenario_id].add(split)


def transform_group(
    plan: GroupPlan,
    type_mapping: dict[str, str],
    chunk_size: int,
    output_root: Path,
    state: RunState,
    manifest_rows: list[dict[str, Any]],
) -> None:
    log(f"Write views: {plan.group_id} | baseline features={len(final_features(plan))}")
    for rec in plan.records:
        router = DuplicateRouter()
        rows_read = 0
        for block_index, frame in enumerate(pd.read_csv(rec.file_path, chunksize=chunk_size, low_memory=False)):
            rows_read += len(frame)
            labels, raw_type, canonical, times = validate_chunk(frame, rec, type_mapping, state)
            base = base_split_codes(times, rec)
            if rec.split_mode == "source_file_holdout":
                final_codes = base
            else:
                fp = duplicate_fingerprints(frame, rec.columns)
                final_codes = router.route(fp, base)
            event_text = event_time_text(frame, rec)
            raw_timestamp = raw_timestamp_text(frame, rec)
            scenario_ids = scenario_id_series(times, rec)
            row_numbers = np.arange(rows_read - len(frame), rows_read, dtype=np.int64)
            for split_code, split in CODE_SPLIT.items():
                mask = final_codes == split_code
                if not mask.any():
                    continue
                selected = frame.loc[mask].copy()
                selected_labels = labels.loc[mask]
                selected_canonical = canonical.loc[mask]
                selected_times = times.loc[mask]
                selected_scenario_ids = scenario_ids.loc[mask]
                selected_event_text = event_text.loc[mask]
                selected_raw_timestamp = raw_timestamp.loc[mask]
                selected_rows = row_numbers[mask]
                baseline = transform_baseline(selected, plan)
                baseline_path, semantic_path = output_paths(output_root, rec, split)
                write_csv_gzip(baseline, baseline_path)
                semantic = build_semantic_frame(
                    selected,
                    rec,
                    split,
                    selected_event_text,
                    selected_raw_timestamp,
                    selected["type"].astype("string"),
                    selected_canonical,
                    selected_scenario_ids,
                    selected_rows,
                )
                write_csv_gzip(semantic, semantic_path)
                state.baseline_paths.add(str(baseline_path.relative_to(PROJECT_ROOT)).replace("\\", "/"))
                state.semantic_paths.add(str(semantic_path.relative_to(PROJECT_ROOT)).replace("\\", "/"))
                update_counts(state, rec, split, selected, selected_canonical, selected_times, selected_scenario_ids)
                finite = selected_times[np.isfinite(selected_times.to_numpy(dtype="float64"))]
                event_min = format_event_range(float(finite.min()), rec.platform) if len(finite) else ""
                event_max = format_event_range(float(finite.max()), rec.platform) if len(finite) else ""
                scenario_values = sorted(selected_scenario_ids.dropna().astype(str).unique().tolist())
                if len(scenario_values) == 1:
                    manifest_scenario = scenario_values[0]
                elif scenario_values:
                    manifest_scenario = f"TIMEBLOCK_RANGE::{scenario_values[0]}..{scenario_values[-1]}"
                else:
                    manifest_scenario = f"TIMEBLOCK::{rec.platform}::MISSING_TIME"
                manifest_rows.append({
                    "platform": rec.platform,
                    "source_file": rec.relative_path,
                    "scenario_id": manifest_scenario,
                    "event_time": f"{event_min}/{event_max}",
                    "event_time_min": event_min,
                    "event_time_max": event_max,
                    "row_identifier": f"{rec.relative_path}::block_{block_index:06d}::{split}",
                    "split": split,
                    "split_reason": rec.split_reason,
                    "source_group": rec.group_id,
                    "rows": int(len(selected)),
                    "normal_count": int((selected_labels == 0).sum()),
                    "attack_count": int((selected_labels == 1).sum()),
                    "duplicate_routed_count": int(router.routed),
                })
        if rows_read != rec.rows_expected:
            raise PipelineError(f"{rec.file_name}: transform read {rows_read}, expected {rec.rows_expected}")
        if router.routed:
            log(f"  {rec.file_name}: routed {router.routed:,} duplicate rows to their first split")
        state.duplicate_routed[rec.file_name] = router.routed
        state.duplicate_cross_split[rec.file_name] = router.cross_split


def clear_generated_output(output_root: Path, overwrite: bool) -> None:
    generated = [
        output_root / "baseline",
        output_root / "semantic",
        output_root / "preprocessors",
        output_root / "manifests",
    ]
    existing = [p for p in generated if p.exists()]
    if existing and not overwrite:
        names = ", ".join(str(p) for p in existing)
        raise PipelineError(f"Generated output already exists; use --overwrite to replace only: {names}")
    for path in existing:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    for path in generated:
        path.mkdir(parents=True, exist_ok=True)


def write_preprocessor(plan: GroupPlan, output_root: Path, chunk_size: int) -> Path:
    features = final_features(plan)
    payload = {
        "format": "phase1_train_only_preprocessor_v1",
        "platform": plan.platform,
        "source_group": plan.group_id,
        "fit_split": "train",
        "scaler": "none; tree-based RF/XGBoost/LightGBM baselines do not require scaling",
        "imputer": "numeric train-only median; categorical missing/unseen=-1",
        "categorical_encoder": "train-only ordinal codes; no one-hot; unknown=-1",
        "baseline_feature_columns": features,
        "features": {
            name: {
                "kind": plan.feature_specs[name].kind,
                "median": safe_json_number(plan.feature_specs[name].median),
                "categories": plan.feature_specs[name].categories,
                "train_nonmissing": plan.feature_specs[name].train_nonmissing,
                "train_missing": plan.feature_specs[name].train_missing,
                "handling": plan.feature_specs[name].reason,
            }
            for name in features
        },
        "excluded_candidates": {
            name: {
                "kind": plan.feature_specs[name].kind,
                "reason": plan.feature_specs[name].exclusion_reason or plan.feature_specs[name].reason,
            }
            for name in plan.feature_order
            if name not in features
        },
        "source_files": [r.relative_path for r in plan.records],
        "reproducibility": {
            "chunk_size": chunk_size,
            "duplicate_policy": "within-source exact fingerprint; keep all rows; route later duplicate to first encountered split",
            "fingerprint": "pandas hash_pandas_object over all original source columns",
        },
    }
    path = output_root / "preprocessors" / f"{plan.preprocessor_name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    return path


def write_manifest(manifest_rows: list[dict[str, Any]], output_root: Path) -> Path:
    path = output_root / "manifests" / "split_manifest.csv"
    columns = [
        "platform", "source_file", "scenario_id", "event_time", "event_time_min",
        "event_time_max", "row_identifier", "split", "split_reason", "source_group",
        "rows", "normal_count", "attack_count", "duplicate_routed_count",
    ]
    df = pd.DataFrame(manifest_rows, columns=columns)
    df.to_csv(path, index=False, encoding="utf-8")
    return path


def validate_final_outputs(state: RunState, records: list[SourceRecord]) -> None:
    """Require a complete, labeled baseline and semantic export."""
    if sum(state.output_rows.values()) != sum(record.rows_expected for record in records):
        raise PipelineError("Output row count does not equal source row count")
    if not state.baseline_paths or not state.semantic_paths:
        raise PipelineError("No baseline or semantic output files were written")
    if any(state.invalid_labels.values()):
        raise PipelineError("Validation counters contain invalid labels")


def export_dataset(
    plans: list[GroupPlan],
    type_mapping: dict[str, str],
    chunk_size: int,
    output_root: Path,
    state: RunState,
) -> Path:
    """Write both data views and their partition-level split manifest."""
    manifest_rows: list[dict[str, Any]] = []
    for plan in plans:
        transform_group(plan, type_mapping, chunk_size, output_root, state, manifest_rows)
    return write_manifest(manifest_rows, output_root)
