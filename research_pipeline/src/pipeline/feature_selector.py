"""Train-only feature planning for the Phase-1 research pipeline."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import pandas as pd

from ..common.config import MAX_CATEGORICAL_LEVELS, SPLIT_CODE
from ..common.errors import PipelineError
from ..common.paths import REPORT_DIR
from ..common.utils import slugify
from .duplicate_handler import DuplicateRouter, duplicate_fingerprints
from .models import FeatureSpec, GroupPlan, RunState
from .normalization import validate_chunk
from .split_data import base_split_codes, remediated_split_codes


def read_feature_risk() -> dict[tuple[str, str, str], dict[str, Any]]:
    path = REPORT_DIR / "feature_risk.csv"
    if not path.exists():
        raise PipelineError(f"Missing feature-risk report: {path}")
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"platform", "file", "feature", "classification", "reason"}
    missing = required - set(df.columns)
    if missing:
        raise PipelineError(f"feature_risk.csv is missing columns: {sorted(missing)}")
    return {(row["platform"], row["file"], row["feature"]): row for row in df.to_dict("records")}


def _candidate_columns(group: GroupPlan, risk: dict[tuple[str, str, str], dict[str, Any]]) -> list[str]:
    if not group.records:
        return []
    first = group.records[0]
    candidates: list[str] = []
    for col in first.columns:
        if col in {"label", "type", "ts", "date", "time"}:
            continue
        if all(col in record.columns for record in group.records):
            classifications = [
                risk.get((record.platform, record.file_name, col), {}).get("classification", "UNKNOWN")
                for record in group.records
            ]
            if all(value in {"VALID_OBSERVABLE_FEATURE", "REVIEW_REQUIRED"} for value in classifications):
                candidates.append(col)
    return candidates


def _make_spec(
    name: str,
    kind: str,
    risk_row: dict[str, Any],
    reason: str,
) -> FeatureSpec:
    spec = FeatureSpec(
        name=name,
        kind=kind,
        risk_classification=risk_row.get("classification", "UNKNOWN"),
        reason=reason,
    )
    if kind == "excluded":
        spec.excluded_after_fit = True
        spec.exclusion_reason = reason
    return spec


def _set_legacy_sample_specs(
    group: GroupPlan,
    candidates: list[str],
    risk: dict[tuple[str, str, str], dict[str, Any]],
) -> None:
    """Keep the original V1 sampler available as an explicit control."""
    sample_frames = {
        record.file_name: pd.read_csv(record.file_path, nrows=5_000, low_memory=False)
        for record in group.records
    }
    first = group.records[0]
    for col in candidates:
        risk_row = risk.get((first.platform, first.file_name, col), {})
        samples = [
            sample_frames[record.file_name][col]
            for record in group.records
            if col in sample_frames[record.file_name].columns
        ]
        if not samples:
            continue
        sample = pd.concat(samples, ignore_index=True)
        numeric = pd.to_numeric(sample, errors="coerce")
        nonmissing = sample.notna().sum()
        numeric_ratio = float(numeric.notna().sum() / nonmissing) if nonmissing else 0.0
        if pd.api.types.is_numeric_dtype(sample) or numeric_ratio >= 0.95:
            kind = "numeric"
            reason = "approved observable numeric feature; numeric strings are coerced without treating '-' as zero"
        else:
            unique = sample.dropna().astype("string").nunique()
            if risk_row.get("classification") == "REVIEW_REQUIRED":
                kind = "excluded"
                reason = "review-required categorical/state feature; kept semantic only until its encoding semantics are explicitly approved"
            elif unique <= MAX_CATEGORICAL_LEVELS:
                kind = "categorical"
                reason = "approved low-cardinality observable category; ordinal encoding fitted on train only"
            else:
                kind = "excluded"
                reason = "approved feature is high-cardinality/text-like; no one-hot or identifier encoding"
        group.feature_specs[col] = _make_spec(col, kind, risk_row, reason)


def infer_feature_specs(
    group: GroupPlan,
    risk: dict[tuple[str, str, str], dict[str, Any]],
    *,
    legacy_sampling: bool = False,
) -> None:
    """Build candidates without reading validation/test samples.

    ``legacy_sampling=True`` is only for reproducing V1.  The corrected path
    creates pending specs here and resolves them in
    :func:`infer_train_feature_specs` using Train rows only.
    """
    candidates = _candidate_columns(group, risk)
    group.feature_order = candidates
    group.feature_specs = {}
    if legacy_sampling:
        _set_legacy_sample_specs(group, candidates, risk)
        return
    first = group.records[0] if group.records else None
    for name in candidates:
        risk_row = risk.get((first.platform, first.file_name, name), {}) if first else {}
        group.feature_specs[name] = _make_spec(
            name,
            "pending",
            risk_row,
            "kind/cardinality is learned from assigned Train rows only",
        )


def _resolve_train_spec(
    group: GroupPlan,
    name: str,
    risk: dict[tuple[str, str, str], dict[str, Any]],
    numeric_count: int,
    nonmissing: int,
    unique_values: set[str],
    cardinality_overflow: bool,
    numeric_dtype_seen: bool,
) -> None:
    first = group.records[0]
    risk_row = risk.get((first.platform, first.file_name, name), {})
    if nonmissing == 0:
        group.feature_specs[name] = _make_spec(
            name,
            "excluded",
            risk_row,
            "no non-missing Train value; retained in semantic output but not fabricated into baseline X",
        )
        return
    numeric_ratio = numeric_count / nonmissing
    if numeric_dtype_seen or numeric_ratio >= 0.95:
        group.feature_specs[name] = _make_spec(
            name,
            "numeric",
            risk_row,
            "approved observable numeric feature; type inference and later statistics use Train rows only",
        )
    elif risk_row.get("classification") == "REVIEW_REQUIRED":
        group.feature_specs[name] = _make_spec(
            name,
            "excluded",
            risk_row,
            "review-required categorical/state feature; kept semantic-only until encoding semantics are approved",
        )
    elif cardinality_overflow or len(unique_values) > MAX_CATEGORICAL_LEVELS:
        group.feature_specs[name] = _make_spec(
            name,
            "excluded",
            risk_row,
            "Train cardinality exceeds 64; no one-hot or identifier encoding",
        )
    else:
        group.feature_specs[name] = _make_spec(
            name,
            "categorical",
            risk_row,
            "approved low-cardinality category; vocabulary is learned from Train rows only",
        )


def infer_train_feature_specs(
    group: GroupPlan,
    risk: dict[tuple[str, str, str], dict[str, Any]],
    type_mapping: dict[str, str],
    chunk_size: int,
    *,
    quarantine_invalid: bool = True,
) -> None:
    """Infer feature kinds/cardinality by streaming only final Train rows."""
    if not group.records:
        return
    numeric_counts = {name: 0 for name in group.feature_order}
    nonmissing_counts = {name: 0 for name in group.feature_order}
    unique_values = {name: set() for name in group.feature_order}
    cardinality_overflow = {name: False for name in group.feature_order}
    numeric_dtype_seen = {name: False for name in group.feature_order}
    scan_state = RunState()

    for record in group.records:
        router = DuplicateRouter()
        for frame in pd.read_csv(record.file_path, chunksize=chunk_size, low_memory=False):
            if list(frame.columns) != record.columns:
                raise PipelineError(f"{record.file_name}: schema changed while reading chunks")
            _, _, _, times = validate_chunk(frame, record, type_mapping, scan_state, validation_only=True)
            base = remediated_split_codes(times, record) if quarantine_invalid else base_split_codes(times, record)
            if record.split_mode == "source_file_holdout":
                final_codes = base
            else:
                fingerprints = duplicate_fingerprints(frame, record.columns)
                eligible = base != -1
                final_codes = (
                    router.route_masked(fingerprints, base, eligible)
                    if quarantine_invalid
                    else router.route(fingerprints, base)
                )
            train_frame = frame.loc[final_codes == SPLIT_CODE["train"]]
            for name in group.feature_order:
                series = train_frame[name]
                nonmissing_counts[name] += int(series.notna().sum())
                numeric = pd.to_numeric(series, errors="coerce")
                numeric_counts[name] += int(numeric.notna().sum())
                numeric_dtype_seen[name] |= bool(pd.api.types.is_numeric_dtype(series))
                if not cardinality_overflow[name]:
                    unique_values[name].update(series.dropna().astype("string").astype(str).unique().tolist())
                    if len(unique_values[name]) > MAX_CATEGORICAL_LEVELS:
                        cardinality_overflow[name] = True
                        unique_values[name].clear()
    for name in group.feature_order:
        _resolve_train_spec(
            group,
            name,
            risk,
            numeric_counts[name],
            nonmissing_counts[name],
            unique_values[name],
            cardinality_overflow[name],
            numeric_dtype_seen[name],
        )


def make_group_plans(
    records: list[SourceRecord],
    risk: dict[tuple[str, str, str], dict[str, Any]],
    *,
    legacy_sampling: bool = False,
) -> list[GroupPlan]:
    grouped: defaultdict[str, list[SourceRecord]] = defaultdict(list)
    for record in records:
        grouped[record.group_id].append(record)
    plans: list[GroupPlan] = []
    for group_id in sorted(grouped):
        group_records = sorted(grouped[group_id], key=lambda record: record.file_name)
        plan = GroupPlan(
            group_id=group_id,
            platform=group_records[0].platform,
            records=group_records,
            feature_specs={},
            feature_order=[],
            source_columns=list(group_records[0].columns),
            preprocessor_name=slugify(group_id),
        )
        infer_feature_specs(plan, risk, legacy_sampling=legacy_sampling)
        plans.append(plan)
    return plans


def final_features(plan: GroupPlan) -> list[str]:
    return [
        name
        for name in plan.feature_order
        if name in plan.feature_specs
        and plan.feature_specs[name].kind in {"numeric", "categorical"}
        and not plan.feature_specs[name].excluded_after_fit
    ]
