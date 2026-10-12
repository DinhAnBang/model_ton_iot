"""Preprocessor for the Phase-1 research pipeline."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..common.config import MAX_CATEGORICAL_LEVELS, MEDIAN_SAMPLE_SIZE, QUARANTINE_CODE, SPLIT_CODE
from ..common.errors import PipelineError
from ..common.utils import log
from .duplicate_handler import DuplicateRouter, duplicate_fingerprints
from .feature_selector import final_features, infer_train_feature_specs
from .models import FeatureSpec, GroupPlan, RunState
from .normalization import validate_chunk
from .split_data import base_split_codes, remediated_split_codes


def update_median_sample(spec: FeatureSpec, values: np.ndarray) -> None:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return
    budget = MEDIAN_SAMPLE_SIZE
    if len(values) > 10_000:
        step = max(1, len(values) // 10_000)
        values = values[::step][:10_000]
    if not spec.median_sample:
        spec.median_sample = values.astype("float64", copy=False).tolist()
    else:
        merged = np.asarray(spec.median_sample + values.tolist(), dtype="float64")
        if len(merged) > budget:
            indexes = np.linspace(0, len(merged) - 1, budget, dtype=np.int64)
            merged = merged[indexes]
        spec.median_sample = merged.tolist()


def fit_group(
    plan: GroupPlan,
    type_mapping: dict[str, str],
    chunk_size: int,
    state: RunState,
    *,
    quarantine_invalid: bool = False,
) -> None:
    log(f"Fit train-only preprocessing: {plan.group_id}")
    state.quarantine_policy = (
        "invalid_timestamp_quarantine" if quarantine_invalid else "v1_missing_timestamps_train"
    )
    for rec in plan.records:
        router = DuplicateRouter()
        rows_read = 0
        for frame in pd.read_csv(rec.file_path, chunksize=chunk_size, low_memory=False):
            if list(frame.columns) != rec.columns:
                raise PipelineError(f"{rec.file_name}: schema changed while reading chunks")
            rows_read += len(frame)
            _, _, _, times = validate_chunk(frame, rec, type_mapping, state, validation_only=True)
            base = remediated_split_codes(times, rec) if quarantine_invalid else base_split_codes(times, rec)
            if rec.split_mode == "source_file_holdout":
                # The complete source file is one split, so duplicate rows
                # cannot cross folds and no per-row router is needed.
                final_codes = base
            else:
                fp = duplicate_fingerprints(frame, rec.columns)
                if quarantine_invalid:
                    final_codes = router.route_masked(
                        fp, base, base != QUARANTINE_CODE, QUARANTINE_CODE
                    )
                else:
                    final_codes = router.route(fp, base)
            mask = final_codes == SPLIT_CODE["train"]
            if not mask.any():
                continue
            train_frame = frame.loc[mask]
            for name in plan.feature_order:
                spec = plan.feature_specs[name]
                if spec.kind == "numeric":
                    numeric = pd.to_numeric(train_frame[name], errors="coerce")
                    values = numeric.to_numpy(dtype="float64", na_value=np.nan)
                    spec.train_missing += int(np.isnan(values).sum())
                    spec.train_nonmissing += int(np.isfinite(values).sum())
                    update_median_sample(spec, values)
                elif spec.kind == "categorical":
                    series = train_frame[name].astype("string")
                    nonmissing = series.dropna()
                    spec.train_missing += int(series.isna().sum())
                    spec.train_nonmissing += int(nonmissing.shape[0])
                    # Keep a bounded set.  If it grows beyond the policy,
                    # the feature is excluded from the baseline.
                    if not spec.excluded_after_fit:
                        current = set(spec.categories)
                        current.update(nonmissing.astype(str).unique().tolist())
                        if len(current) > MAX_CATEGORICAL_LEVELS:
                            spec.excluded_after_fit = True
                            spec.exclusion_reason = "train cardinality exceeds 64; excluded instead of one-hot/high-cardinality encoding"
                        else:
                            spec.categories = sorted(current)
        if rows_read != rec.rows_expected:
            raise PipelineError(f"{rec.file_name}: read {rows_read} rows, inventory expects {rec.rows_expected}")
        state.duplicate_seen[rec.file_name] += router.first_seen
        state.duplicate_routed[rec.file_name] += router.routed
        state.duplicate_cross_split[rec.file_name] += router.cross_split

    for name in plan.feature_order:
        spec = plan.feature_specs[name]
        if spec.kind == "numeric":
            if spec.train_nonmissing == 0 or not spec.median_sample:
                spec.excluded_after_fit = True
                spec.exclusion_reason = "no usable non-missing train statistic; never zero-filled"
            else:
                spec.median = float(np.median(np.asarray(spec.median_sample, dtype="float64")))
        elif spec.kind == "categorical":
            if spec.excluded_after_fit or not spec.categories:
                spec.excluded_after_fit = True
                if not spec.exclusion_reason:
                    spec.exclusion_reason = "no usable train category vocabulary"


def transform_baseline(frame: pd.DataFrame, plan: GroupPlan) -> pd.DataFrame:
    output: dict[str, Any] = {}
    for name in final_features(plan):
        spec = plan.feature_specs[name]
        if spec.kind == "numeric":
            values = pd.to_numeric(frame[name], errors="coerce")
            output[name] = values.fillna(spec.median).astype("float64")
        elif spec.kind == "categorical":
            lookup = {value: idx for idx, value in enumerate(spec.categories)}
            values = frame[name].astype("string").map(lookup).fillna(-1)
            output[name] = values.astype("int16")
    output["label"] = pd.to_numeric(frame["label"], errors="raise").astype("int8")
    return pd.DataFrame(output, index=frame.index)


def fit_preprocessors(
    plans: list[GroupPlan],
    type_mapping: dict[str, str],
    chunk_size: int,
    state: RunState,
    *,
    risk: dict[tuple[str, str, str], dict[str, Any]] | None = None,
    quarantine_invalid: bool = False,
) -> None:
    """Fit every profile using its assigned train rows only."""
    for plan in plans:
        if risk is not None and any(spec.kind == "pending" for spec in plan.feature_specs.values()):
            infer_train_feature_specs(
                plan,
                risk,
                type_mapping,
                chunk_size,
                quarantine_invalid=quarantine_invalid,
            )
        fit_group(
            plan,
            type_mapping,
            chunk_size,
            state,
            quarantine_invalid=quarantine_invalid,
        )
