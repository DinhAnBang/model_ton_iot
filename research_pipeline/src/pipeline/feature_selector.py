"""Feature selector for the Phase-1 research pipeline."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import pandas as pd

from ..common.config import MAX_CATEGORICAL_LEVELS
from ..common.errors import PipelineError
from ..common.paths import REPORT_DIR
from ..common.utils import slugify
from .models import FeatureSpec, GroupPlan, SourceRecord


def read_feature_risk() -> dict[tuple[str, str, str], dict[str, Any]]:
    path = REPORT_DIR / "feature_risk.csv"
    if not path.exists():
        raise PipelineError(f"Missing feature-risk report: {path}")
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"platform", "file", "feature", "classification", "reason"}
    missing = required - set(df.columns)
    if missing:
        raise PipelineError(f"feature_risk.csv is missing columns: {sorted(missing)}")
    result: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in df.to_dict("records"):
        result[(row["platform"], row["file"], row["feature"])] = row
    return result


def infer_feature_specs(group: GroupPlan, risk: dict[tuple[str, str, str], dict[str, Any]]) -> None:
    if not group.records:
        return
    first = group.records[0]
    candidates: list[str] = []
    for col in first.columns:
        if col in {"label", "type", "ts", "date", "time"}:
            continue
        if all(col in r.columns for r in group.records):
            classifications = [risk.get((r.platform, r.file_name, col), {}).get("classification", "UNKNOWN") for r in group.records]
            if all(c in {"VALID_OBSERVABLE_FEATURE", "REVIEW_REQUIRED"} for c in classifications):
                candidates.append(col)

    group.feature_order = candidates
    # Read each source sample once.  Windows has hundreds of columns and
    # reading the same 5,000 rows once per feature would turn planning into an
    # avoidable multi-minute I/O loop.
    sample_frames = {
        rec.file_name: pd.read_csv(rec.file_path, nrows=5_000, low_memory=False)
        for rec in group.records
    }
    for col in candidates:
        risk_row = risk.get((first.platform, first.file_name, col), {})
        samples: list[pd.Series] = [
            sample_frames[rec.file_name][col]
            for rec in group.records
            if col in sample_frames[rec.file_name].columns
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
            all_review = risk_row.get("classification") == "REVIEW_REQUIRED"
            if all_review:
                kind = "excluded"
                reason = "review-required categorical/state feature; kept semantic only until its encoding semantics are explicitly approved"
            elif unique <= MAX_CATEGORICAL_LEVELS:
                kind = "categorical"
                reason = "approved low-cardinality observable category; ordinal encoding fitted on train only"
            else:
                kind = "excluded"
                reason = "approved feature is high-cardinality/text-like; no one-hot or identifier encoding"
        group.feature_specs[col] = FeatureSpec(
            name=col,
            kind=kind,
            risk_classification=risk_row.get("classification", "UNKNOWN"),
            reason=reason,
        )
        if kind == "excluded":
            group.feature_specs[col].excluded_after_fit = True
            group.feature_specs[col].exclusion_reason = reason


def make_group_plans(records: list[SourceRecord], risk: dict[tuple[str, str, str], dict[str, Any]]) -> list[GroupPlan]:
    grouped: defaultdict[str, list[SourceRecord]] = defaultdict(list)
    for rec in records:
        grouped[rec.group_id].append(rec)
    plans: list[GroupPlan] = []
    for group_id in sorted(grouped):
        group_records = sorted(grouped[group_id], key=lambda r: r.file_name)
        platform = group_records[0].platform
        source_columns = list(group_records[0].columns)
        plan = GroupPlan(
            group_id=group_id,
            platform=platform,
            records=group_records,
            feature_specs={},
            feature_order=[],
            source_columns=source_columns,
            preprocessor_name=slugify(group_id),
        )
        infer_feature_specs(plan, risk)
        plans.append(plan)
    return plans


def final_features(plan: GroupPlan) -> list[str]:
    return [
        name for name in plan.feature_order
        if not plan.feature_specs[name].excluded_after_fit
    ]
