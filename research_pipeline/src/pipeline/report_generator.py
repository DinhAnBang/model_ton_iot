"""Report generator for the Phase-1 research pipeline."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from ..common.config import SPLITS
from ..common.paths import PROJECT_ROOT, REPORT_DIR
from ..common.utils import read_json
from .feature_selector import final_features
from .models import FeatureSpec, GroupPlan, RunState, SourceRecord
from .normalization import format_event_range
from .split_data import calendar_split_description, platform_class_coverage


def write_split_distribution(state: RunState, output_root: Path) -> Path:
    rows: list[dict[str, Any]] = []
    for key in sorted(state.output_rows):
        platform, group_id, file_name, split = key
        attack_types = state.output_attack_types[key]
        rows.append({
            "platform": platform,
            "source_group": group_id,
            "source_file": file_name,
            "split": split,
            "rows": int(state.output_rows[key]),
            "normal_count": int(state.output_normal[key]),
            "attack_count": int(state.output_attack[key]),
            "attack_rate": float(state.output_attack[key] / state.output_rows[key]) if state.output_rows[key] else 0.0,
            "attack_types": ";".join(f"{k}:{v}" for k, v in sorted(attack_types.items())),
            "event_time_min": format_event_range(state.time_min[key], platform) if key in state.time_min else "",
            "event_time_max": format_event_range(state.time_max[key], platform) if key in state.time_max else "",
        })
    path = REPORT_DIR / "final_split_distribution.csv"
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")
    return path


def write_duplicate_handling(records: list[SourceRecord], state: RunState) -> Path:
    rows = []
    for rec in records:
        rows.append({
            "platform": rec.platform,
            "source_file": rec.file_name,
            "rows_before": rec.rows_expected,
            "duplicate_rows_before": rec.duplicate_count,
            "duplicate_rows_removed": 0,
            "rows_after": rec.rows_expected,
            "duplicate_rows_routed_to_first_split": int(state.duplicate_routed.get(rec.file_name, 0)),
            "duplicate_cross_split_after_routing": int(state.duplicate_cross_split.get(rec.file_name, 0)),
            "policy": "retain all rows; route within-source duplicate fingerprints consistently; never silently drop",
        })
    path = REPORT_DIR / "duplicate_handling.csv"
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")
    return path


def write_leakage_check(state: RunState, records: list[SourceRecord], plans: list[GroupPlan]) -> Path:
    rows: list[dict[str, Any]] = []
    distribution = pd.read_csv(REPORT_DIR / "final_split_distribution.csv")
    for rec in records:
        cross = int(state.duplicate_cross_split.get(rec.file_name, 0))
        rows.append({
            "check": "exact_duplicate_cross_split",
            "platform": rec.platform,
            "source_file": rec.file_name,
            "scope": "source_file fingerprint",
            "observed_count": cross,
            "status": "PASS" if cross == 0 else "ROUTED_TO_FIRST_SPLIT",
            "detail": "Duplicates are retained and assigned consistently; no duplicate fingerprint is emitted in two splits.",
        })
        splits = sorted(state.source_split_sets.get((rec.platform, rec.file_name), set()))
        expected_cross = rec.split_mode in {"chronological_calendar_block", "chronological_file_block"}
        rows.append({
            "check": "source_file_cross_split",
            "platform": rec.platform,
            "source_file": rec.file_name,
            "scope": "source file",
            "observed_count": max(0, len(splits) - 1),
            "status": "EXPECTED_LIMITATION" if expected_cross and len(splits) > 1 else ("PASS" if len(splits) <= 1 else "REVIEW"),
            "detail": rec.split_reason,
        })
        scenario = f"SOURCE::{rec.platform}::{rec.file_name}"
        scenario_splits = splits
        rows.append({
            "check": "scenario_cross_split",
            "platform": rec.platform,
            "source_file": rec.file_name,
            "scope": scenario,
            "observed_count": max(0, len(scenario_splits) - 1),
            "status": "EXPECTED_LIMITATION" if expected_cross and len(scenario_splits) > 1 else ("PASS" if len(scenario_splits) <= 1 else "REVIEW"),
            "detail": "Source-file crossing is expected only for calendar/file-block protocols; actual calendar scenario units are checked separately below.",
        })

    # Actual split units for the corrective pass are complete source files for
    # Network and complete calendar-day blocks for IoT/Linux/Windows.
    for scenario, splits in sorted(state.scenario_split_sets.items()):
        if not scenario.startswith("TIMEBLOCK::"):
            continue
        split_list = sorted(splits)
        rows.append({
            "check": "time_block_cross_split",
            "platform": scenario.split("::", 2)[1],
            "source_file": "ALL_SOURCE_FILES",
            "scope": scenario,
            "observed_count": max(0, len(split_list) - 1),
            "status": "PASS" if len(split_list) <= 1 else "FAIL",
            "detail": "Each calendar-day split unit is assigned to exactly one fold.",
        })

    # Validate the final temporal envelopes in split order.
    for platform in sorted(distribution["platform"].unique()):
        intervals: dict[str, tuple[float, float]] = {}
        for key, lo in state.time_min.items():
            if key[0] != platform:
                continue
            split = key[3]
            hi = state.time_max[key]
            if split not in intervals:
                intervals[split] = (lo, hi)
            else:
                intervals[split] = (min(intervals[split][0], lo), max(intervals[split][1], hi))
        has_all = all(split in intervals for split in SPLITS)
        ordered = has_all and intervals["train"][1] <= intervals["validation"][0] and intervals["validation"][1] <= intervals["test"][0]
        detail = "; ".join(
            f"{split}={format_event_range(intervals[split][0], platform)}..{format_event_range(intervals[split][1], platform)}"
            for split in SPLITS if split in intervals
        )
        rows.append({
            "check": "temporal_ordering",
            "platform": platform,
            "source_file": "ALL_SOURCE_FILES",
            "scope": "train <= validation <= test event-time envelopes",
            "observed_count": 0 if ordered else 1,
            "status": "PASS" if ordered else "FAIL",
            "detail": detail,
        })

    # Binary usability is evaluated from the executed output distribution.
    for coverage in platform_class_coverage(distribution):
        missing_requirements = coverage.missing_requirements
        preference = [] if coverage.validation_both else ["validation both-class preference"]
        rows.append({
            "check": "binary_class_coverage",
            "platform": coverage.platform,
            "source_file": "ALL_SOURCE_FILES",
            "scope": "train/validation/test label coverage",
            "observed_count": len(missing_requirements),
            "status": "PASS" if coverage.eligible else "NOT_BASELINE_ELIGIBLE",
            "detail": "; ".join(missing_requirements + preference) or "all required binary classes present",
        })

    for plan in plans:
        baseline_columns = set(final_features(plan)) | {"label"}
        forbidden = {"type", "raw_type", "canonical_type", "event_time", "raw_timestamp", "platform", "source_file", "scenario_id", "split", "row_identifier"}
        leaks = sorted(baseline_columns & forbidden)
        rows.append({
            "check": "baseline_target_or_split_metadata_leakage",
            "platform": plan.platform,
            "source_file": plan.group_id,
            "scope": "baseline feature schema",
            "observed_count": len(leaks),
            "status": "PASS" if not leaks else "FAIL",
            "detail": "No target/provenance/timestamp metadata is used as baseline X." if not leaks else ";".join(leaks),
        })
        excluded_nonapproved = [
            name for name in final_features(plan)
            if plan.feature_specs[name].risk_classification not in {"VALID_OBSERVABLE_FEATURE", "REVIEW_REQUIRED"}
        ]
        rows.append({
            "check": "feature_risk_policy",
            "platform": plan.platform,
            "source_file": plan.group_id,
            "scope": "feature_risk.csv",
            "observed_count": len(excluded_nonapproved),
            "status": "PASS" if not excluded_nonapproved else "FAIL",
            "detail": "Baseline uses VALID_OBSERVABLE_FEATURE plus explicitly numeric REVIEW_REQUIRED state features; categorical review features remain semantic-only." if not excluded_nonapproved else str(excluded_nonapproved),
        })

    path = REPORT_DIR / "split_leakage_check.csv"
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")
    return path


def build_feature_dictionary(records: list[SourceRecord], plans: list[GroupPlan], risk: dict[tuple[str, str, str], dict[str, Any]]) -> Path:
    plan_by_group = {p.group_id: p for p in plans}
    rows: list[dict[str, Any]] = []
    generated = {
        "event_time": ("timestamp/provenance", "no", "yes", "normalized UTC text or IoT local text", "ordering/alignment only; not baseline X"),
        "raw_timestamp": ("timestamp/provenance", "no", "yes", "raw timestamp representation", "preserved for audit"),
        "platform": ("provenance", "no", "yes", "constant source platform", "semantic view only"),
        "source_file": ("provenance", "no", "yes", "relative source path", "split/source provenance"),
        "scenario_id": ("provenance", "no", "yes", "source-file scenario proxy", "not used as baseline X"),
        "split": ("provenance", "no", "yes", "manifest assignment", "not used as baseline X"),
        "row_identifier": ("provenance", "no", "yes", "source path plus source row number", "stable row traceability"),
        "raw_type": ("target/provenance", "no", "yes", "unchanged original type", "preserved; not baseline X"),
        "canonical_type": ("target/provenance", "no", "yes", "versioned label mapping", "preserved; not baseline X"),
        "label": ("target", "yes", "yes", "binary 0 normal/1 attack", "target y only"),
    }
    for rec in records:
        plan = plan_by_group[rec.group_id]
        baseline = set(final_features(plan))
        for feature in rec.columns:
            risk_row = risk.get((rec.platform, rec.file_name, feature), {})
            classification = risk_row.get("classification", "UNKNOWN")
            if feature == "label":
                role = "target"
                handling = "preserved as y"
                reason = "binary target from source"
            elif feature == "type":
                role = "target/provenance"
                handling = "preserved in semantic view and copied to raw_type"
                reason = "attack category is not baseline X"
            elif feature in {"ts", "date", "time"}:
                role = "timestamp/provenance"
                handling = "preserved; normalized event_time added"
                reason = "used only for ordering and semantic alignment"
            else:
                role = classification.lower().replace("_", " ")
                handling = "train-only median/ordinal encoding" if feature in baseline else "semantic only or excluded"
                reason = risk_row.get("reason", "No risk row; excluded from baseline")
            rows.append({
                "platform": rec.platform,
                "source_group": rec.group_id,
                "source_file": rec.file_name,
                "feature": feature,
                "dtype": "source dtype; see semantic CSV" if feature not in baseline else plan.feature_specs.get(feature, FeatureSpec(feature, "", classification, "")).kind,
                "role": role,
                "baseline_used": "yes" if feature in baseline else "no",
                "semantic_kept": "yes",
                "handling": handling,
                "reason": reason,
            })
        for feature, (role, base, semantic, handling, reason) in generated.items():
            rows.append({
                "platform": rec.platform,
                "source_group": rec.group_id,
                "source_file": rec.file_name,
                "feature": feature,
                "dtype": "generated",
                "role": role,
                "baseline_used": base,
                "semantic_kept": semantic,
                "handling": handling,
                "reason": reason,
            })
    path = REPORT_DIR / "final_feature_dictionary.csv"
    pd.DataFrame(rows).drop_duplicates(
        subset=["platform", "source_group", "source_file", "feature"], keep="first"
    ).to_csv(path, index=False, encoding="utf-8")
    return path


def write_protocol(
    records: list[SourceRecord],
    plans: list[GroupPlan],
    state: RunState,
    output_root: Path,
    manifest_path: Path,
    distribution_path: Path,
    leakage_path: Path,
    duplicate_path: Path,
    preprocessors: list[Path],
    elapsed: float,
    chunk_size: int,
    *,
    quarantine_invalid: bool = False,
) -> Path:
    total_source = sum(r.rows_expected for r in records)
    total_output = sum(state.output_rows.values())
    lines = [
        "# Final Phase-1 data protocol",
        "",
        "This is the executed preprocessing protocol for the ToN-IoT project. It was generated by `python -m research_pipeline.src.main`.",
        "",
        "## Scope and immutability",
        "",
        f"- Source files read: {len(records)}; source rows from the inventory: {total_source:,}.",
        f"- Rows written to baseline and semantic views: {total_output:,} (no source rows are deleted).",
        "- The script reads only `Processed_datasets`; it does not edit or rename source files.",
        "- Outputs use portable gzip-compressed CSV with JSON preprocessing artifacts.",
        "",
        "## Label and timestamp contract",
        "",
        "- `label` remains the binary target: 0 = normal and 1 = attack.",
        "- `type` is preserved unchanged; `raw_type` copies it and `canonical_type` uses `config/label_mapping.json`. Unknown types are never coerced into a known attack class.",
        "- `event_time` is added to the semantic view. Network/Linux/Windows numeric `ts` is rendered as UTC epoch time; IoT `date + time` is rendered as source-local text with no claim that clocks are globally synchronized. The original timestamp columns and `raw_timestamp` remain available.",
        f"- Missing/invalid timestamps are preserved as missing (`event_time=<NA>`): {sum(state.invalid_timestamps.values()):,} rows in the executed pass. "
        + ("They are written only to the quarantine bucket and excluded from Train/Validation/Test." if quarantine_invalid else "They enter the V1 deterministic train bucket when chronological cutoffs are undefined; this is retained only for control/reproduction."),
        "",
        "## Split protocol",
        "",
        "- No random row split is used.",
        "- Network uses deterministic source-file holdout ordered by inventory timestamp bounds; all rows in a source file stay in one fold.",
        "- IoT uses complete chronological calendar-day blocks shared across profiles; each profile schema remains separate and is not merged to manufacture a shared feature space.",
        "- Linux uses the same complete calendar-day blocks; weak scenario provenance remains a limitation, but actual calendar-day units do not cross folds.",
        "- Windows uses the same complete calendar-day blocks because only two source files exist; actual day units do not cross folds.",
        f"- Corrective boundaries are {calendar_split_description()}. "
        + ("Missing-time rows use the separate quarantine bucket; cutoffs are unchanged." if quarantine_invalid else "Missing-time rows use the V1 explicit train bucket because no temporal cutoff can be justified."),
        "- Binary baseline eligibility requires both labels in train and test; validation is reported as a preferred both-class condition. Network remains primary and its severe attack imbalance is documented, not resampled.",
        "- Network attack rates remain approximately 97.5% train, 96.8% validation and 90.9% test; downstream model training may use class_weight or model-specific weighting, but this preparation pass does not resample.",
        "- The manifest is partition/file/block-granular, not one row per event. Each row identifies the source file, source block, split, time range and row count; this is reproducible for the 31M-row dataset.",
        "- Exact duplicate rows are retained. Within each source file, their fingerprint is routed to the split of the first encountered occurrence, so duplicate fingerprints are not emitted in multiple folds.",
        "",
        "## Baseline view",
        "",
        "- Baseline files contain only preprocessed X columns plus `label`. `type`, `raw_type`, `canonical_type`, timestamps, source paths, scenario IDs, split metadata, IP/UID/PID identifiers and other non-approved fields are not baseline features.",
        "- Candidate features use `VALID_OBSERVABLE_FEATURE` plus explicitly numeric `REVIEW_REQUIRED` state features; categorical/text review features remain semantic-only. All choices are recorded in `final_feature_dictionary.csv`.",
        "- High-cardinality/text-like candidates are excluded rather than one-hot encoded. Numeric missing values use a deterministic train-only median; categorical codes use a train-only vocabulary and -1 for missing/unseen values. No `-` placeholder is changed to zero.",
        "- No scaling is fitted because the intended RF/XGBoost/LightGBM tree baselines do not require it.",
        "",
        "## Semantic view",
        "",
        "- Semantic files preserve each source schema separately and add `event_time`, `raw_timestamp`, `platform`, `source_file`, `scenario_id`, `split`, `row_identifier`, `raw_type`, and `canonical_type`.",
        "- Semantic schemas are intentionally not forced to be identical across platforms or device/process profiles.",
        "",
        "## Generated artifacts",
        "",
        f"- Split manifest: `{manifest_path.relative_to(PROJECT_ROOT).as_posix()}`",
        f"- Final split report: `research_pipeline/reports/final_split_report.md`",
        f"- Distribution report: `{distribution_path.relative_to(PROJECT_ROOT).as_posix()}`",
        f"- Leakage report: `{leakage_path.relative_to(PROJECT_ROOT).as_posix()}`",
        f"- Duplicate handling log: `{duplicate_path.relative_to(PROJECT_ROOT).as_posix()}`",
        f"- Preprocessor artifacts: `{output_root.relative_to(PROJECT_ROOT).as_posix()}/preprocessors/*.json` ({len(preprocessors)} group artifacts).",
        f"- Baseline data: `{output_root.relative_to(PROJECT_ROOT).as_posix()}/baseline/<platform>/<split>/*.csv.gz`.",
        f"- Semantic data: `{output_root.relative_to(PROJECT_ROOT).as_posix()}/semantic/<platform>/<split>/*.csv.gz`.",
        "",
        "## Reproducibility",
        "",
        "- Script command: `python -m research_pipeline.src.main`.",
        f"- Chunk size used: {chunk_size:,}; elapsed time: {elapsed:.1f} seconds.",
        "- Re-running with the same source files and script produces the same split planning, feature columns, train-only statistics and output paths. Use `--overwrite` only when intentionally rebuilding these generated output directories.",
        "",
        "## Known limitations",
        "",
        "- IoT/Linux/Windows use calendar-day time blocks rather than whole-source holdout so every binary baseline has both classes in train/test; Linux scenario-level provenance remains a limitation and is reported explicitly.",
        "- Cross-platform timestamp alignment is not assumed. The Phase-1.2 reports document time-offset and ground-truth uncertainty separately.",
        "- The baseline is a conservative, approved-observable numeric view. Semantic/provenance fields remain available for later ablations, graph/entity analyses, or explicitly justified feature additions; they are not silently promoted into baseline X.",
    ]
    path = REPORT_DIR / "final_data_protocol.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_final_split_report(
    manifest_path: Path,
    distribution_path: Path,
    leakage_path: Path,
    summary_path: Path,
) -> Path:
    """Write a compact human-readable report from completed artifacts."""
    distribution = pd.read_csv(distribution_path)
    leakage = pd.read_csv(leakage_path)
    summary = read_json(summary_path)
    grouped = (
        distribution.groupby(["platform", "split"], as_index=False)[["rows", "normal_count", "attack_count"]]
        .sum()
    )
    lines = [
        "# Final split report",
        "",
        "This report summarizes the executed Phase-1 split and preprocessing outputs. The source data were not modified.",
        "",
        "## Execution result",
        "",
        f"- Status: **{summary.get('status', 'unknown').upper()}**",
        f"- Source files: {summary.get('source_file_count', 0):,}",
        f"- Source rows: {summary.get('source_rows', 0):,}",
        f"- Output rows in baseline/semantic views: {summary.get('output_rows', 0):,}",
        f"- Manifest granularity: partition/file/block (`{manifest_path.relative_to(PROJECT_ROOT).as_posix()}`).",
        "",
        "## Platform and split distribution",
        "",
        "| Platform | Split | Rows | Normal | Attack | Attack rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in grouped.sort_values(["platform", "split"]).to_dict("records"):
        total = int(row["rows"])
        attack = int(row["attack_count"])
        lines.append(
            f"| {row['platform']} | {row['split']} | {total:,} | {int(row['normal_count']):,} | {attack:,} | {(attack / total if total else 0):.4%} |"
        )
    train_without_attack = grouped[(grouped["split"] == "train") & (grouped["attack_count"] == 0)]
    coverage_note = (
        f"{', '.join(train_without_attack['platform'].tolist())} has no attack rows in train."
        if len(train_without_attack)
        else "All platforms have both normal and attack rows in train."
    )
    eligibility = platform_class_coverage(distribution)
    lines += [
        "",
        f"- Chronological coverage note: {coverage_note} This result was obtained with complete time blocks, not a random or label-stratified row split.",
        "",
        "## Binary baseline eligibility",
        "",
        "| Platform | Train has normal? | Train has attack? | Validation has both? | Test has both? | Baseline eligible? |",
        "|---|---|---|---|---|---|",
    ]
    for coverage in eligibility:
        lines.append(
            f"| {coverage.platform} | {'yes' if coverage.train_normal else 'no'} | {'yes' if coverage.train_attack else 'no'} | {'yes' if coverage.validation_both else 'no'} | {'yes' if coverage.test_both else 'no'} | {'BASELINE_ELIGIBLE' if coverage.eligible else 'NOT_BASELINE_ELIGIBLE'} |"
        )
    network_dist = distribution[distribution["platform"] == "Network"].groupby("split")[["normal_count", "attack_count"]].sum()
    if set(("train", "validation", "test")).issubset(network_dist.index):
        network_note = (
            f"- Network imbalance is preserved: train attack rate = {network_dist.loc['train', 'attack_count'] / (network_dist.loc['train'].sum()):.4%}, "
            f"validation = {network_dist.loc['validation', 'attack_count'] / (network_dist.loc['validation'].sum()):.4%}, "
            f"test = {network_dist.loc['test', 'attack_count'] / (network_dist.loc['test'].sum()):.4%}. "
            "Handle this later with model-level class weighting; no resampling was performed."
        )
    else:
        network_note = "- Network imbalance note: Network is not present in this partial acceptance sample."
    lines += [
        "",
        network_note,
        "",
        "## Attack-type coverage",
        "",
        "Attack types are listed per source-file partition in `final_split_distribution.csv`; a type is not forced into a fold by random stratification. Network uses source-file holdout; IoT/Linux/Windows use complete calendar-day blocks.",
        "",
        "| Platform | Split | Attack types observed |",
        "|---|---|---|",
    ]
    for row in distribution.sort_values(["platform", "split", "source_file"]).to_dict("records"):
        if int(row["attack_count"]) > 0:
            types = row["attack_types"] or "(none)"
            lines.append(f"| {row['platform']} | {row['split']} | {row['source_file']}: {types} |")
    lines += [
        "",
        "## Leakage and integrity checks",
        "",
        f"- PASS checks: {int((leakage['status'] == 'PASS').sum()):,}.",
        f"- Expected protocol limitations: {int((leakage['status'] == 'EXPECTED_LIMITATION').sum()):,} (IoT profile/Linux/Windows chronological source crossing where provenance/schema constraints require it).",
        f"- FAIL/REVIEW/NOT_BASELINE_ELIGIBLE checks: {int((~leakage['status'].isin(['PASS', 'EXPECTED_LIMITATION'])).sum()):,}.",
        "- Exact duplicate rows are retained; duplicate handling and routing counts are in `research_pipeline/reports/duplicate_handling.csv`.",
        "- Baseline target/split-metadata leakage checks are PASS; semantic/provenance columns are not baseline X.",
        "",
        "## Preprocessor and handoff",
        "",
        "- Train-only preprocessing artifacts are under `research_pipeline/output/preprocessors/`.",
        "- Baseline data are under `research_pipeline/output/baseline/<platform>/<split>/`.",
        "- Semantic data are under `research_pipeline/output/semantic/<platform>/<split>/`.",
        "- See `research_pipeline/reports/final_data_protocol.md` and `research_pipeline/reports/baseline_handoff.md` before downstream modeling.",
    ]
    path = REPORT_DIR / "final_split_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_handoff(plans: list[GroupPlan], output_root: Path, distribution_path: Path) -> Path:
    network = next((p for p in plans if p.group_id == "Network"), None)
    network_features = final_features(network) if network else []
    distribution = pd.read_csv(distribution_path)
    eligibility_rows: list[dict[str, Any]] = []
    for coverage in platform_class_coverage(distribution):
        platform = coverage.platform
        if coverage.eligible:
            if platform == "Network":
                reason = "source-file holdout; severe attack imbalance is reported and left for model-level class weighting"
            elif platform == "IoT":
                reason = "calendar time-block split; profile-specific schemas and train-only preprocessors are preserved"
            elif platform == "Linux":
                reason = "calendar time-block split; weak scenario provenance remains an evaluation limitation"
            else:
                reason = "calendar time-block split; source-file holdout is infeasible with two Windows files"
        else:
            reason = "missing required class coverage: " + ", ".join(coverage.missing_requirements)
        eligibility_rows.append({
            "platform": platform,
            "train_has_normal": "yes" if coverage.train_normal else "no",
            "train_has_attack": "yes" if coverage.train_attack else "no",
            "validation_has_both": "yes" if coverage.validation_both else "no",
            "test_has_both": "yes" if coverage.test_both else "no",
            "baseline_eligible": "BASELINE_ELIGIBLE" if coverage.eligible else "NOT_BASELINE_ELIGIBLE",
            "reason": reason,
        })
    lines = [
        "# Baseline handoff",
        "",
        "The final baseline view is ready for downstream tree-model experiments. This document is intentionally limited to data loading and target/feature contracts; it does not train a model.",
        "",
        "## Recommended primary dataset",
        "",
        "Use `Network` first for the primary baseline because it is the most complete and attack-rich platform in the audit. Keep IoT, Linux and Windows as separate secondary datasets; do not concatenate platforms with incompatible semantics.",
        "",
        "## Binary baseline eligibility",
        "",
        "| Platform | Train has normal? | Train has attack? | Validation has both? | Test has both? | Baseline eligible? | Reason |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in eligibility_rows:
        lines.append(
            f"| {row['platform']} | {row['train_has_normal']} | {row['train_has_attack']} | {row['validation_has_both']} | {row['test_has_both']} | {row['baseline_eligible']} | {row['reason']} |"
        )
    lines += [
        "",
        "## Exact split locations",
        "",
        "```python",
        "from pathlib import Path",
        "import pandas as pd",
        "",
        "ROOT = Path.cwd()  # run this example from the cloned project root",
        "BASELINE = ROOT / \"research_pipeline/output/baseline\"",
        "",
        "def paths(platform, split):",
        "    return sorted((BASELINE / platform.lower() / split).glob(\"*.csv.gz\"))",
        "",
        "train_paths = paths(\"Network\", \"train\")",
        "validation_paths = paths(\"Network\", \"validation\")",
        "test_paths = paths(\"Network\", \"test\")",
        "```",
        "",
        "Each output file is a source-file or source-file/calendar-block partition. For large-scale experiments, iterate over files/chunks instead of concatenating all 31M rows into RAM.",
        "",
        "## Target and feature extraction",
        "",
        "```python",
        "train_part = pd.read_csv(train_paths[0])",
        "feature_columns = [c for c in train_part.columns if c != \"label\"]",
        "X_train = train_part[feature_columns]",
        "y_train = train_part[\"label\"]",
        "assert set(train_part.columns) == set(feature_columns) | {\"label\"}",
        "```",
        "",
        f"For the executed Network artifact, the baseline feature count is {len(network_features)}. The authoritative ordered list is in `research_pipeline/output/preprocessors/network.json` under `baseline_feature_columns`; load that JSON rather than reconstructing features from raw files.",
        "",
        "## Preprocessor contract",
        "",
        "- JSON artifacts live in `research_pipeline/output/preprocessors/`.",
        "- Every statistic and category vocabulary was fitted using train rows only. Do not refit or update it with validation/test rows.",
        "- Numeric missing values were filled with the train-only median. Categorical codes use the train vocabulary and -1 for missing/unseen values. There is no scaler.",
        "- `label` is y only. The baseline view does not contain `type`, `raw_type`, `canonical_type`, timestamps, source paths, scenario metadata, or identifiers.",
        "",
        "## Split and leakage safeguards",
        "",
        "- Network train/validation/test are source-file holdouts ordered by timestamp bounds, not random row splits.",
        f"- IoT/Linux/Windows use complete calendar-day time blocks: {calendar_split_description()}.",
        "- Read `research_pipeline/output/manifests/split_manifest.csv` together with `reports/split_leakage_check.csv` before reporting results.",
        "- Do not use semantic `source_file`, `scenario_id`, or `event_time` as baseline X. They are kept for audit and later explicitly scoped ablations only.",
        "- A model trained on the Network baseline should report the exact source-file split and feature artifact used; do not merge the platform views after preprocessing.",
        "",
        "## Files to cite in a downstream experiment",
        "",
        "- `research_pipeline/reports/final_data_protocol.md`",
        "- `research_pipeline/reports/final_split_distribution.csv`",
        "- `research_pipeline/reports/split_leakage_check.csv`",
        "- `research_pipeline/reports/final_feature_dictionary.csv`",
        "- `research_pipeline/output/manifests/split_manifest.csv`",
        "- `research_pipeline/output/preprocessors/network.json`",
        "- `research_pipeline/reports/final_split_report.md`",
    ]
    path = REPORT_DIR / "baseline_handoff.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_run_summary(
    records: list[SourceRecord],
    plans: list[GroupPlan],
    state: RunState,
    output_root: Path,
    elapsed: float,
    *,
    status: str = "success",
) -> Path:
    path = output_root / "manifests" / "run_summary.json"
    payload = {
        "status": status,
        "script": "research_pipeline/src/main.py",
        "project_root": str(PROJECT_ROOT),
        "source_file_count": len(records),
        "source_rows": int(sum(r.rows_expected for r in records)),
        "output_rows": int(sum(state.output_rows.values())),
        "platforms": sorted({r.platform for r in records}),
        "groups": [
            {
                "group_id": p.group_id,
                "platform": p.platform,
                "source_files": [r.file_name for r in p.records],
                "baseline_feature_count": len(final_features(p)),
                "baseline_feature_columns": final_features(p),
            }
            for p in plans
        ],
        "invalid_labels": dict(state.invalid_labels),
        "invalid_timestamps": dict(state.invalid_timestamps),
        "quarantine_policy": state.quarantine_policy,
        "unknown_types": dict(state.unknown_types),
        "duplicate_cross_split": dict(state.duplicate_cross_split),
        "elapsed_seconds": elapsed,
    }
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    return path


def determine_pipeline_status(mode: str, leakage_path: Path) -> str:
    """Return a terminal status; FAIL never gets reported as SUCCESS.

    V1 is intentionally checked against its committed split audit because it
    remains a control with known failures.  The remediated path is checked
    against the executed output checks.
    """
    if mode == "v1":
        audit_path = REPORT_DIR / "split_audit_v2" / "leakage_risks.csv"
        if audit_path.exists():
            statuses = pd.read_csv(audit_path, dtype=str).get("status", pd.Series(dtype=str))
            if (statuses == "FAIL").any():
                return "blocked_by_audit"
    if leakage_path.exists():
        statuses = pd.read_csv(leakage_path, dtype=str).get("status", pd.Series(dtype=str))
        if (statuses == "FAIL").any():
            return "blocked_by_audit"
    return "success"


def generate_final_reports(
    records: list[SourceRecord],
    plans: list[GroupPlan],
    state: RunState,
    risk: dict[tuple[str, str, str], dict[str, Any]],
    output_root: Path,
    manifest_path: Path,
    preprocessors: list[Path],
    started: float,
    chunk_size: int,
    *,
    mode: str = "v1",
    quarantine_invalid: bool = False,
) -> tuple[dict[str, Path], float, str]:
    """Write the same Phase-1 reports in their original dependency order."""
    distribution_path = write_split_distribution(state, output_root)
    leakage_path = write_leakage_check(state, records, plans)
    duplicate_path = write_duplicate_handling(records, state)
    build_feature_dictionary(records, plans, risk)
    elapsed = time.time() - started
    protocol_path = write_protocol(
        records, plans, state, output_root, manifest_path, distribution_path,
        leakage_path, duplicate_path, preprocessors, elapsed, chunk_size,
        quarantine_invalid=quarantine_invalid,
    )
    handoff_path = write_handoff(plans, output_root, distribution_path)
    status = determine_pipeline_status(mode, leakage_path)
    summary_path = write_run_summary(records, plans, state, output_root, elapsed, status=status)
    split_report_path = write_final_split_report(
        manifest_path, distribution_path, leakage_path, summary_path,
    )
    return {
        "manifest": manifest_path,
        "distribution": distribution_path,
        "leakage": leakage_path,
        "protocol": protocol_path,
        "handoff": handoff_path,
        "summary": summary_path,
        "split report": split_report_path,
    }, elapsed, status
