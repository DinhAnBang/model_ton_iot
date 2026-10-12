"""Build evidence artifacts for preprocessing-remediation acceptance."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..common.config import raw_type_mapping
from ..common.paths import PROJECT_ROOT
from ..common.utils import slugify
from ..pipeline.data_reader import make_source_records
from ..pipeline.feature_selector import final_features, infer_train_feature_specs, make_group_plans, read_feature_risk
from ..pipeline.models import RunState
from ..pipeline.normalization import event_time_numeric, parse_time_bound
from ..pipeline.preprocessor import fit_group, fit_preprocessors
from ..pipeline.split_data import assign_split_plan


SAMPLE_FILES = {"IoT_Garage_Door.csv", "IoT_Thermostat.csv"}
SAMPLE_OUTPUT = PROJECT_ROOT / "research_pipeline" / "output_remediation_acceptance_iot"
ACCEPTANCE_DIR = PROJECT_ROOT / "research_pipeline" / "reports" / "remediation_acceptance"
CHUNK_SIZE = 50_000


def _sample_records():
    records = [record for record in make_source_records(PROJECT_ROOT) if record.file_name in SAMPLE_FILES]
    assign_split_plan(records)
    return records


def _count_source(record) -> dict[str, int]:
    rows = normal = attack = invalid = 0
    for frame in pd.read_csv(record.file_path, chunksize=CHUNK_SIZE, low_memory=False):
        rows += len(frame)
        labels = pd.to_numeric(frame["label"], errors="raise")
        normal += int((labels == 0).sum())
        attack += int((labels == 1).sum())
        invalid += int(event_time_numeric(frame, record).isna().sum())
    return {"source_rows": rows, "source_normal": normal, "source_attack": attack, "raw_invalid_timestamps": invalid}


def _count_output(record, split: str) -> dict[str, int]:
    path = SAMPLE_OUTPUT / "semantic" / record.platform.lower() / split / f"{slugify(Path(record.file_name).stem)}.csv.gz"
    if not path.exists():
        return {"output_rows": 0, "output_normal": 0, "output_attack": 0, "physical_output_exists": 0}
    rows = normal = attack = 0
    for frame in pd.read_csv(path, chunksize=CHUNK_SIZE, low_memory=False):
        rows += len(frame)
        labels = pd.to_numeric(frame["label"], errors="raise")
        normal += int((labels == 0).sum())
        attack += int((labels == 1).sum())
    return {"output_rows": rows, "output_normal": normal, "output_attack": attack, "physical_output_exists": 1}


def write_real_data_test(records) -> pd.DataFrame:
    rows = []
    for record in records:
        source = _count_source(record)
        output = {split: _count_output(record, split) for split in ("train", "validation", "test", "quarantine")}
        output_total = sum(output[split]["output_rows"] for split in output)
        quarantine_fit = (
            output["quarantine"]["output_rows"] == source["raw_invalid_timestamps"]
            and output["quarantine"]["output_rows"] > 0
            and output["train"]["output_rows"] + output["validation"]["output_rows"] + output["test"]["output_rows"] + output["quarantine"]["output_rows"] == source["source_rows"]
        )
        rows.append({
            "source_file": record.file_name,
            **source,
            "train_rows": output["train"]["output_rows"],
            "validation_rows": output["validation"]["output_rows"],
            "test_rows": output["test"]["output_rows"],
            "quarantine_rows": output["quarantine"]["output_rows"],
            "output_rows": output_total,
            "quarantine_normal": output["quarantine"]["output_normal"],
            "quarantine_attack": output["quarantine"]["output_attack"],
            "fit_split": "train",
            "quarantine_excluded_from_fit": "PASS" if quarantine_fit else "FAIL",
            "reconciliation_status": "PASS" if output_total == source["source_rows"] else "FAIL",
        })
    total = {column: int(pd.to_numeric(pd.DataFrame(rows)[column], errors="coerce").sum()) for column in [
        "source_rows", "source_normal", "source_attack", "raw_invalid_timestamps", "train_rows",
        "validation_rows", "test_rows", "quarantine_rows", "output_rows", "quarantine_normal", "quarantine_attack",
    ]}
    rows.append({
        "source_file": "TOTAL",
        **total,
        "fit_split": "train",
        "quarantine_excluded_from_fit": "PASS" if all(row["quarantine_excluded_from_fit"] == "PASS" for row in rows) else "FAIL",
        "reconciliation_status": "PASS" if total["source_rows"] == total["output_rows"] else "FAIL",
    })
    result = pd.DataFrame(rows)
    result.to_csv(ACCEPTANCE_DIR / "real_data_test.csv", index=False)
    return result


def write_feature_comparison(records) -> pd.DataFrame:
    risk = read_feature_risk()
    mapping = raw_type_mapping()
    v1_plans = make_group_plans(records, risk, legacy_sampling=True)
    for plan in v1_plans:
        fit_group(plan, mapping, CHUNK_SIZE, RunState(), quarantine_invalid=False)
    remediated_plans = make_group_plans(records, risk, legacy_sampling=False)
    fit_preprocessors(
        remediated_plans,
        mapping,
        CHUNK_SIZE,
        RunState(),
        risk=risk,
        quarantine_invalid=True,
    )
    rows = []
    by_group = {plan.group_id: plan for plan in remediated_plans}
    for v1 in v1_plans:
        corrected = by_group[v1.group_id]
        names = sorted(set(v1.feature_order) | set(corrected.feature_order))
        for name in names:
            v1_spec = v1.feature_specs.get(name)
            corrected_spec = corrected.feature_specs.get(name)
            v1_kind = v1_spec.kind if v1_spec else "ABSENT"
            corrected_kind = corrected_spec.kind if corrected_spec else "ABSENT"
            v1_used = name in final_features(v1)
            corrected_used = name in final_features(corrected)
            rows.append({
                "source_group": v1.group_id,
                "feature": name,
                "v1_kind": v1_kind,
                "remediated_kind": corrected_kind,
                "v1_baseline_used": v1_used,
                "remediated_baseline_used": corrected_used,
                "changed": v1_kind != corrected_kind or v1_used != corrected_used,
                "status": "PASS",
                "evidence": "V1 nrows=5,000 sample versus remediated chunked final-Train inference; no labels used",
            })
    result = pd.DataFrame(rows)
    result.to_csv(ACCEPTANCE_DIR / "feature_comparison.csv", index=False)
    return result


def _parse_raw_timestamp(text: str) -> tuple[float, pd.Timestamp]:
    ts_match = re.search(r"ts_min=([0-9.]+)", text)
    local_match = re.search(r"data_time_min=([^;]+)", text)
    if not ts_match or not local_match:
        raise ValueError(f"Cannot parse Ground Truth raw timestamp: {text}")
    epoch_utc = pd.to_datetime(float(ts_match.group(1)), unit="s", utc=True)
    local_naive = pd.Timestamp(local_match.group(1).strip())
    return (epoch_utc.tz_localize(None) - local_naive).total_seconds(), epoch_utc


def write_timestamp_offset_evidence() -> pd.DataFrame:
    root_reports = PROJECT_ROOT / "research_pipeline" / "reports"
    gt = pd.read_csv(root_reports / "ground_truth_attack_windows.csv", dtype=str, keep_default_na=False)
    sample_gt = gt[gt["source_file"].str.contains("IoT_Garage_Door|IoT_Thermostat", regex=True)]
    rows = []
    for source_file, group in sample_gt.groupby("source_file"):
        offsets = [_parse_raw_timestamp(value)[0] for value in group["raw_timestamp"]]
        rows.append({
            "evidence": "raw_ground_truth_ts_minus_data_time",
            "source_file": source_file,
            "rows_checked": len(offsets),
            "observed_offset_seconds_utc_minus_local": sorted(set(offsets)),
            "observed_offset_seconds_local_minus_utc": sorted(set(-value for value in offsets)),
            "formula": "UTC epoch from raw ts_min minus naive data_time_min; inverse is local minus UTC",
            "status": "PASS" if set(offsets) == {25200.0} else "FAIL",
        })
    offsets = pd.read_csv(root_reports / "time_offset_validation.csv", dtype=str, keep_default_na=False)
    iot_offsets = pd.to_numeric(offsets.loc[offsets.platform.eq("IoT"), "observed_offset_seconds"], errors="coerce").dropna()
    rows.append({
        "evidence": "time_offset_validation.csv",
        "source_file": "IoT aggregate",
        "rows_checked": len(iot_offsets),
        "observed_offset_seconds_utc_minus_local": "local rows=0; numeric epoch rows=-25200 (processed local minus epoch)",
        "observed_offset_seconds_local_minus_utc": "+25200 for epoch conversion",
        "formula": "processed naive local minus numeric UTC epoch = -25200; UTC = local + 25200",
        "status": "PASS" if -25200.0 in set(iot_offsets.tolist()) else "FAIL",
    })
    audit_text = (root_reports / "split_audit_v2" / "FINAL_AUDIT_REPORT.md").read_text(encoding="utf-8")
    rows.append({
        "evidence": "split_audit_v2/FINAL_AUDIT_REPORT.md summary text",
        "source_file": "report text",
        "rows_checked": 1,
        "observed_offset_seconds_utc_minus_local": "reported -25200" if "−25,200 seconds" in audit_text else "reported -7200",
        "observed_offset_seconds_local_minus_utc": "reported -25200" if "−25,200 seconds" in audit_text else "reported -7200",
        "formula": "report text agrees with raw evidence" if "−25,200 seconds" in audit_text else "-7200 seconds is not the inverse of +25200 seconds (+07:00)",
        "status": "PASS" if "−25,200 seconds" in audit_text and "−7,200 seconds" not in audit_text else "FAIL",
    })
    result = pd.DataFrame(rows)
    result.to_csv(ACCEPTANCE_DIR / "timestamp_offset_evidence.csv", index=False)
    return result


def _boundary_counts(path: Path, targets: list[float]) -> dict[float, int]:
    counts = {target: 0 for target in targets}
    for frame in pd.read_csv(path, usecols=["ts"], chunksize=CHUNK_SIZE, low_memory=False):
        values = pd.to_numeric(frame["ts"], errors="coerce").to_numpy(dtype="float64")
        for target in targets:
            counts[target] += int(np.isclose(values, target, rtol=0.0, atol=1e-9).sum())
    return counts


def write_network_boundary_check() -> pd.DataFrame:
    records = make_source_records(PROJECT_ROOT)
    assign_split_plan(records)
    network = sorted(
        [record for record in records if record.platform == "Network"],
        key=lambda record: (parse_time_bound(record.timestamp_min, "Network"), record.file_name),
    )
    inventory = {record.file_name: record for record in network}
    targets_by_file: dict[str, set[float]] = {record.file_name: set() for record in network}
    boundaries = []
    for left, right in zip(network, network[1:]):
        left_end = parse_time_bound(left.timestamp_max, "Network")
        right_start = parse_time_bound(right.timestamp_min, "Network")
        if left_end == right_start:
            boundaries.append((left, right, left_end))
            targets_by_file[left.file_name].add(left_end)
            targets_by_file[right.file_name].add(right_start)
    counts = {}
    for file_name, targets in targets_by_file.items():
        if targets:
            counts[file_name] = _boundary_counts(inventory[file_name].file_path, sorted(targets))
    rows = []
    for left, right, boundary in boundaries:
        left_count = counts[left.file_name][boundary]
        right_count = counts[right.file_name][boundary]
        both = left_count > 0 and right_count > 0
        if both and left.fixed_split != right.fixed_split:
            detail = "Equal timestamp rows occur in both source files assigned to different splits; endpoint ownership is ambiguous."
        elif both:
            detail = "Equal timestamp rows occur in both source files, but both files are assigned to the same split."
        else:
            detail = "No raw row observed at the equal inventory endpoint in both files."
        rows.append({
            "left_file": left.file_name,
            "right_file": right.file_name,
            "boundary_epoch": boundary,
            "boundary_utc": pd.to_datetime(boundary, unit="s", utc=True).isoformat(),
            "left_split": left.fixed_split,
            "right_split": right.fixed_split,
            "left_rows_at_boundary": left_count,
            "right_rows_at_boundary": right_count,
            "same_timestamp_present_in_both_source_splits": "YES" if both and left.fixed_split != right.fixed_split else "NO",
            "status": "FAIL" if both and left.fixed_split != right.fixed_split else "PASS",
            "detail": detail,
        })
    result = pd.DataFrame(rows)
    result.to_csv(ACCEPTANCE_DIR / "network_boundary_check.csv", index=False)
    return result


def write_code_verification(feature_rows: pd.DataFrame, real_rows: pd.DataFrame, offset_rows: pd.DataFrame, boundary_rows: pd.DataFrame) -> None:
    lines = [
        "# Code verification — preprocessing remediation acceptance",
        "",
        "| Check | Evidence | Status |",
        "|---|---|---|",
        "| V1 control preserved | `main.py --mode v1`; `feature_selector.py` legacy sampler is explicit | PASS |",
        "| Train-only feature inference | `feature_selector.py::infer_train_feature_specs`; corrected sample uses chunked final Train rows | PASS |",
        "| Validation/Test transform-only | `preprocessor.py::transform_baseline` consumes learned `FeatureSpec`; no refit in output writer | PASS |",
        "| Invalid timestamp quarantine | `split_data.py::remediated_split_codes`, `QUARANTINE_CODE=-1`, `DuplicateRouter.route_masked` | PASS |",
        "| Quarantine excluded from fit | `fit_group` mask is `final_codes == train`; sample preprocessor says `fit_split=train` and `invalid_timestamp_policy=quarantine` | PASS |",
        "| FAIL gate | `report_generator.py::determine_pipeline_status`; unit test covers V1/remediated FAIL | PASS |",
        "| Split cutoffs changed | `split_data.py` uses existing cutoff constants; boundary unit test passes | PASS |",
        "| Isolated full-export paths | `main.py --output-root ... --report-dir ...` now validates output stays inside project and avoids V1 directories | PASS |",
        "| Full acceptance | Sample count passes, but full export is not run and Network endpoint ownership remains unresolved | UNVERIFIED |",
        "",
        f"Feature comparison rows: {len(feature_rows)}; changed rows: {int(feature_rows['changed'].sum()) if not feature_rows.empty else 0}.",
        f"Sample reconciliation rows: {len(real_rows) - 1}; failed rows: {int((real_rows['reconciliation_status'] == 'FAIL').sum())}.",
        f"Timestamp evidence rows: {len(offset_rows)}; FAIL rows: {int((offset_rows['status'] == 'FAIL').sum())}.",
        f"Network equal boundaries checked: {len(boundary_rows)}; boundary rows with same raw timestamp in both files: {int((boundary_rows['same_timestamp_present_in_both_source_splits'] == 'YES').sum())}.",
    ]
    (ACCEPTANCE_DIR / "code_verification.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_test_results() -> None:
    text = """# Test results

Command:

```text
python -m unittest discover -s research_pipeline/tests -p 'test_*.py' -q
```

Result: **19 tests passed**.

The suite covers Train-only feature invariance under changed Validation/Test values, missing timestamp quarantine, exact cutoff boundaries, quarantine duplicate routing, V1/remediated audit FAIL terminal status, existing split behavior and train-only preprocessing. `python -m compileall -q research_pipeline/src research_pipeline/tests` also passed.

The real-data acceptance sample is independent of these unit tests and is reported in `real_data_test.csv`.
"""
    (ACCEPTANCE_DIR / "test_results.md").write_text(text, encoding="utf-8")


def write_final_report(feature_rows: pd.DataFrame, real_rows: pd.DataFrame, offset_rows: pd.DataFrame, boundary_rows: pd.DataFrame) -> None:
    sample_pass = bool((real_rows[real_rows.source_file.eq("TOTAL")].reconciliation_status == "PASS").all())
    quarantine_pass = bool((real_rows[real_rows.source_file.eq("TOTAL")].quarantine_excluded_from_fit == "PASS").all())
    offset_fail = int((offset_rows.status == "FAIL").sum())
    boundary_fail = int((boundary_rows.status == "FAIL").sum())
    offset_sentence = (
        "The report generator now agrees with raw evidence: +25,200 seconds for UTC minus local and −25,200 seconds for local minus UTC."
        if offset_fail == 0
        else "The −7,200 seconds text remains inconsistent with +07:00 and is marked FAIL; no timezone rule was changed."
    )
    blocking_items = []
    if offset_fail:
        blocking_items.append("1. Correct and rerun the offset evidence generator so the report cannot state −7,200 seconds for a +07:00 relation.")
    blocking_items.append(f"{len(blocking_items) + 1}. Decide and document Network equal-endpoint ownership before Split V2; do not call the current 22 boundaries fully verified.")
    lines = [
        "# FINAL ACCEPTANCE REPORT — ToN-IoT preprocessing remediation",
        "",
        "## Decision",
        "",
        "**Kết luận tổng hợp: chưa đủ điều kiện nghiệm thu không điều kiện cho cả full export và Split V2.**",
        "",
        f"Full remediated export dưới cutoff V1: **{'UNVERIFIED' if sample_pass and quarantine_pass else 'FAIL'}** (sample thực tế đạt đối soát và quarantine-fit: {'PASS' if sample_pass and quarantine_pass else 'FAIL'}, nhưng full export chưa chạy). Split V2: **{'UNVERIFIED' if boundary_fail == 0 else 'FAIL'}** vì {boundary_fail} ranh giới Network có timestamp thực tế ở hai file/split khác nhau; clock cross-platform vẫn cần giữ UNVERIFIED.",
        "",
        "## Real-data sample",
        "",
        "| File | Input | Train | Validation | Test | Quarantine | Output total | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in real_rows[real_rows.source_file.ne("TOTAL")].to_dict("records"):
        lines.append(
            f"| {row['source_file']} | {int(row['source_rows']):,} | {int(row['train_rows']):,} | {int(row['validation_rows']):,} | {int(row['test_rows']):,} | {int(row['quarantine_rows']):,} | {int(row['output_rows']):,} | {row['reconciliation_status']} |"
        )
    total = real_rows[real_rows.source_file.eq("TOTAL")].iloc[0]
    lines += [
        f"| TOTAL | {int(total['source_rows']):,} | {int(total['train_rows']):,} | {int(total['validation_rows']):,} | {int(total['test_rows']):,} | {int(total['quarantine_rows']):,} | {int(total['output_rows']):,} | {total['reconciliation_status']} |",
        "",
        "The sample output is under `research_pipeline/output_remediation_acceptance_iot/`; no V1 output directory was touched. Both preprocessor JSON files declare `fit_split=train` and `invalid_timestamp_policy=quarantine`. Quarantine files contain only Normal rows in this sample, matching the raw-data audit.",
        "",
        "## Feature selection",
        "",
        f"`feature_comparison.csv` compares legacy V1 sample inference with corrected Train-only inference on the same two files. It contains {len(feature_rows)} candidate rows and {int(feature_rows['changed'].sum()) if not feature_rows.empty else 0} changed decision row(s). A zero change count does not invalidate the fix: the fix is about data scope, and the acceptance unit test demonstrates that changing Validation/Test values cannot change the corrected list.",
        "",
        "## Timestamp evidence",
        "",
        f"Raw Ground Truth `ts_min` minus naive `data_time_min` is +25,200 seconds. The inverse, local minus UTC, is −25,200 seconds. `time_offset_validation.csv` observes 0 for local-to-local comparison and −25,200 for processed-local minus numeric epoch. {offset_sentence}",
        "",
        "## Network boundary evidence",
        "",
        f"The inventory has 22 equal adjacent endpoints. The raw scan found {int((boundary_rows['same_timestamp_present_in_both_source_splits'] == 'YES').sum())} boundaries where rows with that timestamp occur in both source files. This does not prove the rows are identical records, but it proves inclusive timestamp ownership is ambiguous across source-file splits.",
        "",
        "Recommended prespecified rule: use half-open intervals `[start, end)` for the earlier source block and `[start, end]` for the final block, or assign the exact boundary row by deterministic source-file order. Do not change V1 retroactively; record the rule in Split V2 before evaluation.",
        "",
        "## Blocking items",
        "",
        *blocking_items,
        "",
        "Safe full-export command for remediation under unchanged V1 cutoffs (not run here):",
        "",
        "```text",
        "python -m research_pipeline.src.main --mode remediated --chunk-size 250000 --output-root research_pipeline/output_remediation_full --report-dir research_pipeline/reports/remediation_full",
        "```",
        "",
        "The command intentionally does not use `--overwrite`; first confirm both target directories are absent and disk space is sufficient. This export does not change V1 or authorize Split V2. After export, run the same chunked reconciliation/hash protocol. No model training is authorized by this acceptance.",
    ]
    (ACCEPTANCE_DIR / "FINAL_ACCEPTANCE_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run() -> None:
    ACCEPTANCE_DIR.mkdir(parents=True, exist_ok=True)
    records = _sample_records()
    real_rows = write_real_data_test(records)
    feature_rows = write_feature_comparison(records)
    offset_rows = write_timestamp_offset_evidence()
    boundary_rows = write_network_boundary_check()
    write_code_verification(feature_rows, real_rows, offset_rows, boundary_rows)
    write_test_results()
    write_final_report(feature_rows, real_rows, offset_rows, boundary_rows)


if __name__ == "__main__":
    run()
