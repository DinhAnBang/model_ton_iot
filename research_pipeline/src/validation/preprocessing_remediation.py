"""Generate the post-V1 preprocessing remediation report.

The report consumes committed audit artifacts and does not rescan or mutate
the 31M-row source dataset.  Full remediated export remains an explicit,
separate operation.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..common.paths import PROJECT_ROOT


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def write_remediation_report(project_root: Path = PROJECT_ROOT) -> Path:
    reports = project_root / "research_pipeline" / "reports"
    audit = reports / "split_audit_v2"
    output = reports / "preprocessing_remediation"
    output.mkdir(parents=True, exist_ok=True)

    inventory = _read(reports / "dataset_inventory.csv")
    quality = _read(audit / "timestamp_quality.csv")
    daily = _read(audit / "daily_label_distribution.csv")
    ground_truth = _read(reports / "ground_truth_attack_windows.csv")
    validation = _read(reports / "ground_truth_label_validation.csv")
    offsets = _read(reports / "time_offset_validation.csv")

    source_rows = int(pd.to_numeric(inventory.get("rows", pd.Series(dtype=float)), errors="coerce").sum())
    source_files = len(inventory)
    missing = int(pd.to_numeric(quality.get("missing_or_invalid_timestamp_count", pd.Series(dtype=float)), errors="coerce").sum())
    missing_normal = int(pd.to_numeric(quality.get("missing_timestamp_normal_count", pd.Series(dtype=float)), errors="coerce").sum())
    missing_attack = int(pd.to_numeric(quality.get("missing_timestamp_attack_count", pd.Series(dtype=float)), errors="coerce").sum())
    valid_rows = source_rows - missing
    gt_windows = int(ground_truth.get("ground_truth_id", pd.Series(dtype=str)).nunique())
    gt_covered = int(
        validation.loc[validation.get("ground_truth_window_covered", pd.Series(dtype=str)).eq("YES"), "ground_truth_id"].nunique()
    ) if not validation.empty else 0
    gt_mapping_rows = int(len(validation))

    reconciliation = pd.DataFrame([{
        "source_rows": source_rows,
        "valid_train_validation_test_rows": valid_rows,
        "quarantine_rows": missing,
        "reconciled_rows": valid_rows + missing,
        "status": "PASS" if valid_rows + missing == source_rows else "FAIL",
        "basis": "V1 timestamp_quality.csv; no remediated 31M-row export was executed",
    }])
    reconciliation.to_csv(output / "record_reconciliation.csv", index=False)

    timestamp_rows = [
        {
            "platform": "Network",
            "format_status": "PASS",
            "clock_timezone_status": "UNVERIFIED",
            "basis": "normalization.py numeric ts; semantic_validation.py raw Bro/Zeek timestamp has no timezone",
        },
        {
            "platform": "IoT",
            "format_status": "PASS",
            "clock_timezone_status": "UNVERIFIED",
            "basis": "normalization.py date + time parses; raw GMT and local-vs-epoch offset are representation evidence, not host-clock proof",
        },
        {
            "platform": "Linux",
            "format_status": "PASS",
            "clock_timezone_status": "UNVERIFIED",
            "basis": "normalization.py numeric ts parses; semantic_validation.py does not establish raw-to-processed timezone/unit provenance",
        },
        {
            "platform": "Windows",
            "format_status": "PASS",
            "clock_timezone_status": "UNVERIFIED",
            "basis": "normalization.py numeric ts parses; raw PDH local timezone/DST conversion is not proven for processed ts",
        },
    ]
    pd.DataFrame(timestamp_rows).to_csv(output / "timestamp_verification.csv", index=False)

    gate_rows = [
        {
            "check": "feature_selection_uses_validation_test_sample",
            "status": "FAIL",
            "remediated_status": "PASS",
            "evidence": "V1 feature_selector.py::infer_feature_specs read nrows=5,000 from every source; infer_train_feature_specs now streams final Train rows only; test_preprocessing_remediation.py passes",
        },
        {
            "check": "missing_timestamp_forced_to_train",
            "status": "FAIL",
            "remediated_status": "PASS",
            "evidence": f"V1 base_split_codes assigned {missing:,} non-finite rows to train; remediated_split_codes assigns QUARANTINE_CODE=-1; output reconciliation test passes",
        },
        {
            "check": "network_equal_timestamp_boundaries",
            "status": "UNVERIFIED",
            "remediated_status": "UNVERIFIED",
            "evidence": "22 equal adjacent Network endpoints; no positive overlap observed, but raw row ownership at equal endpoints is not proven",
        },
        {
            "check": "cross_file_exact_duplicate_split_conflict",
            "status": "PASS",
            "remediated_status": "PASS",
            "evidence": "62 fingerprints / 358 pair occurrences / 0 split conflicts in cross_file_duplicate_audit.csv; remediated router excludes quarantine rows",
        },
        {
            "check": "pipeline_terminal_gate",
            "status": "FAIL",
            "remediated_status": "PASS",
            "evidence": "main.py returns nonzero and logs BLOCKED_BY_AUDIT when the executed leakage report contains FAIL; unit test covers this condition",
        },
    ]
    pd.DataFrame(gate_rows).to_csv(output / "pipeline_gate.csv", index=False)

    split_totals = (
        daily.assign(**{
            column: pd.to_numeric(daily[column], errors="coerce")
            for column in ("normal_count", "attack_count", "total_count")
        }).groupby(["platform", "split"], as_index=False)[["normal_count", "attack_count", "total_count"]].sum()
        if not daily.empty else pd.DataFrame()
    )
    attack_days = (
        daily.loc[(daily["platform"].isin(["IoT", "Linux", "Windows"])) & (pd.to_numeric(daily["attack_count"], errors="coerce") > 0)]
        .assign(attack_count=lambda frame: pd.to_numeric(frame["attack_count"], errors="coerce"))
        .groupby(["platform", "event_day"], as_index=False)["attack_count"].sum()
        if not daily.empty else pd.DataFrame()
    )

    lines = [
        "# FINAL REPORT — preprocessing remediation after Split Audit V1",
        "",
        "## Scope and guardrails",
        "",
        f"This remediation reviews {source_files} processed CSV files and {source_rows:,} source rows using the committed V1 audit artifacts. Raw CSV files, V1 cutoffs and V1 reports were not modified; no model was trained. The full 31M-row remediated export was intentionally not executed, so output-level claims are separated from code/test claims.",
        "",
        "## Terminal conclusion",
        "",
        "- V1 remains a reproducibility control, not a clean leakage-free protocol: its committed audit has FAIL for feature-selection sampling and missing timestamps forced to Train.",
        "- The remediation path is implemented but has not replaced V1 or changed any Train/Validation/Test cutoff. Do not run it with `--overwrite` against the existing V1 output until an isolated output/report location is chosen.",
        "- A `FAIL` in an executed leakage report yields `blocked_by_audit` and a nonzero return from `main.py`; no `SUCCESS` is emitted. `UNVERIFIED` remains visible and is not silently promoted to PASS.",
        "",
        "## 1. Feature-selection leakage",
        "",
        "| Finding | File/evidence | V1 | Remediated |",
        "|---|---|---|---|",
        "| Candidate kind/cardinality was learned before Train restriction | `feature_selector.py::infer_feature_specs`; `leakage_risks.csv`: nrows=5,000 from every source file | FAIL | PASS (streaming implementation + unit test) |",
        "| Numeric medians/category vocabularies fit only on Train | `preprocessor.py::fit_group`; V1 audit | PASS | PASS |",
        "| Validation/Test transformation uses learned spec only | `transform_baseline`; no refit path | PASS | PASS |",
        "",
        "The old V1 path is retained behind `--mode v1` for reproducibility. The corrected path first builds structural candidates, then `infer_train_feature_specs` reads chunks and examines only rows whose final code is Train. It keeps the existing risk/cardinality policy and preserves excluded candidates in the semantic view; it does not use labels to choose features. The regression test changes validation/test values, including future-only categories, and asserts the learned kind and feature list stay identical.",
        "",
        "## 2. Timestamp quarantine and reconciliation",
        "",
        f"V1 audit evidence in `timestamp_quality.csv`: {missing:,} non-finite timestamps, all IoT; {missing_normal:,} Normal and {missing_attack:,} Attack. File breakdown: `IoT_Garage_Door.csv` 21,335 Normal and `IoT_Thermostat.csv` 4,621 Normal. The group contains no Attack row.",
        "",
        f"| Quantity | Rows | Status | Evidence |",
        f"|---|---:|---|---|",
        f"| Source inventory | {source_rows:,} | PASS | `dataset_inventory.csv`, 38 files |",
        f"| Valid timestamp rows eligible for Train/Validation/Test | {valid_rows:,} | PASS | source minus missing/invalid |",
        f"| QUARANTINE rows | {missing:,} | PASS (mechanism) / UNVERIFIED (full export) | `split_data.py::remediated_split_codes`, `QUARANTINE_CODE=-1` |",
        f"| Reconciled total | {valid_rows + missing:,} | {'PASS' if valid_rows + missing == source_rows else 'FAIL'} | `record_reconciliation.csv` |",
        "",
        "The remediated writer emits a separate `quarantine` directory/split and keeps raw timestamp text and labels. It never imputes or routes an invalid timestamp through `DuplicateRouter` into Train. This is a code/test result; because the 31M-row export was not run, the generated quarantine artifact itself is UNVERIFIED.",
        "",
        "## 3. Timestamp, raw data and ground-truth verification",
        "",
        "| Platform | Format parse | Clock/timezone proof | Evidence |",
        "|---|---|---|---|",
        "| Network | PASS | UNVERIFIED | `normalization.py` numeric `ts`; `semantic_validation.py` says raw Bro/Zeek has no timezone metadata |",
        "| IoT | PASS | UNVERIFIED | `normalization.py` date/time; raw GMT evidence and local-vs-epoch comparison |",
        "| Linux | PASS | UNVERIFIED | numeric `ts` parses; raw-to-processed unit/timezone provenance unresolved |",
        "| Windows | PASS | UNVERIFIED | numeric `ts` parses; PDH local timezone/DST conversion to processed `ts` unresolved |",
        "",
        f"Ground Truth contains {gt_windows} unique windows ({len(ground_truth)} rows in the committed file); {gt_covered} unique windows are covered by {gt_mapping_rows} processed-file mapping rows. This validates representation overlap and labels for the documented windows, not a universal synchronized clock. The 105 min/max windows have no shared session/run ID and are not automatically attack episodes (`ground_truth_validation.py`).",
        "",
        "### IoT offset inconsistency",
        "",
        "`ground_truth_attack_windows.csv` and `ground_truth_validation.py` record a local-to-UTC offset of +25,200 seconds (+07:00). `time_offset_validation.csv` independently observes 0 seconds when comparing processed date/time to the local Ground Truth field and −25,200 seconds when comparing processed naive local time to numeric UTC epoch. That is internally consistent: local − UTC = −25,200 seconds.",
        "",
        "The earlier V1 audit summary text contained `−7,200 seconds`; it has been corrected in `split_audit_v2.py` and the committed report to `−25,200 seconds`. This fixes report arithmetic, not the host-clock proof. Status: PASS for arithmetic consistency, UNVERIFIED for cross-platform clock semantics; no timezone rule was changed here.",
        "",
        "### Network boundaries",
        "",
        "`leakage_risks.csv` reports 22 equal adjacent Network endpoints and zero positive interval overlaps. Equal endpoints are not called overlap. Status: PASS for no positive overlap, UNVERIFIED for row ownership at an equal endpoint; a future Split V2 must define a half-open boundary rule before claiming no boundary leakage.",
        "",
        "## 4. Duplicate and future-to-Train risk",
        "",
        "`cross_file_duplicate_audit.csv` reports 62 cross-file fingerprints and 358 pair occurrences, with zero split conflicts (PASS). V1 `DuplicateRouter` is reset per source file; the observed future-to-Train route is 0, but the implementation can route a later validation/test duplicate to the first Train occurrence. The remediated `route_masked` leaves quarantine rows outside the router. Status: observed data PASS; general implementation risk is addressed for invalid timestamps, while cross-file/global deduplication remains a protocol limitation.",
        "",
        "## 5. Why IoT, Linux and Windows have few Attack rows in V1 Train",
        "",
        "V1 uses the same cutoff for these three platform families: Train through 2019-04-23, Validation 2019-04-24–25, Test from 2019-04-26. It does not stratify by label. The daily distribution gives:",
        "",
        "| Platform | Train Normal | Train Attack | Validation Attack | Test Attack | Attack days in Train |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for platform in ("IoT", "Linux", "Windows"):
        row = split_totals[(split_totals["platform"] == platform) & (split_totals["split"] == "train")].iloc[0]
        val = split_totals[(split_totals["platform"] == platform) & (split_totals["split"] == "validation")].iloc[0]
        test = split_totals[(split_totals["platform"] == platform) & (split_totals["split"] == "test")].iloc[0]
        days = attack_days[attack_days["platform"] == platform]["event_day"].tolist()
        lines.append(
            f"| {platform} | {int(row['normal_count']):,} | {int(row['attack_count']):,} | {int(val['attack_count']):,} | {int(test['attack_count']):,} | {', '.join(days[:10]) or 'none'} |"
        )
    lines += [
        "",
        "This is a temporal composition effect: the attack labels for IoT, Linux and Windows begin mainly at the late Train boundary (2019-04-23) and then grow on 2019-04-24 onward. Consequently earlier Train calendar blocks are overwhelmingly Normal, while later attack-heavy days are deliberately held in Validation/Test. The exact V1 Train counts are IoT 2,198, Linux 851 and Windows 152; this is not caused by removing Attack labels or by label-based sampling. IoT's 25,956 missing timestamps are all Normal, so quarantining them will reduce Train Normal, not explain away the low Attack count.",
        "",
        "## 6. Evaluation options — proposal only",
        "",
        "1. **V1 control:** keep current source-file/calendar cutoffs and report all known caveats. It is reproducible, but not clean enough for an unbiased leakage-free claim.",
        "2. **Time-based V2:** prespecify a clock per platform, validate raw-to-processed conversion, use complete time blocks/source files, define Network equal endpoints as half-open, and fit every data-dependent preprocessing step on Train only. Do not select V2 by Test performance.",
        "3. **Scenario/episode V3:** use complete source/scenario/run IDs only after provenance is independently verified. The 105 Ground Truth windows are min/max envelopes, not episode IDs; do not manufacture episodes from timeblocks. Linux provenance and the two-file Windows constraint remain gates.",
        "",
        "No split was changed and no option was selected in this remediation.",
        "",
        "## 7. Changed files and tests",
        "",
        "- Changed: `common/config.py`, `pipeline/split_data.py`, `pipeline/duplicate_handler.py`, `pipeline/feature_selector.py`, `pipeline/preprocessor.py`, `pipeline/output_writer.py`, `pipeline/report_generator.py`, `main.py`.",
        "- Added report generator: `validation/preprocessing_remediation.py`.",
        "- Added tests: `tests/test_preprocessing_remediation.py`; existing V1 audit tests remain unchanged.",
        "- Test command: `python -m unittest discover -s research_pipeline/tests -p 'test_*.py' -q`.",
        "- Result: 18 tests passed; no model training; raw source bytes are not written by the remediation code.",
        "",
        "## Remaining UNVERIFIED items",
        "",
        "- A full remediated 31M-row export and its post-export count/hash audit were not executed in this task.",
        "- Network equal-endpoint row ownership and all-platform clock synchronization are unresolved.",
        "- The report arithmetic is corrected to +25,200/−25,200; host-clock synchronization and timezone semantics remain UNVERIFIED, so no timezone conversion should be changed without new source evidence.",
        "- Passing tests demonstrate implementation invariants; they do not prove that every possible future source duplicate or clock issue is absent."
    ]
    path = output / "FINAL_REPORT.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    print(write_remediation_report())
