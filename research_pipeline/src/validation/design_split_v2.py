"""Generate an evidence-only design for the two proposed Split V2 protocols.

This module never assigns a new split to an exported dataset.  It reads the
committed V1 reports and only reads the small set of Network rows located at
the 22 equal endpoints.  Counts labelled ``Protocol A`` are prospective
counts obtained from V1 counts plus deterministic endpoint ownership and the
known IoT quarantine correction; they are not a Split V2 export.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from ..common.config import canonical_type, raw_type_mapping
from ..common.paths import PROJECT_ROOT, REPORT_DIR
from ..pipeline.data_reader import make_source_records
from ..pipeline.normalization import parse_time_bound
from ..pipeline.split_data import assign_split_plan


DESIGN_DIR = REPORT_DIR / "split_v2_design"
AUDIT_DIR = REPORT_DIR / "split_audit_v2"
CHUNK_SIZE = 250_000
CONNECTION_COLUMNS = ("src_ip", "src_port", "dst_ip", "dst_port", "proto", "service")
PLATFORMS = ("Network", "IoT", "Linux", "Windows")
SPLITS = ("train", "validation", "test")


def _network_records():
    records = make_source_records(PROJECT_ROOT)
    assign_split_plan(records)
    return sorted(
        [record for record in records if record.platform == "Network"],
        key=lambda record: (parse_time_bound(record.timestamp_min, "Network"), record.file_name),
    )


def _equal_boundaries(records):
    return [
        (left, right, parse_time_bound(left.timestamp_max, "Network"))
        for left, right in zip(records, records[1:])
        if parse_time_bound(left.timestamp_max, "Network")
        == parse_time_bound(right.timestamp_min, "Network")
    ]


def _read_target_rows(path: Path, targets: set[float]) -> dict[float, pd.DataFrame]:
    collected: dict[float, list[pd.DataFrame]] = defaultdict(list)
    if not targets:
        return {}
    for frame in pd.read_csv(path, chunksize=CHUNK_SIZE, low_memory=False):
        values = pd.to_numeric(frame["ts"], errors="coerce").to_numpy(dtype="float64")
        for target in targets:
            mask = np.isclose(values, target, rtol=0.0, atol=1e-9)
            if mask.any():
                collected[target].append(frame.loc[mask].copy())
    return {target: pd.concat(frames, ignore_index=True) for target, frames in collected.items()}


def _value_keys(frame: pd.DataFrame, columns: list[str]) -> set[tuple[str, ...]]:
    if frame.empty:
        return set()
    normalized = frame.loc[:, columns].astype("string").fillna("<NA>")
    return set(map(tuple, normalized.itertuples(index=False, name=None)))


def build_network_boundary_details() -> tuple[pd.DataFrame, dict[tuple[str, float], pd.DataFrame]]:
    records = _network_records()
    boundaries = _equal_boundaries(records)
    targets_by_file: dict[str, set[float]] = {record.file_name: set() for record in records}
    for left, right, target in boundaries:
        targets_by_file[left.file_name].add(target)
        targets_by_file[right.file_name].add(target)

    target_rows: dict[tuple[str, float], pd.DataFrame] = {}
    by_name = {record.file_name: record for record in records}
    for file_name, targets in targets_by_file.items():
        for target, frame in _read_target_rows(by_name[file_name].file_path, targets).items():
            target_rows[(file_name, target)] = frame

    rows = []
    for index, (left, right, target) in enumerate(boundaries, start=1):
        left_frame = target_rows.get((left.file_name, target), pd.DataFrame())
        right_frame = target_rows.get((right.file_name, target), pd.DataFrame())
        common_columns = [column for column in left_frame.columns if column in right_frame.columns]
        left_keys = _value_keys(left_frame, common_columns)
        right_keys = _value_keys(right_frame, common_columns)
        exact_intersection = left_keys & right_keys
        key_columns = [column for column in CONNECTION_COLUMNS if column in common_columns]
        connection_intersection = _value_keys(left_frame, key_columns) & _value_keys(right_frame, key_columns)
        cross_split = left.fixed_split != right.fixed_split
        if cross_split:
            status = "FAIL"
            detail = "Equal timestamp is present in source files assigned to different V1 splits; V1 endpoint ownership is ambiguous."
        elif exact_intersection:
            status = "PASS"
            detail = "Exact shared rows are confined to one V1 split; prospective V2 still needs one endpoint rule."
        else:
            status = "PASS"
            detail = "Equal timestamp rows are confined to one V1 split; no exact shared row was found across the two files."
        if cross_split and exact_intersection:
            detail += " Exact duplicate intersection is also present across the V1 split boundary."
        elif cross_split and connection_intersection:
            detail += " A shared connection key is present across the V1 split boundary; this is not a session identifier."
        rows.append({
            "boundary_index": index,
            "left_file": left.file_name,
            "right_file": right.file_name,
            "boundary_epoch": target,
            "boundary_utc": pd.to_datetime(target, unit="s", utc=True).isoformat(),
            "left_split_v1": left.fixed_split,
            "right_split_v1": right.fixed_split,
            "left_rows_at_boundary": len(left_frame),
            "right_rows_at_boundary": len(right_frame),
            "rows_at_boundary_total": len(left_frame) + len(right_frame),
            "exact_duplicate_intersection_count": len(exact_intersection),
            "connection_key_intersection_count": len(connection_intersection),
            "connection_key_definition": ",".join(key_columns),
            "session_id_available": "NO",
            "same_v1_split": "YES" if not cross_split else "NO",
            "status": status,
            "detail": detail,
        })
    result = pd.DataFrame(rows)
    result.to_csv(DESIGN_DIR / "network_boundary_details.csv", index=False)
    return result, target_rows


def _base_counts() -> pd.DataFrame:
    daily = pd.read_csv(AUDIT_DIR / "daily_label_distribution.csv")
    return (
        daily.groupby(["platform", "split"], as_index=False)[["normal_count", "attack_count", "total_count"]]
        .sum()
        .sort_values(["platform", "split"])
    )


def _type_counts() -> pd.DataFrame:
    return pd.read_csv(AUDIT_DIR / "attack_type_coverage.csv")


def _adjust_for_quarantine(counts: pd.DataFrame) -> pd.DataFrame:
    result = counts.copy()
    mask = result.platform.eq("IoT") & result.split.eq("train")
    result.loc[mask, "normal_count"] -= 25_956
    result.loc[mask, "total_count"] -= 25_956
    result["quarantine_count"] = 0
    result.loc[result.platform.eq("IoT") & result.split.eq("train"), "quarantine_count"] = 25_956
    return result


def _apply_half_open_network(counts: pd.DataFrame, types: pd.DataFrame, boundary_rows) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = counts.copy()
    type_result = types.copy()
    mapping = raw_type_mapping()
    # Ownership rule: every non-final source interval is [start, end); the
    # exact endpoint is assigned to the following interval.  Rows physically
    # stored in the left file at that endpoint therefore move to right.fixed_split.
    network = _network_records()
    for left, right, target in _equal_boundaries(network):
        if left.fixed_split == right.fixed_split:
            continue
        frame = boundary_rows.get((left.file_name, target), pd.DataFrame())
        if frame.empty:
            continue
        for split, sign in ((left.fixed_split, -1), (right.fixed_split, 1)):
            selector = result.platform.eq("Network") & result.split.eq(split)
            result.loc[selector, "normal_count"] += sign * int((pd.to_numeric(frame["label"], errors="coerce") == 0).sum())
            result.loc[selector, "attack_count"] += sign * int((pd.to_numeric(frame["label"], errors="coerce") == 1).sum())
            result.loc[selector, "total_count"] += sign * len(frame)
        attack_frame = frame.loc[pd.to_numeric(frame["label"], errors="coerce") == 1]
        for value in attack_frame.get("type", pd.Series(dtype="string")).astype(str):
            name = mapping.get(value, canonical_type(value) or value.strip().lower())
            selector = type_result.platform.eq("Network") & type_result.attack_type.eq(name)
            type_result.loc[selector & type_result.train_attack_count.ge(0), "train_attack_count"] += 0
            type_result.loc[selector, f"{left.fixed_split}_attack_count"] -= 1
            type_result.loc[selector, f"{right.fixed_split}_attack_count"] += 1
    for column in ("train_attack_count", "validation_attack_count", "test_attack_count"):
        type_result[column] = type_result[column].astype("int64")
    type_result["total_attack_count"] = type_result[["train_attack_count", "validation_attack_count", "test_attack_count"]].sum(axis=1)
    for split in SPLITS:
        type_result[f"{split}_seen"] = np.where(type_result[f"{split}_attack_count"] > 0, "YES", "NO")
    train_seen = type_result.train_seen.eq("YES")
    test_seen = type_result.test_seen.eq("YES")
    validation_seen = type_result.validation_seen.eq("YES")
    type_result["seen_unseen_from_train"] = np.where(train_seen, "SEEN", "UNSEEN")
    type_result["coverage_status"] = np.select(
        [type_result.total_attack_count.eq(0), train_seen & test_seen, (~train_seen) & test_seen & validation_seen, (~train_seen) & test_seen],
        ["ABSENT_IN_PLATFORM", "SEEN_IN_TRAIN", "UNSEEN_IN_TRAIN_VALIDATION_AND_TEST", "UNSEEN_IN_TRAIN_TEST_ONLY"],
        default="UNSEEN_IN_TRAIN_VALIDATION_ONLY",
    )
    result["quarantine_count"] = result.get("quarantine_count", 0)
    return result, type_result


def _type_lists(types: pd.DataFrame, platform: str, split: str) -> list[str]:
    rows = types[(types.platform == platform) & (types[f"{split}_attack_count"] > 0)]
    return sorted(rows.attack_type.tolist())


def _stats_row(counts: pd.DataFrame, types: pd.DataFrame, platform: str, split: str) -> dict[str, object]:
    row = counts[(counts.platform == platform) & (counts.split == split)].iloc[0]
    normal = int(row.normal_count)
    attack = int(row.attack_count)
    return {
        f"{split}_rows": int(row.total_count),
        f"{split}_normal": normal,
        f"{split}_attack": attack,
        f"{split}_attack_rate": attack / (normal + attack) if normal + attack else 0.0,
        f"{split}_attack_types": ";".join(_type_lists(types, platform, split)),
    }


def _candidate_rows(counts: pd.DataFrame, types: pd.DataFrame) -> pd.DataFrame:
    rows = []
    candidates = [
        ("V1_CONTROL", "control", "current V1 source-file/calendar assignment", "PASS"),
        ("A_TIME_HALF_OPEN", "known_attack_temporal", "V1 assignment plus right-owned half-open Network endpoints; no export", "UNVERIFIED"),
        ("B_NATURAL_UNSEEN_AUDIT", "unseen_attack", "observed unseen-type proxy; no scenario/episode reallocation", "UNVERIFIED"),
    ]
    for candidate_id, protocol, basis, status in candidates:
        for platform in PLATFORMS:
            platform_counts = counts
            if candidate_id == "V1_CONTROL":
                # V1 counts include the 25,956 invalid IoT rows in Train by definition.
                platform_counts = _base_counts()
                platform_types = pd.read_csv(AUDIT_DIR / "attack_type_coverage.csv")
                quarantine = 0
            else:
                platform_types = types
                quarantine = 25_956 if platform == "IoT" else 0
            record = {
                "candidate_id": candidate_id,
                "protocol": protocol,
                "platform": platform,
                "assignment_unit": "source_file_holdout" if candidate_id == "V1_CONTROL" and platform == "Network" else ("source_file+endpoint_rule" if platform == "Network" else "calendar_block"),
                "selection_basis": basis,
                "expected_counts_kind": "observed V1" if candidate_id == "V1_CONTROL" else "prospective/proxy; not exported",
                "quarantine_rows": quarantine,
                "status": status,
            }
            for split in SPLITS:
                record.update(_stats_row(platform_counts, platform_types, platform, split))
            train_types = set(_type_lists(platform_types, platform, "train"))
            test_types = set(_type_lists(platform_types, platform, "test"))
            record["test_attack_types_seen_in_train"] = ";".join(sorted(test_types & train_types))
            record["test_attack_types_unseen_in_train"] = ";".join(sorted(test_types - train_types))
            record["known_attack_test_count"] = int(platform_types.loc[(platform_types.platform == platform) & platform_types.test_seen.eq("YES") & platform_types.train_seen.eq("YES"), "test_attack_count"].sum())
            record["unseen_attack_test_count"] = int(platform_types.loc[(platform_types.platform == platform) & platform_types.test_seen.eq("YES") & platform_types.train_seen.eq("NO"), "test_attack_count"].sum())
            record["attack_rate_shift_test_minus_train"] = record["test_attack_rate"] - record["train_attack_rate"]
            rows.append(record)
    return pd.DataFrame(rows)


def _coverage_comparison(candidates: pd.DataFrame, types: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in candidates.iterrows():
        platform = row.platform
        train_types = set(filter(None, str(row.train_attack_types).split(";")))
        test_types = set(filter(None, str(row.test_attack_types).split(";")))
        known = test_types & train_types
        unseen = test_types - train_types
        attack_total_train = int(row.train_attack)
        attack_total_test = int(row.test_attack)
        train_share = {name: 0.0 for name in train_types | test_types}
        test_share = {name: 0.0 for name in train_types | test_types}
        platform_types = types[types.platform.eq(platform)]
        for _, type_row in platform_types.iterrows():
            name = type_row.attack_type
            if attack_total_train:
                train_share[name] = int(type_row.train_attack_count) / attack_total_train
            if attack_total_test:
                test_share[name] = int(type_row.test_attack_count) / attack_total_test
        tvd = 0.5 * sum(abs(train_share[name] - test_share[name]) for name in train_share)
        rows.append({
            "candidate_id": row.candidate_id,
            "protocol": row.protocol,
            "platform": platform,
            "train_attack_types": row.train_attack_types,
            "test_attack_types": row.test_attack_types,
            "known_attack_types_in_test": ";".join(sorted(known)),
            "unseen_attack_types_in_test": ";".join(sorted(unseen)),
            "known_attack_test_count": int(row.known_attack_test_count),
            "unseen_attack_test_count": int(row.unseen_attack_test_count),
            "train_attack_rate": row.train_attack_rate,
            "test_attack_rate": row.test_attack_rate,
            "test_minus_train_attack_rate": row.attack_rate_shift_test_minus_train,
            "attack_type_total_variation_train_test": tvd,
            "coverage_status": "PASS" if known and int(row.test_attack) > 0 else "FAIL",
            "status": row.status,
        })
    result = pd.DataFrame(rows)
    result.to_csv(DESIGN_DIR / "class_coverage_comparison.csv", index=False)
    return result


def _bound_text(boundary: pd.DataFrame) -> str:
    fail = boundary[boundary.status.eq("FAIL")]
    return ", ".join(f"{row.left_file}/{row.right_file} ({row.boundary_utc}, left={row.left_rows_at_boundary}, right={row.right_rows_at_boundary}, exact={row.exact_duplicate_intersection_count}, key={row.connection_key_intersection_count})" for _, row in fail.iterrows())


def write_protocol_reports(boundary: pd.DataFrame, candidates: pd.DataFrame, coverage: pd.DataFrame) -> None:
    a = candidates[candidates.candidate_id.eq("A_TIME_HALF_OPEN")]
    b = candidates[candidates.candidate_id.eq("B_NATURAL_UNSEEN_AUDIT")]
    a_lines = [
        "# Protocol A — Known-Attack Temporal Evaluation",
        "",
        "## Protocol definition",
        "",
        "Mục tiêu là huấn luyện trên attack type đã xuất hiện trong Train và đánh giá trên dữ liệu có timestamp tương lai. Đây là giao thức theo thời gian; timeblock không được gọi là attack episode.",
        "",
        "Quy tắc Network được đề xuất trước khi nhìn kết quả model: với các khoảng nguồn đã sắp thứ tự, dùng `[start_i, end_i)` cho mọi khoảng không cuối và `[start_last, end_last]` cho khoảng cuối. Timestamp đúng endpoint thuộc khoảng bên phải. V1 không bị sửa.",
        "",
        "## Feasibility from current V1 evidence",
        "",
        "| Platform | Train rows | Validation rows | Test rows | Train attack types | Test attack types | Known-type attack rows in Test | Unseen-type attack rows in Test | Status |",
        "|---|---:|---:|---:|---|---|---:|---:|---|",
    ]
    for _, row in a.iterrows():
        a_lines.append(f"| {row.platform} | {int(row.train_rows):,} | {int(row.validation_rows):,} | {int(row.test_rows):,} | {row.train_attack_types or 'none'} | {row.test_attack_types or 'none'} | {int(row.known_attack_test_count):,} | {int(row.unseen_attack_test_count):,} | {row.status} |")
    a_lines += [
        "",
        "Coverage conclusion: Network has no known attack type in Test (known-type Test attack rows = 0), and Windows likewise has no known attack type in Test. IoT and Linux have only `scanning` as a known Test attack type (1,775 and 75,020 rows respectively). Therefore this protocol is not a valid single global known-attack claim for all four platforms; it is usable only as platform-specific evaluation, subject to clock verification.",
        "",
        "The candidate counts are not selected for a desirable label ratio. They are the committed V1 counts, subtracting the 25,956 invalid IoT timestamps from Train and moving only the left-file rows at the two cross-split Network endpoints to the right owner. No Test metric or model threshold was used.",
        "",
        "Evidence: `daily_label_distribution.csv`, `attack_type_coverage.csv`, `network_boundary_details.csv`, and `candidate_splits.csv`. Timestamp syntax is parseable, but Network/Linux/Windows clock provenance is not fully synchronized; status remains UNVERIFIED until that evidence exists.",
        "",
        "## Required gates before implementation",
        "",
        "1. Freeze the endpoint rule and implement it in a new V2 module without changing V1.",
        "2. Validate the raw-to-processed clock basis per platform and keep unverified rows out of a temporal claim.",
        "3. Fit feature specification, vocabulary, medians, and all thresholds on Train only; open Test once for the final estimate.",
        "4. Report Network and Windows as coverage failures for known-attack Test assessment, not as successful known-attack benchmarks.",
    ]
    (DESIGN_DIR / "known_attack_protocol.md").write_text("\n".join(a_lines) + "\n", encoding="utf-8")

    b_lines = [
        "# Protocol B — Unseen-Attack Evaluation",
        "",
        "## Protocol definition",
        "",
        "Mục tiêu là giữ attack type/kịch bản của Test không xuất hiện trong Train. Đây không được gọi là temporal generalization: hiện chưa có định danh episode/session dùng chung và chưa chứng minh thứ tự thời gian cho mọi nền tảng.",
        "",
        "## Observed unseen-type proxy",
        "",
        "| Platform | Train attack types | Test unseen types | Unseen attack rows in Test | Status |",
        "|---|---|---|---:|---|",
    ]
    for _, row in b.iterrows():
        b_lines.append(f"| {row.platform} | {row.train_attack_types or 'none'} | {row.test_attack_types_unseen_in_train or 'none'} | {int(row.unseen_attack_test_count):,} | {row.status} |")
    b_lines += [
        "",
        "The natural V1 composition already contains many unseen types: Network `backdoor, mitm, password, ransomware, xss`; IoT `backdoor, ddos, password, ransomware, xss`; Linux `ddos, dos, injection, mitm, password, xss`; Windows `backdoor, ddos, injection, mitm, password, ransomware, xss` in Test. This is evidence of an available audit signal, not evidence of a clean deliberate holdout.",
        "",
        "## Required construction for a publishable B protocol",
        "",
        "- Pre-register the held-out type/scenario set from source provenance, not from Test predictions.",
        "- Keep all rows of a verified scenario/episode in exactly one fold. Do not manufacture episodes from the 105 Ground Truth min/max windows; they are envelopes without a shared session/run ID.",
        "- Build a global fingerprint index across files before fold assignment. The current audit found 62 cross-file fingerprints / 358 pair occurrences and no observed V1 split conflict, but the current router is reset per source file, so that is not a general global guarantee.",
        "- Use the connection tuple `src_ip,src_port,dst_ip,dst_port,proto,service` only as a conservative related-flow proxy. There is no session ID in the Network schema; same key is not proof of same session.",
        "- Refit no feature selector or preprocessor on Validation/Test. Report any type absent from Train as a coverage limitation rather than relabelling it.",
        "",
        "Current status is UNVERIFIED for all platforms: the rows support an exploratory unseen-type analysis, but scenario/session isolation and a deliberate allocation have not been executed or independently proven. B must not be reported as temporal generalization.",
    ]
    (DESIGN_DIR / "unseen_attack_protocol.md").write_text("\n".join(b_lines) + "\n", encoding="utf-8")


def write_leakage_assessment(boundary: pd.DataFrame) -> None:
    risks = pd.read_csv(AUDIT_DIR / "leakage_risks.csv")
    cross = pd.read_csv(AUDIT_DIR / "cross_file_duplicate_audit.csv")
    rows = boundary[boundary.status.eq("FAIL")]
    text = f"""# Leakage assessment — Split V2 design

## Evidence matrix

| Risk | Evidence | Status | Interpretation |
|---|---|---|---|
| Network positive interval overlap | `leakage_risks.csv`: 0 positive overlaps | PASS | Equal endpoints are not positive overlap. |
| Network equal endpoint ownership | `network_boundary_details.csv`: 22 equal endpoints; {len(rows)} cross-V1-split boundaries | FAIL | V1 assigns the two endpoint pairs to different folds; inclusive ownership is ambiguous. |
| Exact duplicate at the two cross-split endpoints | `network_boundary_details.csv`: counts in the exact intersection column | {('FAIL' if int(rows.exact_duplicate_intersection_count.sum()) > 0 else 'PASS')} | Full shared-column row comparison was performed only on timestamp-matching boundary rows. |
| Same connection key at the two cross-split endpoints | `network_boundary_details.csv`: counts in the connection intersection column | UNVERIFIED | The tuple is a related-flow proxy, not a session ID. |
| Cross-file duplicate split conflict | `cross_file_duplicate_audit.csv`: 62 fingerprints, 358 pair occurrences, 0 conflicts | PASS (observed V1) | The report is data evidence, not a proof for future data; current routing resets per source file. |
| Future duplicate routed to Train | `leakage_risks.csv`: observed 0 | PASS (observed) / UNVERIFIED (general) | A zero observation does not establish a global duplicate policy. |
| Feature selection scope | corrected `feature_selector.py` and remediation tests | PASS (code invariant) | Candidate decisions are learned from final Train rows; Validation/Test are transform-only. |
| Timestamp clock synchronization | remediation reports: syntax PASS, cross-platform clock proof UNVERIFIED | UNVERIFIED | A parseable epoch is not proof of the physical clock or timezone. |
| Ground Truth windows as episodes | `ground_truth_validation.py`; 105 windows, no shared session/run ID | UNVERIFIED | Windows are min/max envelopes and cannot define episodes by themselves. |

## Deterministic endpoint rule

For the new protocol only, define Network blocks as half-open `[start_i, end_i)` and assign the exact endpoint to the following block; the final block is closed at the right edge. This gives one owner for each timestamp. The rule must be applied before duplicate/session checks and frozen in the protocol manifest. It does not rewrite V1 outputs.

## Remaining limitations

The two endpoint pairs are `Network_dataset_16/17` at `2019-04-26T02:30:34Z` and `Network_dataset_19/20` at `2019-04-27T01:25:17Z`. The new table records the exact row counts, exact shared rows, and connection-key intersections. Because the source has no session identifier, same-key results cannot establish episode leakage. A full global duplicate/related-flow index is still required before Protocol B is publishable.
"""
    (DESIGN_DIR / "leakage_assessment.md").write_text(text, encoding="utf-8")


def write_final_report(boundary: pd.DataFrame, candidates: pd.DataFrame, coverage: pd.DataFrame) -> None:
    fail = boundary[boundary.status.eq("FAIL")]
    a = coverage[coverage.candidate_id.eq("A_TIME_HALF_OPEN")]
    b = coverage[coverage.candidate_id.eq("B_NATURAL_UNSEEN_AUDIT")]
    lines = [
        "# FINAL SPLIT V2 DESIGN — ToN-IoT",
        "",
        "## Scope and decision",
        "",
        "Đây là thiết kế/audit evidence-only. Không có Split V1 nào bị thay đổi, không có full export, không train model và không chỉnh sửa raw data.",
        "",
        "**Overall status: UNVERIFIED — chưa được phép triển khai Split V2.** Lý do chặn là hai ranh giới Network có timestamp thực tế ở cả hai file nhưng thuộc hai V1 split khác nhau; thêm vào đó clock provenance của Linux/Windows/Network chưa đủ để tuyên bố temporal generalization.",
        "",
        "## A. Network boundary result",
        "",
        f"Có đúng {len(boundary)} ranh giới endpoint bằng nhau. Hai ranh giới thực sự cắt qua split là:",
        "",
        *[f"- `{row.left_file}` / `{row.right_file}`, `{row.boundary_utc}`, left={int(row.left_rows_at_boundary):,}, right={int(row.right_rows_at_boundary):,}, exact intersection={int(row.exact_duplicate_intersection_count):,}, connection-key intersection={int(row.connection_key_intersection_count):,}." for _, row in fail.iterrows()],
        "",
        "`network_boundary_details.csv` là bảng đầy đủ 22 ranh giới. Schema Network không có session ID; khóa `src_ip,src_port,dst_ip,dst_port,proto,service` chỉ là proxy cho related connection. Vì vậy hiện chưa được phép kết luận cùng phiên nếu chỉ có shared key.",
        "",
        "Quy tắc V2 được đề xuất: `[start, end)` cho interval trước và `[start_last, end_last]` cho interval cuối; timestamp đúng endpoint thuộc interval sau. Quy tắc này bảo đảm một timestamp chỉ có một owner trong giao thức mới, nhưng chưa được triển khai và không được áp dụng ngược vào V1.",
        "",
        "**Boundary readiness: FAIL for V1-as-is; PASS as a proposed deterministic rule; overall UNVERIFIED until implemented and re-audited.**",
        "",
        "## B. Protocol comparison",
        "",
        "| Protocol | Research question | Empirical support | Leakage control | Timestamp requirement | Status |",
        "|---|---|---|---|---|---|",
        "| A Known-Attack Temporal | Known types in Train, future data in Test | IoT/Linux have only scanning known in Test; Network/Windows have zero known-type Test attack rows | Train-only preprocessing; endpoint rule still to implement | Per-platform clock proof required | UNVERIFIED / coverage FAIL on Network+Windows |",
        "| B Unseen-Attack | Held-out type/scenario absent from Train | Natural V1 unseen types exist on all platforms | Needs global duplicate and verified scenario/session isolation | Must not claim temporal ordering | UNVERIFIED |",
        "",
        "The candidate counts and attack-rate/type-shift metrics are in `candidate_splits.csv` and `class_coverage_comparison.csv`. They are estimates/proxies for the proposed protocol, not an exported split.",
        "",
        "## C. Platform feasibility",
        "",
        "- **Network:** known-attack Test evaluation is not supported by current coverage (0 known-type attack rows in Test). It is useful for unseen-type analysis only after endpoint/dedup controls.",
        "- **IoT:** has 2,198 Train attack rows and 1,775 known-type (`scanning`) Test attack rows; the 25,956 invalid IoT timestamps are excluded to quarantine in the prospective counts. Known-attack coverage is limited and time clock remains a protocol caveat.",
        "- **Linux:** has 851 Train attack rows and 75,020 known-type (`scanning`) Test attack rows. The data volume is sufficient for a platform-specific known-type estimate, but source/scenario and clock provenance remain limitations.",
        "- **Windows:** has 152 Train attack rows and 0 known-type attack rows in Test; it cannot support a known-attack Test claim under the current V1 composition. Two source files also constrain scenario holdout.",
        "",
        "## D. Timestamp and episode limits",
        "",
        "Format parsing is PASS in the remediation reports. IoT raw Ground Truth arithmetic is internally consistent: UTC epoch minus naive local time = +25,200 seconds; the inverse local minus UTC = −25,200 seconds. This does not prove cross-platform synchronization. Network/Linux/Windows physical clock/timezone provenance remains UNVERIFIED. The 105 Ground Truth windows are min/max envelopes, not attack episodes.",
        "",
        "These limitations currently block a claim that a Temporal GNN sees a correctly ordered physical event stream. A timeblock may contain multiple attacks and an attack may span a boundary; do not equate the two.",
        "",
        "## Direct answers",
        "",
        "1. **Ranh giới Network đã đủ bằng chứng chưa?** Chưa cho V1; đã đủ bằng chứng để xác định chính xác 22 endpoints và 2 cặp có dữ liệu hai file. Cần freeze/implement half-open rule and re-audit before V2.",
        "2. **Split V2 phù hợp nhất?** Protocol A là lựa chọn chính nếu mục tiêu là future known-attack temporal evaluation, nhưng chỉ báo cáo theo nền tảng và phải giữ Network/Windows coverage failure. Protocol B chỉ là exploratory until scenario/session controls pass.",
        "3. **Mỗi nền tảng đủ dữ liệu Known-Attack?** IoT và Linux có hạn chế đủ để đánh giá `scanning`; Network và Windows không có known-type attack rows trong Test, nên FAIL cho mục tiêu đó.",
        "4. **Hạn chế timestamp cản Temporal GNN?** Clock/timezone provenance chưa chứng minh được cho Network/Linux/Windows; IoT có local-vs-epoch ambiguity đã được tính đúng nhưng chưa chứng minh đồng hồ vật lý; missing IoT timestamps phải quarantine; Network endpoints cần ownership rule.",
        "5. **Có thể bắt đầu triển khai V2?** Chưa. Hoàn thành endpoint rule implementation + re-audit, global duplicate/related-flow check, and per-platform clock evidence first. Chỉ sau các gate đó mới tạo export V2 riêng.",
        "",
        "## Artifacts",
        "",
        "`network_boundary_details.csv`, `known_attack_protocol.md`, `unseen_attack_protocol.md`, `candidate_splits.csv`, `class_coverage_comparison.csv`, and `leakage_assessment.md` are generated together by `src/validation/design_split_v2.py`.",
    ]
    (DESIGN_DIR / "FINAL_SPLIT_V2_DESIGN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    DESIGN_DIR.mkdir(parents=True, exist_ok=True)
    boundary, target_rows = build_network_boundary_details()
    counts = _adjust_for_quarantine(_base_counts())
    types = _type_counts()
    counts, types = _apply_half_open_network(counts, types, target_rows)
    candidates = _candidate_rows(counts, types)
    candidates.to_csv(DESIGN_DIR / "candidate_splits.csv", index=False)
    coverage = _coverage_comparison(candidates, types)
    write_protocol_reports(boundary, candidates, coverage)
    write_leakage_assessment(boundary)
    write_final_report(boundary, candidates, coverage)
    print(f"Wrote Split V2 design reports to {DESIGN_DIR}")
    print(f"Network equal boundaries: {len(boundary)}; cross-split: {int(boundary.status.eq('FAIL').sum())}")


if __name__ == "__main__":
    main()
