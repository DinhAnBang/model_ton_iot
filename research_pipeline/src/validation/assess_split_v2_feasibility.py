"""Evidence-only feasibility assessment for Split V2.

The module consumes the existing chunked audit artifacts and the committed
Ground Truth window table.  It does not rerun preprocessing, alter V1, create
an export, or train a model.  Ground Truth rows are treated as labelled
min/max envelopes, never as independently verified attack episodes.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..common.paths import PROJECT_ROOT, REPORT_DIR


OUT_DIR = REPORT_DIR / "split_v2_feasibility"
AUDIT_DIR = REPORT_DIR / "split_audit_v2"
SPLITS = ("train", "validation", "test")
PLATFORMS = ("Network", "IoT", "Linux", "Windows")
V1_TRAIN_END = pd.Timestamp("2019-04-24", tz="UTC")
V1_VALIDATION_END = pd.Timestamp("2019-04-26", tz="UTC")
LATE_TRAIN_END = pd.Timestamp("2019-04-27", tz="UTC")
LATE_VALIDATION_END = pd.Timestamp("2019-04-28", tz="UTC")


def _load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    daily = pd.read_csv(AUDIT_DIR / "daily_label_distribution.csv")
    coverage = pd.read_csv(AUDIT_DIR / "attack_type_coverage.csv")
    gt = pd.read_csv(REPORT_DIR / "ground_truth_attack_windows.csv")
    inventory = pd.read_csv(REPORT_DIR / "dataset_inventory.csv")
    gt["start"] = pd.to_datetime(gt["start_time"], utc=True, errors="coerce")
    gt["end"] = pd.to_datetime(gt["end_time"], utc=True, errors="coerce")
    return daily, coverage, gt, inventory


def _source_file_exists(relative: str) -> bool:
    return (PROJECT_ROOT / str(relative).replace("/", "\\")).exists()


def _interval_metrics(group: pd.DataFrame) -> tuple[int, int]:
    intervals = group[["start", "end"]].dropna().sort_values("start").itertuples(index=False)
    components = 0
    overlap_pairs = 0
    prior: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    current_end: pd.Timestamp | None = None
    for start, end in intervals:
        if current_end is None or start > current_end:
            components += 1
            current_end = end
        else:
            current_end = max(current_end, end)
        overlap_pairs += sum(start <= old_end and old_start <= end for old_start, old_end in prior)
        prior.append((start, end))
    return components, overlap_pairs


def _coverage_lookup(coverage: pd.DataFrame, platform: str, attack_type: str) -> dict[str, object]:
    rows = coverage[(coverage.platform == platform) & (coverage.attack_type == attack_type)]
    if rows.empty:
        return {f"{split}_attack_count": 0 for split in SPLITS} | {f"{split}_seen": "NO" for split in SPLITS}
    return rows.iloc[0].to_dict()


def build_attack_timeline(gt: pd.DataFrame, coverage: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, item in gt.sort_values(["platform", "start", "source_file", "attack_type"]).iterrows():
        lookup = _coverage_lookup(coverage, item.platform, item.attack_type)
        span_days = (item.end - item.start).total_seconds() / 86_400 if pd.notna(item.start) and pd.notna(item.end) else np.nan
        crosses_v1 = bool(
            pd.notna(item.start)
            and pd.notna(item.end)
            and ((item.start < V1_TRAIN_END < item.end) or (item.start < V1_VALIDATION_END < item.end))
        )
        rows.append({
            "ground_truth_id": item.ground_truth_id,
            "platform": item.platform,
            "attack_type": item.attack_type,
            "source_file": item.source_file,
            "source_file_exists": "PASS" if _source_file_exists(item.source_file) else "FAIL",
            "window_start_utc": item.start.isoformat() if pd.notna(item.start) else "UNKNOWN",
            "window_end_utc": item.end.isoformat() if pd.notna(item.end) else "UNKNOWN",
            "window_start_date_utc": item.start.date().isoformat() if pd.notna(item.start) else "UNKNOWN",
            "window_end_date_utc": item.end.date().isoformat() if pd.notna(item.end) else "UNKNOWN",
            "window_span_days": span_days,
            "v1_boundary_crossed_by_envelope": "YES" if crosses_v1 else "NO",
            "train_attack_count_current_v1": int(lookup.get("train_attack_count", 0)),
            "validation_attack_count_current_v1": int(lookup.get("validation_attack_count", 0)),
            "test_attack_count_current_v1": int(lookup.get("test_attack_count", 0)),
            "episode_id_present": "NO",
            "independence_status": "UNVERIFIED",
            "evidence_status": "PASS" if _source_file_exists(item.source_file) else "FAIL",
        })
    result = pd.DataFrame(rows)
    # Keep the expression above readable while avoiding any claim that a
    # parsed envelope is an episode.  Missing source paths are the only hard
    # failure in this artifact.
    result["evidence_status"] = np.where(result.source_file_exists.eq("PASS"), "PASS", "FAIL")
    result.to_csv(OUT_DIR / "attack_timeline.csv", index=False)
    return result


def build_attack_recurrence(gt: pd.DataFrame, coverage: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (platform, attack_type), group in gt.groupby(["platform", "attack_type"], sort=True):
        components, overlap_pairs = _interval_metrics(group)
        start_dates = sorted(group.start.dt.date.astype(str).unique())
        files = sorted(group.source_file.unique())
        lookup = _coverage_lookup(coverage, platform, attack_type)
        rows.append({
            "platform": platform,
            "attack_type": attack_type,
            "ground_truth_window_count": len(group),
            "ground_truth_source_file_count": len(files),
            "ground_truth_start_date_count": len(start_dates),
            "ground_truth_start_dates_utc": ";".join(start_dates),
            "first_window_start_utc": group.start.min().isoformat(),
            "last_window_end_utc": group.end.max().isoformat(),
            "disjoint_envelope_components": components,
            "overlapping_envelope_pairs": overlap_pairs,
            "recurs_across_source_files": "YES" if len(files) > 1 else "NO",
            "recurs_across_start_dates": "YES" if len(start_dates) > 1 else "NO",
            "current_v1_train_attack_count": int(lookup.get("train_attack_count", 0)),
            "current_v1_validation_attack_count": int(lookup.get("validation_attack_count", 0)),
            "current_v1_test_attack_count": int(lookup.get("test_attack_count", 0)),
            "current_v1_train_seen": lookup.get("train_seen", "NO"),
            "current_v1_test_seen": lookup.get("test_seen", "NO"),
            "episode_independence_evidence": "NO_SHARED_SESSION_OR_RUN_ID",
            "recurrence_interpretation": "REPEATED_LABELLED_ENVELOPES_NOT_VERIFIED_EPISODES",
            "status": "UNVERIFIED",
        })
    result = pd.DataFrame(rows)
    result.to_csv(OUT_DIR / "attack_recurrence.csv", index=False)
    return result


def _assign_dates(daily: pd.DataFrame, train_end: pd.Timestamp, validation_end: pd.Timestamp, include_missing_in_train: bool) -> pd.DataFrame:
    data = daily.copy()
    data["parsed_day"] = pd.to_datetime(data.event_day, errors="coerce", utc=True)
    data["candidate_split"] = "quarantine"
    valid = data.parsed_day.notna()
    data.loc[valid, "candidate_split"] = "test"
    data.loc[valid & (data.parsed_day < validation_end), "candidate_split"] = "validation"
    data.loc[valid & (data.parsed_day < train_end), "candidate_split"] = "train"
    missing = ~valid
    if include_missing_in_train:
        data.loc[missing, "candidate_split"] = "train"
    return data


def _counts(data: pd.DataFrame) -> pd.DataFrame:
    grouped = data.groupby(["platform", "candidate_split"], as_index=False)[["normal_count", "attack_count", "total_count"]].sum()
    result = []
    for platform in PLATFORMS:
        item = {"platform": platform}
        for split in (*SPLITS, "quarantine"):
            row = grouped[(grouped.platform == platform) & (grouped.candidate_split == split)]
            if row.empty:
                values = {"normal_count": 0, "attack_count": 0, "total_count": 0}
            else:
                values = row.iloc[0].to_dict()
            item[f"{split}_rows"] = int(values["total_count"])
            item[f"{split}_normal"] = int(values["normal_count"])
            item[f"{split}_attack"] = int(values["attack_count"])
            item[f"{split}_attack_rate"] = (item[f"{split}_attack"] / item[f"{split}_rows"]) if item[f"{split}_rows"] else 0.0
        result.append(item)
    return pd.DataFrame(result)


def _gt_split_evidence(gt: pd.DataFrame, train_end: pd.Timestamp, validation_end: pd.Timestamp) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for platform, group in gt.groupby("platform"):
        values: dict[str, set[str]] = {split: set() for split in SPLITS}
        crossing = 0
        for _, item in group.iterrows():
            memberships = set()
            if item.start < train_end:
                memberships.add("train")
            if item.start < validation_end and item.end > train_end:
                memberships.add("validation")
            if item.end > validation_end:
                memberships.add("test")
            if len(memberships) > 1:
                crossing += 1
            for split in memberships:
                values[split].add(str(item.attack_type))
        train_types = values["train"]
        test_types = values["test"]
        result[platform] = {
            "train_attack_types_evidence": ";".join(sorted(train_types)),
            "validation_attack_types_evidence": ";".join(sorted(values["validation"])),
            "test_attack_types_evidence": ";".join(sorted(test_types)),
            "known_test_types_evidence": ";".join(sorted(test_types & train_types)),
            "unseen_test_types_evidence": ";".join(sorted(test_types - train_types)),
            "envelopes_crossing_candidate_boundaries": crossing,
        }
    return result


def _v1_type_evidence(coverage: pd.DataFrame, platform: str) -> dict[str, object]:
    part = coverage[(coverage.platform == platform) & (coverage.total_attack_count > 0)]
    train = set(part.loc[part.train_seen.eq("YES"), "attack_type"])
    test = set(part.loc[part.test_seen.eq("YES"), "attack_type"])
    return {
        "train_attack_types_evidence": ";".join(sorted(train)),
        "validation_attack_types_evidence": ";".join(sorted(part.loc[part.validation_seen.eq("YES"), "attack_type"])),
        "test_attack_types_evidence": ";".join(sorted(test)),
        "known_test_types_evidence": ";".join(sorted(train & test)),
        "unseen_test_types_evidence": ";".join(sorted(test - train)),
        "envelopes_crossing_candidate_boundaries": "NOT_USED_EXACT_V1_COVERAGE",
    }


def build_temporal_candidates(daily: pd.DataFrame, coverage: pd.DataFrame, gt: pd.DataFrame) -> pd.DataFrame:
    candidates = [
        ("TEMP_V1_CONTROL", V1_TRAIN_END, V1_VALIDATION_END, True, "Current V1 boundaries; invalid IoT time remains Train only for control."),
        ("TEMP_LATE_TRAIN_20270427", LATE_TRAIN_END, LATE_VALIDATION_END, False, "Prespecified sensitivity candidate: Train before 2019-04-27, Validation on 2019-04-27, Test from 2019-04-28; not selected by a metric."),
    ]
    rows = []
    for candidate_id, train_end, validation_end, include_missing, basis in candidates:
        if candidate_id == "TEMP_V1_CONTROL":
            assigned = daily.copy()
            assigned["candidate_split"] = assigned["split"]
        else:
            assigned = _assign_dates(daily, train_end, validation_end, include_missing)
        counts = _counts(assigned).set_index("platform")
        gt_evidence = _gt_split_evidence(gt, train_end, validation_end)
        for platform in PLATFORMS:
            record = {"candidate_id": candidate_id, "protocol": "known_attack_temporal", "platform": platform, "train_end_exclusive_utc": train_end.isoformat(), "validation_end_exclusive_utc": validation_end.isoformat(), "selection_basis": basis, "invalid_timestamp_policy": "train_for_control" if include_missing else "quarantine", "timestamp_status": "UNVERIFIED"}
            for split in (*SPLITS, "quarantine"):
                for metric in ("rows", "normal", "attack", "attack_rate"):
                    record[f"{split}_{metric}"] = counts.loc[platform, f"{split}_{metric}"]
            evidence = _v1_type_evidence(coverage, platform) if candidate_id == "TEMP_V1_CONTROL" else gt_evidence.get(platform, {})
            record.update(evidence)
            known = set(filter(None, str(record.get("known_test_types_evidence", "")).split(";")))
            record["coverage_status"] = "PASS" if known else "FAIL"
            record["episode_boundary_status"] = "UNVERIFIED" if record.get("envelopes_crossing_candidate_boundaries") != 0 else "UNVERIFIED"
            record["overall_status"] = "PASS" if candidate_id == "TEMP_V1_CONTROL" and platform in {"IoT", "Linux"} and known else "UNVERIFIED"
            rows.append(record)
    result = pd.DataFrame(rows)
    result.to_csv(OUT_DIR / "candidate_temporal_splits.csv", index=False)
    return result


def _source_unit(platform: str, file_name: str) -> str:
    if platform == "Network":
        match = re.search(r"(\d+)", file_name)
        return f"Network_dataset_{int(match.group(1))}.csv" if match else file_name
    if platform == "IoT":
        return file_name
    if platform == "Linux":
        lower = file_name.lower()
        for family in ("disk", "memory", "process"):
            if family in lower:
                return f"linux_{family}"
    if platform == "Windows":
        return file_name
    return file_name


def _gt_unit(platform: str, source_file: str) -> str:
    name = Path(source_file).stem
    if platform == "Network":
        match = re.search(r"Network_(\d+)$", name)
        return f"Network_dataset_{int(match.group(1))}.csv" if match else name
    if platform == "IoT":
        return name.replace("GroundTruth_", "") + ".csv"
    if platform == "Linux":
        family = name.replace("GroundTruth_Linux_", "").lower()
        return f"linux_{family}"
    if platform == "Windows":
        suffix = name.replace("GroundTruth_Windows", "")
        return f"windows{suffix}_dataset.csv"
    return name


def _unit_assignments(inventory: pd.DataFrame) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    network = inventory[inventory.platform.eq("Network")].copy()
    network["sort_time"] = pd.to_datetime(network.timestamp_min, errors="coerce", utc=True)
    network = network.sort_values(["sort_time", "file_name"])
    result["Network"] = {row.file_name: ("train" if i < 16 else "validation" if i < 19 else "test") for i, row in enumerate(network.itertuples())}
    for platform, train_n, val_n in (("IoT", 5, 1), ("Linux", 4, 1), ("Windows", 1, 0)):
        files = sorted(inventory.loc[inventory.platform.eq(platform), "file_name"].tolist(), key=str.lower)
        result[platform] = {file_name: ("train" if i < train_n else "validation" if i < train_n + val_n else "test") for i, file_name in enumerate(files)}
    return result


def build_scenario_candidates(daily: pd.DataFrame, coverage: pd.DataFrame, gt: pd.DataFrame, inventory: pd.DataFrame) -> pd.DataFrame:
    assignments = _unit_assignments(inventory)
    file_stats = daily.groupby(["platform", "source_file"], as_index=False)[["normal_count", "attack_count", "total_count"]].sum()
    missing_stats = (
        daily[daily.event_day.eq("MISSING_TIMESTAMP")]
        .groupby(["platform", "source_file"], as_index=False)["total_count"]
        .sum()
        .rename(columns={"total_count": "missing_count"})
    )
    gt = gt.copy()
    gt["unit"] = [_gt_unit(platform, source) for platform, source in zip(gt.platform, gt.source_file)]
    rows = []
    for platform in PLATFORMS:
        mapping = assignments[platform]
        unit_rows = []
        for unit, split in mapping.items():
            file_name = unit if platform != "Linux" else None
            if platform == "Linux":
                needle = unit.lower().replace("linux_", "")
                matching = inventory[(inventory.platform == platform) & inventory.file_name.str.lower().str.contains(needle, regex=False)]
                file_names = matching.file_name.tolist()
            else:
                file_names = [file_name]
            stats = file_stats[(file_stats.platform == platform) & file_stats.source_file.isin(file_names)]
            missing = missing_stats[(missing_stats.platform == platform) & missing_stats.source_file.isin(file_names)]
            unit_rows.append({"unit": unit, "split": split, "normal": int(stats.normal_count.sum()), "attack": int(stats.attack_count.sum()), "rows": int(stats.total_count.sum()), "missing": int(missing.missing_count.sum()), "files": file_names})
        record = {"candidate_id": f"SCENARIO_SOURCE_HOLDOUT_{platform.upper()}", "platform": platform, "unit_definition": "complete source file" if platform != "Linux" else "complete telemetry family/source files", "unit_count": len(mapping), "train_units": ";".join(item["unit"] for item in unit_rows if item["split"] == "train"), "validation_units": ";".join(item["unit"] for item in unit_rows if item["split"] == "validation"), "test_units": ";".join(item["unit"] for item in unit_rows if item["split"] == "test"), "selection_basis": "deterministic inventory order; not optimized on labels or model output"}
        for split in (*SPLITS, "quarantine"):
            selected = [item for item in unit_rows if item["split"] == split]
            normal = sum(item["normal"] for item in selected)
            attack = sum(item["attack"] for item in selected)
            quarantine = sum(item["missing"] for item in selected)
            if split == "quarantine":
                quarantine = sum(item["missing"] for item in unit_rows)
                normal = quarantine
                attack = 0
            else:
                normal -= quarantine
            record[f"{split}_rows"] = int(normal + attack)
            record[f"{split}_normal"] = int(normal)
            record[f"{split}_attack"] = int(attack)
            record[f"{split}_quarantine"] = int(quarantine)
            record[f"{split}_attack_rate"] = (record[f"{split}_attack"] / record[f"{split}_rows"]) if record[f"{split}_rows"] else 0.0
        types_by_split = {}
        for split in SPLITS:
            units = {item["unit"] for item in unit_rows if item["split"] == split}
            if platform == "Linux":
                gt_units = {family for unit in units for family in ("disk", "memory", "process") if family in unit.lower()}
                types = sorted(set(gt.loc[(gt.platform == platform) & gt.unit.isin({f"linux_{family}" for family in gt_units}), "attack_type"]))
            else:
                types = sorted(set(gt.loc[(gt.platform == platform) & gt.unit.isin(units), "attack_type"]))
            types_by_split[split] = types
            record[f"{split}_attack_types_evidence"] = ";".join(types)
        train_types = set(types_by_split["train"])
        test_types = set(types_by_split["test"])
        record["known_test_types_evidence"] = ";".join(sorted(train_types & test_types))
        record["unseen_test_types_evidence"] = ";".join(sorted(test_types - train_types))
        if platform == "Windows":
            status = "FAIL"
            reason = "Only two source files; no complete three-fold source holdout is possible."
        elif platform == "Linux":
            status = "NOT_FEASIBLE"
            reason = "Disk/memory/process files are telemetry families/segments; independent attack scenarios are not identified."
        elif platform == "IoT":
            status = "UNVERIFIED"
            reason = "Device files can form folds, but the same labelled campaigns are repeated across devices; independent episodes are not proven."
        else:
            status = "UNVERIFIED"
            reason = "Chronological Network source files form a deterministic holdout, but are not verified independent scenarios; equal endpoints require the V2 half-open rule."
        record["independence_status"] = "UNVERIFIED"
        record["timestamp_status"] = "UNVERIFIED"
        record["status"] = status
        record["limitation"] = reason
        rows.append(record)
    rows.append({
        "candidate_id": "SCENARIO_GROUND_TRUTH_WINDOW_HOLDOUT",
        "platform": "ALL",
        "unit_definition": "105 source_file+attack_type min/max windows",
        "unit_count": 105,
        "train_units": "NOT_ASSIGNED",
        "validation_units": "NOT_ASSIGNED",
        "test_units": "NOT_ASSIGNED",
        "selection_basis": "Ground Truth inventory only",
        "independence_status": "UNVERIFIED",
        "timestamp_status": "UNVERIFIED",
        "status": "NOT_FEASIBLE",
        "limitation": "No shared session/run ID; windows overlap and are explicitly envelopes, not verified independent episodes. Counts are not assigned to avoid inventing rows.",
    })
    result = pd.DataFrame(rows)
    result.to_csv(OUT_DIR / "candidate_scenario_splits.csv", index=False)
    return result


def write_split_comparison(temporal: pd.DataFrame, scenario: pd.DataFrame, recurrence: pd.DataFrame) -> None:
    lines = [
        "# Split V2 feasibility comparison",
        "",
        "## Evidence basis",
        "",
        "The assessment reuses `split_audit_v2/daily_label_distribution.csv`, `attack_type_coverage.csv`, `timestamp_quality.csv`, the preprocessing remediation reports, and the 105-row `ground_truth_attack_windows.csv`. Ground Truth source paths were checked without reparsing the 31M processed rows. No preprocessing or model training was run.",
        "",
        "## Attack recurrence",
        "",
        f"Ground Truth contains {len(recurrence) and int(recurrence.ground_truth_window_count.sum())} labelled type/file envelopes. Repeated windows show recurrence of a labelled type across files or dates, but every row remains `UNVERIFIED` as an independent episode because the Ground Truth parser records no shared session/run ID and uses source-file/type min/max envelopes.",
        "",
        "Interpretation rules:",
        "- `recurs_across_start_dates=YES` is evidence that the type appears in more than one observed date, not proof of independent attacks.",
        "- `disjoint_envelope_components>1` indicates separated envelope groups, but still does not create an episode ID.",
        "- Overlapping envelopes are explicitly not split into separate attacks.",
        "",
        "## Temporal candidates",
        "",
        "| Candidate | Definition | Exact counts | Known-test coverage | Status |\n|---|---|---|---|---|",
        "| TEMP_V1_CONTROL | Train < 2019-04-24; Validation 2019-04-24–25; Test >= 2019-04-26 | Yes, from V1 daily audit | IoT/Linux only have known `scanning`; Network/Windows have none | CONTROL / PARTIAL |\n| TEMP_LATE_TRAIN_20270427 | Train < 2019-04-27; Validation 2019-04-27; Test >= 2019-04-28 | Date aggregation from existing audit; invalid IoT timestamps quarantined | Ground Truth envelope evidence only; boundary-crossing windows remain | UNVERIFIED |",
        "",
        "The late-train candidate is a sensitivity design, not a selected final split. Its purpose is to test whether later observations create a real known-type future evaluation. It must not be chosen because it improves class balance or expected Accuracy.",
        "",
        "## Scenario candidates",
        "",
        "| Candidate | Evidence | Status |\n|---|---|---|",
        "| Network complete source files | 23 chronological source files, deterministic 16/3/4 holdout; endpoint ambiguity has two cross-split cases | UNVERIFIED as scenario independence |\n| IoT complete device files | 7 devices can form 5/1/1 folds, but attack campaigns repeat across devices | UNVERIFIED |\n| Linux complete files/families | 6 files are disk/memory/process telemetry segments; no independent scenario ID | NOT_FEASIBLE as scenario holdout |\n| Windows complete source files | Only 2 processed files | FAIL for 3-way source holdout |\n| 105 Ground Truth windows | Min/max envelopes, no session/run ID | NOT_FEASIBLE |",
        "",
        "## Leakage, timestamp and independence",
        "",
        "Feature selection and preprocessing are Train-only after remediation (`preprocessing_remediation/FINAL_REPORT.md`, PASS code invariant). Cross-file exact duplicate audit observed 62 fingerprints/358 pair occurrences and zero V1 split conflicts, but the router is reset per source file; this is not proof of global scenario isolation. Network has two equal endpoints across V1 folds; V2 needs the half-open rule already documented in `split_v2_design/network_boundary_details.csv`.",
        "",
        "Timestamp syntax is parseable. IoT raw Ground Truth arithmetic is +25,200 seconds UTC-minus-local and −25,200 seconds for the inverse, but physical clock synchronization remains unverified. Network/Linux/Windows clock provenance is also unverified. Therefore a single cross-platform event order for Temporal GNN is not yet proven.",
        "",
        "## Practical choice",
        "",
        "The evidence supports a **partially feasible** Heterogeneous Temporal GNN study: use real platform-local chronological data, a binary detection objective, and coverage-conditioned evaluation. Report IoT/Linux known-scanning future performance separately; report Network/Windows future attacks as unseen/OOD or exclude them from the known-attack claim. Do not synthesize episodes, align clocks without evidence, or pool cross-platform timestamps as if they were one physical clock.",
        "",
        "A publishable scenario/unseen-episode protocol is not currently feasible. It requires a verified run/session/scenario identifier or new provenance evidence, not another arbitrary time cut.",
    ]
    (OUT_DIR / "split_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_final_decision(temporal: pd.DataFrame, scenario: pd.DataFrame, recurrence: pd.DataFrame) -> None:
    late = temporal[temporal.candidate_id.eq("TEMP_LATE_TRAIN_20270427")].set_index("platform")
    count_lines = [
        f"| {platform} | {int(late.loc[platform, 'train_rows']):,} | {int(late.loc[platform, 'validation_rows']):,} | {int(late.loc[platform, 'test_rows']):,} | {int(late.loc[platform, 'quarantine_rows']):,} |"
        for platform in PLATFORMS
    ]
    lines = [
        "# FINAL FEASIBILITY DECISION — Split V2 / Heterogeneous Temporal GNN",
        "",
        "## Decision",
        "",
        "# PARTIALLY FEASIBLE",
        "",
        "ToN-IoT có thể được dùng để triển khai một nghiên cứu Heterogeneous Temporal GNN có phạm vi kiểm soát, nhưng **chưa đủ bằng chứng cho một Split V2 duy nhất vừa known-attack temporal vừa scenario-based unseen-attack trên cả Network, IoT, Linux và Windows**.",
        "",
        "## What is feasible now",
        "",
        "1. **Platform-local temporal binary detection:** có thể dùng các dòng có timestamp hợp lệ, giữ thứ tự trong từng nền tảng, dùng preprocessing Train-only đã remediation. IoT và Linux có bằng chứng known `scanning` trong Train và Test của V1: lần lượt 2,198/1,775 và 851/75,020 dòng Attack. Đây là phạm vi khả thi có điều kiện, không phải coverage đầy đủ mọi attack type.",
        "2. **Network temporal/OOD analysis:** Network có thứ tự source-file rõ và 22 endpoint đã được kiểm kê, nhưng Test hiện có 0 dòng của attack type đã thấy trong Train; vì vậy chỉ được báo cáo như natural unseen/OOD hoặc temporal composition analysis, không phải known-attack temporal evaluation.",
        "3. **Windows binary temporal analysis:** có thể giữ làm nhánh phụ, nhưng Test có 0 dòng known attack type từ Train và chỉ có hai source files; không đủ cho known-attack hoặc three-way scenario holdout.",
        "",
        "## What is not feasible",
        "",
        "- **Scenario/episode holdout:** NOT FEASIBLE với bằng chứng hiện có. 105 Ground Truth rows là source_file+attack_type min/max envelopes; không có shared session/run ID. Không được biến chúng thành 105 episode độc lập.",
        "- **Một temporal graph chung cho bốn nền tảng:** UNVERIFIED. IoT có local-vs-epoch evidence nhưng chưa chứng minh physical clock; Network/Linux/Windows timezone/clock provenance chưa đủ. Không được tự quy đổi timezone hay nối các sự kiện cross-platform theo một trục thời gian giả định.",
        "- **Known-attack benchmark toàn bộ bốn nền tảng:** NOT FEASIBLE với V1 coverage: Network và Windows không có known attack trong Test; IoT/Linux chỉ có `scanning`.",
        "",
        "## Candidate counts",
        "",
        "`candidate_temporal_splits.csv` contains exact date-aggregated counts. For the remediated late-train candidate (Train < 2019-04-27, Validation 2019-04-27, Test >= 2019-04-28), the counts are:",
        "",
        "| Platform | Train rows | Validation rows | Test rows | Quarantine |",
        "|---|---:|---:|---:|---:|",
        *count_lines,
        "",
        "These are exact calendar-day aggregates from the existing audit; they are not an executed export. No candidate is selected by label ratio or expected Accuracy.",
        "",
        "## Minimum conditions to proceed",
        "",
        "1. Freeze a protocol scope: platform-local temporal binary detection, plus separate known/unseen coverage reports.",
        "2. Implement and re-audit the already proposed Network half-open endpoint rule in a new V2 module; do not alter V1.",
        "3. Keep 25,956 invalid IoT timestamps in QUARANTINE and exclude them from graph construction/training/evaluation claims.",
        "4. Do not create cross-platform temporal edges until clock basis is verified; use platform-local time or an explicitly non-synchronous graph.",
        "5. Fit feature selection/preprocessing on Train only and report attack-type coverage as an evaluation mask.",
        "6. Obtain a verified scenario/session/run identifier before attempting Protocol B as a publishable unseen-episode experiment.",
        "",
        "## Final answer to the research proposal",
        "",
        "**Có thể tiếp tục nghiên cứu Heterogeneous Temporal GNN, nhưng phải thu hẹp claim:** mô hình có thể học trên heterogeneous platform/feature schema và đánh giá theo thời gian trong từng nền tảng; không được tuyên bố đã chứng minh generalization theo thời gian và theo episode cho toàn bộ ToN-IoT. Phạm vi known-attack đáng tin cậy hiện là IoT/Linux với `scanning`; Network/Windows phải tách thành unseen/OOD hoặc báo cáo giới hạn coverage.",
        "",
        "No Split V2 manifest is emitted because the overall decision is PARTIALLY FEASIBLE, not FEASIBLE. No data, V1 split, preprocessing output, or model was changed.",
    ]
    (OUT_DIR / "FINAL_FEASIBILITY_DECISION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    daily, coverage, gt, inventory = _load()
    timeline = build_attack_timeline(gt, coverage)
    recurrence = build_attack_recurrence(gt, coverage)
    temporal = build_temporal_candidates(daily, coverage, gt)
    scenario = build_scenario_candidates(daily, coverage, gt, inventory)
    write_split_comparison(temporal, scenario, recurrence)
    write_final_decision(temporal, scenario, recurrence)
    print(f"Wrote feasibility reports to {OUT_DIR}")
    print(f"Ground Truth windows: {len(timeline)}; recurrence groups: {len(recurrence)}")
    print(f"Temporal candidates: {len(temporal)} rows; scenario candidates: {len(scenario)} rows")


if __name__ == "__main__":
    main()
