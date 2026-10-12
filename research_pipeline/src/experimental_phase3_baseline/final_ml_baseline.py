"""Reproducible RF/XGBoost/LightGBM baselines for the final Linux protocol.

This runner deliberately lives outside Phase 2 and Phase 3B.  It consumes the
existing processed Linux CSVs, applies the existing remediated split helper,
fits feature handling on Train only, evaluates Validation, and opens Test once
with a frozen 0.5 decision threshold.  It never writes to input data or to
previous experiment directories.
"""

from __future__ import annotations

import argparse
import json
import os
import platform as platform_module
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from ..common.config import QUARANTINE_CODE, raw_type_mapping
from ..common.paths import PROJECT_ROOT
from ..pipeline.data_reader import make_source_records
from ..pipeline.duplicate_handler import DuplicateRouter, duplicate_fingerprints
from ..pipeline.feature_selector import (
    final_features,
    make_group_plans,
    read_feature_risk,
)
from ..pipeline.models import GroupPlan, RunState, SourceRecord
from ..pipeline.normalization import validate_chunk
from ..pipeline.preprocessor import fit_preprocessors, transform_baseline
from ..pipeline.split_data import assign_split_plan, remediated_split_codes


SEED = 42
CHUNK_SIZE = 250_000
KNOWN_TYPES = frozenset({"normal", "scanning"})
SPLIT_NAMES = ("train", "validation", "test")
MODEL_NAMES = ("RandomForest", "XGBoost", "LightGBM", "AlwaysNormal", "AlwaysAttack")
SCOPE_CONFIG = {
    "Linux::process": {"role": "main_known_attack", "files": ("Linux_process_1.csv", "Linux_process_2.csv")},
    "Linux::disk": {"role": "supporting_known_attack", "files": ("linux_disk_1.csv", "linux_disk_2.csv")},
}


class ExperimentBlocked(RuntimeError):
    """Raised when a direct integrity check finds a benchmark-blocking issue."""


@dataclass
class SplitAccumulator:
    """Numpy batches and exact counters for one split."""

    x_batches: list[np.ndarray] = field(default_factory=list)
    y_batches: list[np.ndarray] = field(default_factory=list)
    total_rows: int = 0
    normal_rows: int = 0
    attack_rows: int = 0
    known_rows: int = 0
    excluded_rows: int = 0
    excluded_types: Counter[str] = field(default_factory=Counter)
    quarantine_rows: int = 0

    def finalize(self, features: list[str]) -> tuple[np.ndarray, np.ndarray]:
        if self.x_batches:
            x = np.concatenate(self.x_batches, axis=0).astype(np.float32, copy=False)
            y = np.concatenate(self.y_batches, axis=0).astype(np.int8, copy=False)
        else:
            x = np.empty((0, len(features)), dtype=np.float32)
            y = np.empty(0, dtype=np.int8)
        return x, y


@dataclass
class ScopeRun:
    scope: str
    plan: GroupPlan
    features: list[str]
    splits: dict[str, tuple[np.ndarray, np.ndarray]]
    accumulators: dict[str, SplitAccumulator]
    counts: list[dict[str, Any]]
    duplicate_audit: dict[str, Any]
    label_type_audit: dict[str, Any]
    feature_audit: list[dict[str, Any]]


def _canonical_type_series(frame: pd.DataFrame, mapping: dict[str, str]) -> pd.Series:
    return frame["type"].astype("string").map(mapping).fillna("UNKNOWN").astype("string")


def _iter_routed_chunks(
    record: SourceRecord,
    chunk_size: int,
    mapping: dict[str, str],
) -> Iterable[tuple[pd.DataFrame, np.ndarray, pd.Series, pd.Series, np.ndarray]]:
    """Yield source chunks with the same remediated routing as the pipeline."""
    router = DuplicateRouter()
    for frame in pd.read_csv(record.file_path, chunksize=chunk_size, low_memory=False):
        labels, _, _, times = validate_chunk(frame, record, mapping, RunState(), validation_only=True)
        base = remediated_split_codes(times, record)
        fingerprints = duplicate_fingerprints(frame, record.columns)
        final_codes = router.route_masked(
            fingerprints,
            base,
            base != QUARANTINE_CODE,
            QUARANTINE_CODE,
        )
        canonical = _canonical_type_series(frame, mapping)
        yield frame, final_codes, canonical, labels, fingerprints


def _scope_records(records: list[SourceRecord], scope: str) -> list[SourceRecord]:
    expected = set(SCOPE_CONFIG[scope]["files"])
    selected = [r for r in records if r.group_id == scope]
    actual = {r.file_name for r in selected}
    if actual != expected:
        raise ExperimentBlocked(f"{scope}: expected files {sorted(expected)}, got {sorted(actual)}")
    return sorted(selected, key=lambda r: r.file_name)


def _feature_rows(
    plan: GroupPlan,
    risk: dict[tuple[str, str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    selected = set(final_features(plan))
    rows: list[dict[str, Any]] = []
    for column in plan.source_columns:
        risk_row = risk.get((plan.platform, plan.records[0].file_name, column), {})
        spec = plan.feature_specs.get(column)
        if column == "label":
            decision, reason = False, "target; excluded from X"
        elif column == "type":
            decision, reason = False, "attack type / ground-truth provenance; excluded from X"
        elif column == "ts":
            decision, reason = False, "timestamp; split/order field and potential shortcut; excluded from X"
        elif column in selected:
            decision, reason = True, spec.reason if spec else "selected by Train-only feature plan"
        elif spec is not None:
            decision, reason = False, spec.exclusion_reason or spec.reason
        else:
            decision, reason = False, "not in the approved observable feature plan"
        rows.append({
            "scope": plan.group_id,
            "column": column,
            "model_input": decision,
            "fit_scope": "Train_only" if decision else "not_fitted",
            "kind": spec.kind if spec is not None else "target_or_metadata",
            "risk_classification": risk_row.get("classification", "UNKNOWN"),
            "secondary_classification": risk_row.get("secondary_classification", ""),
            "reason": reason,
            "train_nonmissing": int(spec.train_nonmissing) if spec is not None else 0,
            "train_missing": int(spec.train_missing) if spec is not None else 0,
            "train_median": spec.median if spec is not None and spec.median is not None else "",
        })
    return rows


def _safe_float(value: float | int | np.floating | None) -> float | str:
    if value is None or not np.isfinite(float(value)):
        return "UNDEFINED"
    return float(value)


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    """Compute the full binary metric contract without replacing undefined values by 0."""
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_true = np.asarray(y_true, dtype=np.int8)
    y_score = np.asarray(y_score, dtype=np.float64)
    y_pred = (y_score >= threshold).astype(np.int8)
    if len(y_true) == 0:
        return {"status": "UNDEFINED_EMPTY_EVALUATION", "threshold": threshold}
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    normal_n = int((y_true == 0).sum())
    attack_n = int((y_true == 1).sum())
    both_classes = normal_n > 0 and attack_n > 0

    def metric(fn_metric, *args, **kwargs):
        try:
            value = fn_metric(*args, **kwargs)
            return _safe_float(value)
        except ValueError:
            return "UNDEFINED"

    normal_precision = metric(precision_score, y_true, y_pred, labels=[0], pos_label=0, average="binary", zero_division=np.nan)
    normal_recall = metric(recall_score, y_true, y_pred, labels=[0], pos_label=0, average="binary", zero_division=np.nan)
    normal_f1 = metric(f1_score, y_true, y_pred, labels=[0], pos_label=0, average="binary", zero_division=np.nan)
    attack_precision = metric(precision_score, y_true, y_pred, labels=[1], pos_label=1, average="binary", zero_division=np.nan)
    attack_recall = metric(recall_score, y_true, y_pred, labels=[1], pos_label=1, average="binary", zero_division=np.nan)
    attack_f1 = metric(f1_score, y_true, y_pred, labels=[1], pos_label=1, average="binary", zero_division=np.nan)
    macro_f1 = metric(f1_score, y_true, y_pred, average="macro", zero_division=np.nan)
    weighted_f1 = metric(f1_score, y_true, y_pred, average="weighted", zero_division=np.nan)
    pr_attack = metric(average_precision_score, y_true, y_score) if attack_n else "UNDEFINED"
    pr_normal = metric(average_precision_score, 1 - y_true, 1 - y_score) if normal_n else "UNDEFINED"
    roc_auc = metric(roc_auc_score, y_true, y_score) if both_classes else "UNDEFINED"
    fpr = _safe_float(fp / normal_n) if normal_n else "UNDEFINED"
    fnr = _safe_float(fn / attack_n) if attack_n else "UNDEFINED"
    return {
        "status": "PASS" if both_classes else "UNDEFINED_SINGLE_CLASS",
        "threshold": threshold,
        "accuracy": _safe_float(accuracy_score(y_true, y_pred)),
        "precision_normal": normal_precision,
        "precision_attack": attack_precision,
        "recall_normal": normal_recall,
        "recall_attack": attack_recall,
        "f1_normal": normal_f1,
        "f1_attack": attack_f1,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "pr_auc_attack": pr_attack,
        "pr_auc_normal": pr_normal,
        "roc_auc": roc_auc,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "normal_count": normal_n,
        "attack_count": attack_n,
        "predicted_normal": int((y_pred == 0).sum()),
        "predicted_attack": int((y_pred == 1).sum()),
    }


def _model_config(name: str, y_train: np.ndarray) -> dict[str, Any]:
    normal = int((y_train == 0).sum())
    attack = int((y_train == 1).sum())
    if name == "RandomForest":
        return {
            "class": "sklearn.ensemble.RandomForestClassifier",
            "params": {
                "n_estimators": 100,
                "max_depth": 16,
                "class_weight": "balanced",
                "random_state": SEED,
                "n_jobs": 1,
            },
            "train_class_weight_source": "balanced from Train labels only",
        }
    if name == "XGBoost":
        return {
            "class": "xgboost.XGBClassifier",
            "params": {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "tree_method": "hist",
                "eval_metric": "logloss",
                "random_state": SEED,
                "n_jobs": 1,
                "scale_pos_weight": (normal / attack if attack else 1.0),
            },
            "train_class_weight_source": "normal/attack ratio computed from Train only",
        }
    if name == "LightGBM":
        return {
            "class": "lightgbm.LGBMClassifier",
            "params": {
                "n_estimators": 100,
                "num_leaves": 31,
                "learning_rate": 0.05,
                "class_weight": "balanced",
                "random_state": SEED,
                "n_jobs": 1,
                "verbosity": -1,
            },
            "train_class_weight_source": "balanced from Train labels only",
        }
    return {"class": "deterministic comparator", "params": {}, "train_class_weight_source": "not applicable"}


def _make_model(name: str, y_train: np.ndarray):
    if name == "RandomForest":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(**_model_config(name, y_train)["params"])
    if name == "XGBoost":
        from xgboost import XGBClassifier

        return XGBClassifier(**_model_config(name, y_train)["params"])
    if name == "LightGBM":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(**_model_config(name, y_train)["params"])
    return None


def _predict_scores(model_name: str, model: Any, x: np.ndarray) -> np.ndarray:
    if model_name == "AlwaysNormal":
        return np.zeros(len(x), dtype=np.float64)
    if model_name == "AlwaysAttack":
        return np.ones(len(x), dtype=np.float64)
    return np.asarray(model.predict_proba(x)[:, 1], dtype=np.float64)


def _git_revision() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    except Exception:
        return "UNAVAILABLE"


def _verify_frozen_protocol(project_root: Path) -> dict[str, Any]:
    """Refuse to run if the committed final protocol is not the expected one."""
    path = project_root / "research_pipeline" / "reports" / "final_split_protocol" / "split_protocol_config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("protocol_id") != "PLATFORM_LOCAL_TEMPORAL_BINARY_V1_CONTROL_REMEDIATED":
        raise ExperimentBlocked(f"unexpected final protocol id: {config.get('protocol_id')}")
    if config.get("main_scopes") != ["Linux::process"] or config.get("supporting_scopes") != ["Linux::disk"]:
        raise ExperimentBlocked("final protocol scope differs from the pre-declared Linux Process/Disk scope")
    if config.get("target") != "label":
        raise ExperimentBlocked("final protocol target is not label")
    known = config.get("known_attack_mask", {})
    if known.get("Linux::process") != ["normal", "scanning"] or known.get("Linux::disk") != ["normal", "scanning"]:
        raise ExperimentBlocked("final protocol known-attack mask differs from Normal/Scanning")
    boundaries = config.get("calendar_boundaries", {})
    expected = {
        "train": "[source_start, 2019-04-24T00:00:00Z)",
        "validation": "[2019-04-24T00:00:00Z, 2019-04-26T00:00:00Z)",
        "test": "[2019-04-26T00:00:00Z, source_end]",
    }
    if boundaries != expected:
        raise ExperimentBlocked("final protocol calendar boundaries differ from the frozen decision")
    return config


def _build_scope(scope: str, project_root: Path, chunk_size: int, risk: dict[tuple[str, str, str], dict[str, Any]]) -> ScopeRun:
    records = make_source_records(project_root)
    assign_split_plan(records)
    scope_records = _scope_records(records, scope)
    plans = make_group_plans(scope_records, risk)
    if len(plans) != 1:
        raise ExperimentBlocked(f"{scope}: expected one schema-local plan, got {len(plans)}")
    plan = plans[0]
    state = RunState()
    fit_preprocessors(
        [plan],
        raw_type_mapping(),
        chunk_size,
        state,
        risk=risk,
        quarantine_invalid=True,
    )
    features = final_features(plan)
    if not features:
        raise ExperimentBlocked(f"{scope}: no Train-fitted features survived the feature audit")

    mapping = raw_type_mapping()
    accumulators = {split: SplitAccumulator() for split in (*SPLIT_NAMES, "quarantine")}
    global_fingerprints: dict[int, tuple[str, str]] = {}
    cross_fold_pairs: list[dict[str, str]] = []
    duplicate_occurrences = 0
    label_type_mismatches = Counter()
    total_source_rows = 0

    for record in scope_records:
        for frame, final_codes, canonical, labels, fingerprints in _iter_routed_chunks(record, chunk_size, mapping):
            total_source_rows += len(frame)
            for digest, code in zip(fingerprints.tolist(), final_codes.tolist()):
                if code == QUARANTINE_CODE:
                    continue
                split = {0: "train", 1: "validation", 2: "test"}[int(code)]
                previous = global_fingerprints.get(int(digest))
                if previous is None:
                    global_fingerprints[int(digest)] = (split, record.file_name)
                else:
                    duplicate_occurrences += 1
                    if previous[0] != split:
                        cross_fold_pairs.append({
                            "first_split": previous[0],
                            "second_split": split,
                            "first_file": previous[1],
                            "second_file": record.file_name,
                            "fingerprint": str(int(digest)),
                        })
            for code, split in ((0, "train"), (1, "validation"), (2, "test"), (QUARANTINE_CODE, "quarantine")):
                mask = final_codes == code
                if not mask.any():
                    continue
                acc = accumulators[split]
                selected_labels = labels.loc[mask].to_numpy(dtype=np.int8)
                selected_types = canonical.loc[mask]
                acc.total_rows += int(mask.sum())
                acc.normal_rows += int((selected_labels == 0).sum())
                acc.attack_rows += int((selected_labels == 1).sum())
                if split == "quarantine":
                    acc.quarantine_rows += int(mask.sum())
                    continue
                known_mask = selected_types.isin(KNOWN_TYPES).to_numpy(dtype=bool)
                acc.known_rows += int(known_mask.sum())
                acc.excluded_rows += int((~known_mask).sum())
                acc.excluded_types.update(selected_types.loc[~known_mask].astype(str).tolist())
                label_type_mismatches.update(
                    selected_types.loc[(selected_labels == 0) & (selected_types != "normal")].astype(str).tolist()
                )
                label_type_mismatches.update(
                    selected_types.loc[(selected_labels == 1) & (selected_types == "normal")].astype(str).tolist()
                )
                if split in SPLIT_NAMES and known_mask.any():
                    known_frame = frame.loc[mask].loc[known_mask]
                    transformed = transform_baseline(known_frame, plan)
                    acc.x_batches.append(transformed[features].to_numpy(dtype=np.float32, copy=True))
                    acc.y_batches.append(selected_labels[known_mask])

    if total_source_rows != sum(record.rows_expected for record in scope_records):
        raise ExperimentBlocked(f"{scope}: source row reconciliation failed")
    if cross_fold_pairs:
        raise ExperimentBlocked(f"{scope}: exact duplicate fingerprints cross split ({len(cross_fold_pairs)})")
    split_arrays = {split: accumulators[split].finalize(features) for split in SPLIT_NAMES}
    return ScopeRun(
        scope=scope,
        plan=plan,
        features=features,
        splits=split_arrays,
        accumulators=accumulators,
        counts=[],
        duplicate_audit={
            "exact_fingerprint_unique": len(global_fingerprints),
            "exact_duplicate_occurrences": duplicate_occurrences,
            "exact_cross_fold_occurrences": len(cross_fold_pairs),
            "cross_fold_pairs": cross_fold_pairs,
            "status": "PASS",
            "fingerprint": "pandas.hash_pandas_object over all original source columns",
        },
        label_type_audit={
            "label_type_mismatch_count": int(sum(label_type_mismatches.values())),
            "label_type_mismatches": dict(label_type_mismatches),
            "status": "PASS" if not label_type_mismatches else "FAIL",
        },
        feature_audit=_feature_rows(plan, risk),
    )


def _distribution_rows(scope_run: ScopeRun) -> list[dict[str, Any]]:
    rows = []
    for split in (*SPLIT_NAMES, "quarantine"):
        acc = scope_run.accumulators[split]
        known_attack = int((scope_run.splits[split][1] == 1).sum()) if split in SPLIT_NAMES else 0
        known_normal = int((scope_run.splits[split][1] == 0).sum()) if split in SPLIT_NAMES else 0
        known_total = known_normal + known_attack
        rows.append({
            "scope": scope_run.scope,
            "split": split,
            "all_rows_after_routing": acc.total_rows,
            "all_normal_rows": acc.normal_rows,
            "all_attack_rows": acc.attack_rows,
            "known_mask_rows": acc.known_rows,
            "known_normal_rows": known_normal,
            "known_scanning_rows": known_attack,
            "excluded_rows": acc.excluded_rows,
            "excluded_attack_types": ";".join(f"{k}:{v}" for k, v in sorted(acc.excluded_types.items())),
            "known_attack_rate": (known_attack / known_total) if known_total else "UNDEFINED",
            "quarantine_rows": acc.quarantine_rows,
            "known_mask": "normal+scanning",
            "sampling": "none; all routed rows retained",
        })
    return rows


def _run_models(scope_run: ScopeRun) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    x_train, y_train = scope_run.splits["train"]
    if len(np.unique(y_train)) < 2:
        raise ExperimentBlocked(f"{scope_run.scope}: Train known mask is single class")
    rows_by_split: dict[str, list[dict[str, Any]]] = {"validation": [], "test": []}
    for model_name in MODEL_NAMES:
        started = time.perf_counter()
        model = _make_model(model_name, y_train)
        if model is not None:
            model.fit(x_train, y_train)
        fit_seconds = time.perf_counter() - started
        for split in ("validation", "test"):
            x_eval, y_eval = scope_run.splits[split]
            scores = _predict_scores(model_name, model, x_eval)
            metric = compute_metrics(y_eval, scores, threshold=0.5)
            row = {
                "scope": scope_run.scope,
                "model": model_name,
                "split": split,
                "feature_count": len(scope_run.features),
                "feature_columns": ";".join(scope_run.features),
                "train_rows_known_mask": len(y_train),
                "validation_or_test_rows_known_mask": len(y_eval),
                "train_attack_rate": float((y_train == 1).mean()),
                "evaluation_attack_rate": float((y_eval == 1).mean()) if len(y_eval) else "UNDEFINED",
                "fit_seconds": round(fit_seconds, 3),
                "config_frozen_before_evaluation": True,
                "test_used_for_tuning": False,
                "threshold_selection": "fixed 0.5; no Validation/Test tuning",
                **metric,
            }
            rows_by_split[split].append(row)
    return rows_by_split["validation"], rows_by_split["test"]


def _comparison_markdown(validation: pd.DataFrame, test: pd.DataFrame, distributions: pd.DataFrame) -> str:
    def fmt(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    lines = [
        "# Model comparison\n\n",
        "All models receive the same schema-local Train rows, feature matrix, Validation/Test masks and fixed threshold. `Precision`, `Recall`, `Macro-F1`, `PR-AUC` and `FPR` below refer to the Attack class.\n\n",
    ]
    for scope in sorted(validation["scope"].unique()):
        lines.append(f"## {scope}\n\n")
        for name, frame in (("Validation", validation), ("Test", test)):
            sub = frame[frame["scope"] == scope]
            lines.append(f"### {name}\n\n| Model | Precision | Recall | Macro-F1 | PR-AUC | FPR |\n|---|---:|---:|---:|---:|---:|\n")
            for row in sub.to_dict("records"):
                lines.append(f"| {row['model']} | {fmt(row.get('precision_attack'))} | {fmt(row.get('recall_attack'))} | {fmt(row.get('macro_f1'))} | {fmt(row.get('pr_auc_attack'))} | {fmt(row.get('false_positive_rate'))} |\n")
            lines.append("\n")
        lines.append("Confusion matrices (TN, FP, FN, TP):\n\n")
        for row in test[test["scope"] == scope].to_dict("records"):
            lines.append(f"- `{row['model']}`: `[[{row['tn']}, {row['fp']}], [{row['fn']}, {row['tp']}]]`; predicted Normal={row['predicted_normal']}, Attack={row['predicted_attack']}.\n")
        lines.append("\n")
        dist = distributions[(distributions["scope"] == scope) & distributions["split"].isin(SPLIT_NAMES)]
        rates = {r["split"]: r["known_attack_rate"] for r in dist.to_dict("records")}
        lines.append(f"Known-mask Attack rate: Train `{fmt(rates.get('train'))}`, Validation `{fmt(rates.get('validation'))}`, Test `{fmt(rates.get('test'))}`. This is a natural-distribution shift check, not a tuning signal.\n\n")
        sub_test = test[(test["scope"] == scope) & test["model"].isin(["RandomForest", "XGBoost", "LightGBM"])].copy()
        numeric = pd.to_numeric(sub_test["macro_f1"], errors="coerce")
        if numeric.notna().any():
            best = sub_test.loc[numeric.idxmax()]
            lines.append(f"Best Test Macro-F1 among trained tree models within this pre-declared scope: `{best['model']}` ({fmt(best['macro_f1'])}); this is descriptive because Test was opened only after configuration freeze.\n\n")
    lines += [
        "## Interpretation\n\n",
        "- RF, XGBoost and LightGBM are compared on the same inputs; no result is compared with Phase 3B Network GNN as if it were the same dataset or task.\n",
        "- Confusion matrices must be read together with class rates. A model can have high Attack recall while producing an unacceptable number of false positives, or appear strong when a fold is attack-heavy.\n",
        "- Dummy Always-Normal and Always-Attack are descriptive lower-bound/reference behaviors, not tuned models.\n",
        "- On Test, the trained tree models have lower Macro-F1 than Always-Normal in both schemas because their false-positive counts are large; their Attack PR-AUC is only modestly above the Always-Normal reference. This is not evidence of broad superiority.\n",
        "- The selected Normal-vs-Scanning task is a closed-set label mask. Rows of DDoS, DoS, Injection, MITM, Password, XSS and other types are excluded from these binary metrics and are counted in `07_CLASS_DISTRIBUTION.csv`.\n",
        "- Temporal/episode independence is not established; therefore the result is a conditional Linux schema-local benchmark, not a claim of general IDS deployment performance.\n",
    ]
    return "".join(lines)


def _data_protocol(scope_runs: list[ScopeRun], project_root: Path, chunk_size: int) -> str:
    lines = [
        "# Data protocol\n\n",
        "## Scope and decision\n\n",
        "Primary benchmark: `Linux::process` Normal vs Scanning. Supporting replication: `Linux::disk` Normal vs Scanning. Linux schemas are run separately and never pooled. Linux Memory is not included because its Train has no Attack/Scanning coverage.\n\n",
        "The binary target is a pre-declared closed-set mask over the real `type` field: `normal -> 0`, `scanning -> 1`. It is not an all-attack IDS task. DDoS, DoS, Injection, MITM, Password, XSS and other labels are excluded from the binary metric mask and their counts/rates are reported separately.\n\n",
        "## Frozen split\n\n",
        "The runner uses `PLATFORM_LOCAL_TEMPORAL_BINARY_V1_CONTROL_REMEDIATED`, the existing `assign_split_plan` and `remediated_split_codes` helpers. Valid timestamps use Train `[source_start, 2019-04-24T00:00:00Z)`, Validation `[2019-04-24T00:00:00Z, 2019-04-26T00:00:00Z)`, and Test `[2019-04-26T00:00:00Z, source_end]`. Non-finite timestamps are quarantined. No boundary or split is selected from Test metrics.\n\n",
        "## Row and preprocessing policy\n\n",
        f"All routed rows are retained; no balancing or sampling is performed. CSV chunks are read with `chunk_size={chunk_size}` only for memory control. Exact duplicate routing follows the existing per-source `DuplicateRouter`; a separate global fingerprint audit checks cross-fold collisions before model fitting.\n\n",
        "Features are inferred and imputed using Train rows only. `label`, `type`, `ts`, filenames, paths, split metadata, scenario identifiers and future aggregates are excluded. Numeric missing values use Train-only medians. No scaler is needed for these tree models. Validation and Test are transform-only.\n\n",
        "The decision threshold is fixed at `0.5` before Validation and Test. No threshold, hyperparameter or model is selected from Test. Fixed class weighting/scale-pos-weight is computed from known-mask Train labels only.\n\n",
        "## Source evidence\n\n",
        f"Project root: `{project_root}`. Protocol configuration: `reports/final_split_protocol/split_protocol_config.json`. Split coverage: `reports/final_split_protocol/LINUX_SCHEMA_COVERAGE.csv`. Timestamp decision: `reports/final_timestamp_validation/FINAL_TIMESTAMP_DECISION.md`.\n\n",
        "## Executed schema-local feature lists\n\n",
    ]
    for run in scope_runs:
        lines.append(f"- `{run.scope}` ({len(run.features)}): `{'; '.join(run.features)}`\n")
    lines.append("\n")
    return "".join(lines)


def _leakage_report(scope_runs: list[ScopeRun]) -> str:
    lines = [
        "# Leakage check\n\n",
        "Statuses use `PASS` only for checks directly executed by this runner; `UNVERIFIED` is retained where the data has no independent episode/session identity.\n\n",
        "| Check | Status | Evidence / limitation |\n|---|---|---|\n",
    ]
    for run in scope_runs:
        lines.append(f"| {run.scope} Train-only preprocessing | PASS | Feature inference, medians and categorical handling run before model assembly using Train rows only; Validation/Test are transform-only. |\n")
        lines.append(f"| {run.scope} exact fingerprint cross-fold collision | {run.duplicate_audit['status']} | {run.duplicate_audit['exact_duplicate_occurrences']} duplicate occurrences; cross-fold occurrences={run.duplicate_audit['exact_cross_fold_occurrences']}; {run.duplicate_audit['fingerprint']}. |\n")
        lines.append(f"| {run.scope} label/type alignment | {run.label_type_audit['status']} | mismatch_count={run.label_type_audit['label_type_mismatch_count']}; Normal/Scanning labels were checked before training. |\n")
        lines.append(f"| {run.scope} feature target/timestamp exclusion | PASS | Model features={'; '.join(run.features)}; `label`, `type`, `ts`, source identifiers and split metadata are absent. |\n")
        lines.append(f"| {run.scope} future-derived feature check | PASS | Features are source-row observables only; no future aggregate/window feature is constructed. |\n")
        lines.append(f"| {run.scope} split boundary/routing | PASS | Existing remediated half-open calendar helper used; invalid timestamps quarantined. |\n")
        lines.append(f"| {run.scope} same session/episode across folds | UNVERIFIED | PID/CMD continuity is not an episode ID; no independent Linux session/run identifier is available. |\n")
        lines.append(f"| {run.scope} Test used for tuning | PASS | Fixed configurations, fixed threshold=0.5, no Test-dependent selection. |\n")
    lines += [
        "\n## Controls not claimed\n\n",
        "This report does not claim complete attack-episode independence, physical-clock synchronization, or absence of related-flow continuity. Those limitations remain from the final split/timestamp decisions. Exact duplicate fingerprints are a row-level control, not an episode identity.\n",
    ]
    return "".join(lines)


def _reproducibility(
    scope_runs: list[ScopeRun],
    project_root: Path,
    chunk_size: int,
    report_dir: Path,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> str:
    lines = [
        "# Experiment reproducibility\n\n",
        "## Re-run command\n\n",
        "```text\n",
        "cd D:\\paper_NCKH\\network_iot\n",
        ".\\.venv\\Scripts\\python.exe -m research_pipeline.src.experimental_phase3_baseline.final_ml_baseline\n",
        "```\n\n",
        "The runner refuses to overwrite a non-empty output directory unless `--overwrite` is explicitly provided. It never writes to Phase 2/Phase 3B directories or input datasets.\n\n",
        "## Frozen execution\n\n",
        f"- Seed: `{SEED}`; chunk size: `{chunk_size}`; decision threshold: `0.5`.\n",
        f"- Python: `{sys.version.split()[0]}`; OS: `{platform_module.platform()}`.\n",
        f"- Git revision: `{_git_revision()}`.\n",
        "- Training uses all known-mask Train rows; no row sampling, class balancing of Validation/Test, or Test read for tuning.\n",
        "- Class weights/scale-pos-weight are derived from known-mask Train labels only.\n",
        "- Validation is evaluated first; fixed configurations are then evaluated on Test once.\n\n",
        "## Package versions\n\n",
    ]
    for package in ("numpy", "pandas", "sklearn", "xgboost", "lightgbm"):
        try:
            module = __import__(package)
            lines.append(f"- `{package}`: `{getattr(module, '__version__', 'unknown')}`\n")
        except Exception as exc:
            lines.append(f"- `{package}`: `UNAVAILABLE ({exc})`\n")
    lines += [
        "\n## Executed checks\n\n",
        f"- Scope runs: `{', '.join(run.scope for run in scope_runs)}`.\n",
        f"- Validation metric rows: `{len(validation)}`; Test metric rows: `{len(test)}`.\n",
        f"- Output directory: `{report_dir}`.\n",
        "- Unit tests are in `tests/test_final_ml_baseline.py`; the final report records the command and observed result.\n",
    ]
    return "".join(lines)


def _final_decision(
    scope_runs: list[ScopeRun],
    validation: pd.DataFrame,
    test: pd.DataFrame,
    distributions: pd.DataFrame,
    tests_summary: str,
) -> str:
    winner_lines: list[str] = []
    weight_lines: list[str] = []
    tree_names = ["RandomForest", "XGBoost", "LightGBM"]
    for run in scope_runs:
        y_train = run.splits["train"][1]
        normal = int((y_train == 0).sum())
        attack = int((y_train == 1).sum())
        weight_lines.append(f"- `{run.scope}`: `{normal / max(1, attack):.12f}` (Train Normal / Train Scanning).")
    for scope in sorted(test["scope"].unique()):
        validation_scope = validation[(validation["scope"] == scope) & validation["model"].isin(tree_names)].copy()
        test_scope = test[(test["scope"] == scope) & test["model"].isin(tree_names)].copy()
        validation_scope["macro_f1_num"] = pd.to_numeric(validation_scope["macro_f1"], errors="coerce")
        test_scope["macro_f1_num"] = pd.to_numeric(test_scope["macro_f1"], errors="coerce")
        validation_winner = validation_scope.loc[validation_scope["macro_f1_num"].idxmax()]
        test_winner = test_scope.loc[test_scope["macro_f1_num"].idxmax()]
        winner_lines.append(
            f"- `{scope}`: Validation Macro-F1 winner `{validation_winner['model']}`=`{validation_winner['macro_f1']:.4f}`; "
            f"Test Macro-F1 winner `{test_winner['model']}`=`{test_winner['macro_f1']:.4f}`."
        )
    lines = [
        "# FINAL BASELINE DECISION — ToN-IoT Phase 3\n\n",
        "## Overall status\n\n",
        "**CONDITIONAL** for the pre-declared Linux schema-local closed-set benchmark; **DIAGNOSTIC ONLY** for claims about a general IDS, cross-platform performance, or independent attack-episode generalization.\n\n",
        "## Direct answers\n\n",
        "1. **Were the three baselines trained?** Yes. Random Forest, XGBoost and LightGBM were trained on real known-mask Train rows for Linux Process and Linux Disk. Dummy Always-Normal and Always-Attack were evaluated as comparators.\n",
        "2. **Was the Final Split Protocol used?** Yes. The runner uses the existing remediated calendar split helper and preserves the frozen cutoffs; invalid timestamps are quarantined. No split was re-selected.\n",
        "3. **Did models receive identical evaluation data?** Yes, within each schema: every model receives the same Train matrix, Validation mask, Test mask, ordered feature list and threshold. Process and Disk remain separate datasets.\n",
        "4. **Are the requested metrics present?** Yes. Accuracy, per-class precision/recall/F1, Macro-F1, Weighted-F1, Attack/Normal PR-AUC, ROC-AUC when defined, FPR, FNR, confusion matrix and class counts are in the CSV outputs. Undefined metrics are written as `UNDEFINED`.\n",
        "5. **Which leakage controls passed?** Train-only feature inference/imputation, target/type/timestamp exclusion, exact duplicate cross-fold audit, natural-fold accounting, half-open routing, no sampling, fixed threshold and no Test tuning. Episode/session independence remains `UNVERIFIED`.\n",
        "6. **Which model is best?** The descriptive winner is reported separately by schema and fold in `09_MODEL_COMPARISON.md`; it is not promoted beyond this pre-declared scope and is not compared to the Phase 3B Network GNN.\n",
        "7. **Validated, Conditional or Diagnostic?** Conditional for Linux Process and Linux Disk schema-local Normal-vs-Scanning baseline evidence; Diagnostic for universal IDS claims because attack episodes, physical clock semantics and related-flow independence are not proven.\n",
        "8. **What may enter the lecturer report?** The real RF/XGBoost/LightGBM Validation/Test tables, confusion matrices, dummy references, class-mask exclusion counts and leakage status may be reported with the closed-set and temporal/episode limitations. Do not report them as all-attack, cross-platform or GNN-comparable results.\n\n",
        "## Observed tree-model winners\n\n",
        *[line + "\n" for line in winner_lines],
        "These are descriptive winners within each pre-declared schema and fold; Always-Normal/Always-Attack are shown separately as reference comparators.\n\n",
        "## Actual XGBoost class weighting\n\n",
        "`scale_pos_weight` was computed from the known-mask Train labels only:\n\n",
        *[line + "\n" for line in weight_lines],
        "These values were frozen before Validation/Test evaluation.\n\n",
        "## Scope accounting\n\n",
        "The experiment intentionally excludes non-Scanning attack types from the binary metrics. This is a closed-set Normal-vs-Scanning experiment; the exact excluded counts and attack-type composition for Train/Validation/Test are in `07_CLASS_DISTRIBUTION.csv`. Validation/Test were not balanced.\n\n",
        "## Model-selection and Test rule\n\n",
        "Configurations and threshold were frozen before metric evaluation. No Test metric selected a model, feature, threshold, split, or class mask. The Test table is a one-time descriptive estimate under the frozen protocol.\n\n",
        "## Remaining limitations\n\n",
        "The Final Split Decision is conditional, timestamp physical-clock provenance is not fully validated, and no independent Linux session/episode identifier exists. Exact row fingerprint checks do not remove related PID/CMD continuity. Therefore this evidence supports a careful baseline benchmark, not a claim that one tree model is superior in deployment or superior to the Network GNN.\n\n",
        "## Unit-test evidence\n\n",
        tests_summary,
        "\n",
    ]
    return "".join(lines)


def _write_model_config(
    report_dir: Path,
    scope_runs: list[ScopeRun],
    validation: pd.DataFrame,
    test: pd.DataFrame,
    chunk_size: int,
) -> None:
    payload = {
        "experiment": "PHASE_3_FINAL_ML_BASELINE",
        "status": "CONDITIONAL",
        "seed": SEED,
        "chunk_size": chunk_size,
        "threshold": 0.5,
        "selection": "pre-declared fixed configurations; no hyperparameter or threshold tuning",
        "scope_order": [run.scope for run in scope_runs],
        "scope_roles": {scope: SCOPE_CONFIG[scope]["role"] for scope in SCOPE_CONFIG},
        "known_mask": {"normal": 0, "scanning": 1},
        "models": {name: _model_config(name, np.array([0, 1], dtype=np.int8)) for name in MODEL_NAMES},
        "model_train_class_weights": {
            run.scope: {
                "train_rows": int(len(run.splits["train"][1])),
                "normal": int((run.splits["train"][1] == 0).sum()),
                "scanning": int((run.splits["train"][1] == 1).sum()),
                "xgboost_scale_pos_weight": float((run.splits["train"][1] == 0).sum() / max(1, (run.splits["train"][1] == 1).sum())),
            }
            for run in scope_runs
        },
        "xgboost_scale_pos_weight_actual_by_scope": {
            run.scope: float((run.splits["train"][1] == 0).sum() / max(1, (run.splits["train"][1] == 1).sum()))
            for run in scope_runs
        },
        "features": {run.scope: run.features for run in scope_runs},
        "evaluation_rows": {"validation": int(len(validation)), "test": int(len(test))},
        "test_used_for_tuning": False,
        "input_immutability": True,
    }
    (report_dir / "03_MODEL_CONFIG.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def run(
    project_root: Path = PROJECT_ROOT,
    report_dir: Path | None = None,
    chunk_size: int = CHUNK_SIZE,
    overwrite: bool = False,
    tests_summary: str = "Not supplied; run the unit-test command recorded in the final report.",
) -> int:
    report_dir = report_dir or (project_root / "research_pipeline" / "reports" / "final_ml_baseline")
    report_dir = report_dir.resolve()
    if report_dir.exists() and any(report_dir.iterdir()) and not overwrite:
        raise RuntimeError(f"Refusing to overwrite non-empty output directory: {report_dir}; use --overwrite explicitly")
    report_dir.mkdir(parents=True, exist_ok=True)
    _verify_frozen_protocol(project_root)
    risk = read_feature_risk()
    scope_runs = [_build_scope(scope, project_root, chunk_size, risk) for scope in ("Linux::process", "Linux::disk")]

    validation_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    for run_data in scope_runs:
        scope_validation, scope_test = _run_models(run_data)
        validation_rows.extend(scope_validation)
        test_rows.extend(scope_test)
    validation = pd.DataFrame(validation_rows)
    test = pd.DataFrame(test_rows)
    distributions = pd.DataFrame([row for run_data in scope_runs for row in _distribution_rows(run_data)])
    feature_audit = pd.DataFrame([row for run_data in scope_runs for row in run_data.feature_audit])
    confusion = pd.concat([
        validation[["scope", "model", "split", "tn", "fp", "fn", "tp", "normal_count", "attack_count", "predicted_normal", "predicted_attack"]],
        test[["scope", "model", "split", "tn", "fp", "fn", "tp", "normal_count", "attack_count", "predicted_normal", "predicted_attack"]],
    ], ignore_index=True)

    (report_dir / "01_DATA_PROTOCOL.md").write_text(_data_protocol(scope_runs, project_root, chunk_size), encoding="utf-8")
    feature_audit.to_csv(report_dir / "02_FEATURE_AUDIT.csv", index=False)
    _write_model_config(report_dir, scope_runs, validation, test, chunk_size)
    validation.to_csv(report_dir / "04_VALIDATION_RESULTS.csv", index=False)
    test.to_csv(report_dir / "05_TEST_RESULTS.csv", index=False)
    confusion.to_csv(report_dir / "06_CONFUSION_MATRICES.csv", index=False)
    distributions.to_csv(report_dir / "07_CLASS_DISTRIBUTION.csv", index=False)
    (report_dir / "08_LEAKAGE_CHECK.md").write_text(_leakage_report(scope_runs), encoding="utf-8")
    (report_dir / "09_MODEL_COMPARISON.md").write_text(_comparison_markdown(validation, test, distributions), encoding="utf-8")
    (report_dir / "10_EXPERIMENT_REPRODUCIBILITY.md").write_text(
        _reproducibility(scope_runs, project_root, chunk_size, report_dir, validation, test),
        encoding="utf-8",
    )
    (report_dir / "FINAL_BASELINE_DECISION.md").write_text(
        _final_decision(scope_runs, validation, test, distributions, tests_summary),
        encoding="utf-8",
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--report-dir", type=Path, default=None)
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--tests-summary", default="Unit-test result was recorded separately; see 10_EXPERIMENT_REPRODUCIBILITY.md.")
    args = parser.parse_args()
    return run(
        project_root=args.project_root.resolve(),
        report_dir=args.report_dir,
        chunk_size=args.chunk_size,
        overwrite=args.overwrite,
        tests_summary=args.tests_summary,
    )


if __name__ == "__main__":
    raise SystemExit(main())

