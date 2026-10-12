"""Split data for the Phase-1 research pipeline."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from ..common.config import (
    CORRECTIVE_TRAIN_END,
    CORRECTIVE_VALIDATION_END,
    QUARANTINE_CODE,
    SPLITS,
    SPLIT_CODE,
)
from ..common.errors import PipelineError
from .models import ClassCoverage, SourceRecord
from .normalization import parse_time_bound


def assign_split_plan(records: list[SourceRecord]) -> None:
    grouped: defaultdict[str, list[SourceRecord]] = defaultdict(list)
    for rec in records:
        grouped[rec.platform].append(rec)

    # Network has enough provenance and a compatible schema across source
    # files for a source-file holdout with one shared train-only preprocessor.
    network_files = sorted(
        grouped["Network"],
        key=lambda rec: (parse_time_bound(rec.timestamp_min, "Network"), rec.file_name),
    )
    if network_files:
        if len(network_files) < 3:
            raise PipelineError("Network: source holdout requires at least three files")
        train_files = max(1, int(round(len(network_files) * 0.70)))
        validation_files = max(1, int(round(len(network_files) * 0.15)))
        if train_files + validation_files >= len(network_files):
            validation_files = 1
            train_files = len(network_files) - 2
        for index, rec in enumerate(network_files):
            if index < train_files:
                split = "train"
            elif index < train_files + validation_files:
                split = "validation"
            else:
                split = "test"
            rec.split_mode = "source_file_holdout"
            rec.fixed_split = split
            rec.split_reason = "scenario/source holdout by deterministic chronological source-file order"

    # Corrective usability pass: the audited temporal coverage shows a mixed
    # normal/attack day on 2019-04-23, mixed validation days on 2019-04-24/25,
    # and mixed normal/attack deployment days from 2019-04-26 onward. Keep
    # complete calendar blocks together; do not split rows randomly or by label.
    for platform in ("IoT", "Linux", "Windows"):
        for rec in grouped[platform]:
            rec.split_mode = "chronological_calendar_block"
            if platform == "IoT":
                rec.split_reason = "corrective chronological calendar blocks; profile schemas stay separate; no random row split"
            elif platform == "Linux":
                rec.split_reason = "corrective chronological calendar blocks; Linux scenario provenance remains weak; no random row split"
            else:
                rec.split_reason = "corrective chronological calendar blocks; two Windows source files prevent full source holdout"
            rec.block_min = parse_time_bound(rec.timestamp_min, rec.platform)
            rec.block_cut_1 = CORRECTIVE_TRAIN_END
            rec.block_cut_2 = CORRECTIVE_VALIDATION_END


def base_split_codes(times: pd.Series, rec: SourceRecord) -> np.ndarray:
    if rec.split_mode == "source_file_holdout":
        if rec.fixed_split is None:
            raise PipelineError(f"{rec.file_name}: missing fixed source split")
        return np.full(len(times), SPLIT_CODE[rec.fixed_split], dtype=np.int8)
    if rec.split_mode != "chronological_calendar_block":
        raise PipelineError(f"{rec.file_name}: unsupported split mode {rec.split_mode!r}")
    if rec.block_cut_1 is None or rec.block_cut_2 is None:
        raise PipelineError(f"{rec.file_name}: calendar split boundaries are missing")
    values = times.to_numpy(dtype="float64", copy=False)
    # Missing event times are not imputed and are placed in an explicit,
    # deterministic train bucket because no chronological cutoff can be
    # justified for them.  The semantic view still exposes the missing value.
    codes = np.full(len(values), SPLIT_CODE["train"], dtype=np.int8)
    valid = np.isfinite(values)
    codes[valid] = SPLIT_CODE["test"]
    codes[valid & (values < rec.block_cut_2)] = SPLIT_CODE["validation"]
    codes[valid & (values < rec.block_cut_1)] = SPLIT_CODE["train"]
    return codes


def remediated_split_codes(times: pd.Series, rec: SourceRecord) -> np.ndarray:
    """Apply the unchanged V1 cutoffs, but quarantine unknown event times.

    This function intentionally does not alter ``base_split_codes``.  V1 can
    therefore still be reproduced, while the remediation path cannot claim a
    chronological placement for rows whose timestamp was not parsed.
    """
    codes = base_split_codes(times, rec)
    values = times.to_numpy(dtype="float64", copy=False)
    codes[~np.isfinite(values)] = QUARANTINE_CODE
    return codes


def calendar_split_description() -> str:
    """Describe the configured complete-day boundaries in generated reports."""
    validation_start = pd.Timestamp(CORRECTIVE_TRAIN_END, unit="s", tz="UTC")
    test_start = pd.Timestamp(CORRECTIVE_VALIDATION_END, unit="s", tz="UTC")
    if test_start <= validation_start:
        raise PipelineError("Calendar test boundary must be later than validation start")
    train_last = (validation_start - pd.Timedelta(days=1)).date()
    validation_last = (test_start - pd.Timedelta(days=1)).date()
    return (
        f"train through {train_last}, validation on "
        f"{validation_start.date()} through {validation_last}, "
        f"test from {test_start.date()} onward"
    )


def platform_class_coverage(distribution: pd.DataFrame) -> list[ClassCoverage]:
    """Compute report eligibility once, using train/test as required folds."""
    coverage = []
    for platform, rows in distribution.groupby("platform", sort=True):
        counts = (
            rows.groupby("split")[["normal_count", "attack_count"]]
            .sum()
            .reindex(SPLITS, fill_value=0)
        )
        coverage.append(ClassCoverage(
            platform=str(platform),
            train_normal=bool(counts.loc["train", "normal_count"] > 0),
            train_attack=bool(counts.loc["train", "attack_count"] > 0),
            validation_normal=bool(counts.loc["validation", "normal_count"] > 0),
            validation_attack=bool(counts.loc["validation", "attack_count"] > 0),
            test_normal=bool(counts.loc["test", "normal_count"] > 0),
            test_attack=bool(counts.loc["test", "attack_count"] > 0),
        ))
    return coverage


def scenario_id_series(times: pd.Series, rec: SourceRecord) -> pd.Series:
    """Return the actual split unit ID for semantic/provenance validation."""
    values = times.to_numpy(dtype="float64", copy=False)
    if rec.split_mode == "source_file_holdout":
        return pd.Series(
            f"SOURCE::{rec.platform}::{rec.file_name}",
            index=times.index,
            dtype="string",
        )
    valid = np.isfinite(values)
    ids = np.full(len(values), f"TIMEBLOCK::{rec.platform}::MISSING_TIME", dtype=object)
    day = np.floor(values[valid] / 86_400).astype("int64")
    ids[valid] = np.char.add(
        f"TIMEBLOCK::{rec.platform}::",
        day.astype(str),
    )
    return pd.Series(ids, index=times.index, dtype="string")
