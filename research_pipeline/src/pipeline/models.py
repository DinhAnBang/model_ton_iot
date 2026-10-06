"""Models for the Phase-1 research pipeline."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SourceRecord:
    platform: str
    file_name: str
    file_path: Path
    relative_path: str
    columns: list[str]
    rows_expected: int
    timestamp_column: str
    timestamp_min: str
    timestamp_max: str
    duplicate_count: int = 0
    group_id: str = ""
    split_mode: str = ""
    fixed_split: str | None = None
    split_reason: str = ""
    block_min: float | None = None
    block_cut_1: float | None = None
    block_cut_2: float | None = None
    row_number: int = 0


@dataclass
class FeatureSpec:
    name: str
    kind: str
    risk_classification: str
    reason: str
    train_missing: int = 0
    train_nonmissing: int = 0
    train_unique: int = 0
    median: float | None = None
    categories: list[str] = field(default_factory=list)
    median_sample: list[float] = field(default_factory=list)
    excluded_after_fit: bool = False
    exclusion_reason: str = ""


@dataclass
class GroupPlan:
    group_id: str
    platform: str
    records: list[SourceRecord]
    feature_specs: dict[str, FeatureSpec]
    feature_order: list[str]
    source_columns: list[str]
    preprocessor_name: str


@dataclass(frozen=True)
class ClassCoverage:
    """Binary label coverage for one platform in the executed split."""

    platform: str
    train_normal: bool
    train_attack: bool
    validation_normal: bool
    validation_attack: bool
    test_normal: bool
    test_attack: bool

    @property
    def validation_both(self) -> bool:
        return self.validation_normal and self.validation_attack

    @property
    def test_both(self) -> bool:
        return self.test_normal and self.test_attack

    @property
    def eligible(self) -> bool:
        return self.train_normal and self.train_attack and self.test_both

    @property
    def missing_requirements(self) -> list[str]:
        missing = []
        if not self.train_normal:
            missing.append("train normal")
        if not self.train_attack:
            missing.append("train attack")
        if not self.test_normal:
            missing.append("test normal")
        if not self.test_attack:
            missing.append("test attack")
        return missing


@dataclass
class RunState:
    source_rows: Counter = field(default_factory=Counter)
    output_rows: Counter = field(default_factory=Counter)
    output_normal: Counter = field(default_factory=Counter)
    output_attack: Counter = field(default_factory=Counter)
    output_attack_types: defaultdict = field(default_factory=lambda: defaultdict(Counter))
    time_min: dict[tuple[str, str, str], float] = field(default_factory=dict)
    time_max: dict[tuple[str, str, str], float] = field(default_factory=dict)
    invalid_labels: Counter = field(default_factory=Counter)
    invalid_timestamps: Counter = field(default_factory=Counter)
    unknown_types: Counter = field(default_factory=Counter)
    duplicate_seen: Counter = field(default_factory=Counter)
    duplicate_routed: Counter = field(default_factory=Counter)
    duplicate_cross_split: Counter = field(default_factory=Counter)
    block_manifest_rows: list[dict[str, Any]] = field(default_factory=list)
    source_split_sets: defaultdict = field(default_factory=lambda: defaultdict(set))
    scenario_split_sets: defaultdict = field(default_factory=lambda: defaultdict(set))
    baseline_paths: set[str] = field(default_factory=set)
    semantic_paths: set[str] = field(default_factory=set)
    validation_rows: list[dict[str, Any]] = field(default_factory=list)
