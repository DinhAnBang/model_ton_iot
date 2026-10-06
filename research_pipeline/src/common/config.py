"""One place for Phase-1 constants and the versioned label mapping."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Any

import pandas as pd

from .paths import CONFIG_DIR


PLATFORM_ORDER = ["Network", "IoT", "Linux", "Windows"]
NORMAL_LABEL = 0
ATTACK_LABEL = 1

SPLITS = ("train", "validation", "test")
SPLIT_CODE = {name: index for index, name in enumerate(SPLITS)}
CODE_SPLIT = {index: name for name, index in SPLIT_CODE.items()}
DEFAULT_CHUNK_SIZE = 250_000
MAX_CATEGORICAL_LEVELS = 64
MEDIAN_SAMPLE_SIZE = 50_000
UNKNOWN_CANONICAL = "UNKNOWN"
CORRECTIVE_TRAIN_END = pd.Timestamp("2019-04-24", tz="UTC").timestamp()
CORRECTIVE_VALIDATION_END = pd.Timestamp("2019-04-26", tz="UTC").timestamp()


@lru_cache(maxsize=1)
def load_label_mapping() -> dict[str, Any]:
    """Read the committed raw-to-canonical mapping without recomputing it."""
    path = CONFIG_DIR / "label_mapping.json"
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def raw_type_mapping() -> dict[str, str]:
    """Map exact source type spellings to canonical attack names."""
    return {
        str(row["raw_type"]): str(row["canonical_type"])
        for row in load_label_mapping()["type_mapping"]
    }


def normalized_type_mapping() -> dict[str, str]:
    """Map case-insensitive alphanumeric aliases used by validation."""
    return {
        re.sub(r"[^a-z0-9]+", "", raw.lower()): canonical
        for raw, canonical in raw_type_mapping().items()
    }


def canonical_type(value: Any) -> str:
    """Resolve a raw attack name; unknown names remain unclassified."""
    normalized = re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())
    return normalized_type_mapping().get(normalized, "")


CANONICAL_TYPES = list(load_label_mapping()["canonical_vocabulary"])
ATTACK_CATEGORIES = [name for name in CANONICAL_TYPES if name != "normal"]
