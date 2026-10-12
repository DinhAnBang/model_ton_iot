"""Train-only numerical preprocessing for the Phase 3B Network pilot."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TrainOnlyScaler:
    feature_columns: tuple[str, ...]
    medians: tuple[float, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    epsilon: float = 1e-8

    def transform(self, encoded: np.ndarray) -> np.ndarray:
        """Transform [raw value columns | missingness bits] with frozen Train stats."""
        width = len(self.feature_columns)
        raw = np.asarray(encoded[:, :width], dtype=np.float32).copy()
        missing = np.asarray(encoded[:, width:], dtype=np.float32).copy()
        medians = np.asarray(self.medians, dtype=np.float32)
        means = np.asarray(self.means, dtype=np.float32)
        scales = np.asarray(self.scales, dtype=np.float32)
        # The fill below uses only frozen Train medians.
        for index in range(width):
            raw[missing[:, index].astype(bool), index] = medians[index]
        raw = (raw - means) / scales
        output = np.concatenate([raw, missing], axis=1)
        return output.astype(np.float32, copy=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_columns": list(self.feature_columns),
            "medians": list(self.medians),
            "means": list(self.means),
            "scales": list(self.scales),
            "epsilon": self.epsilon,
            "fit_scope": "Network Train sample only",
        }


def fit_train_scaler(encoded_train: np.ndarray, feature_columns: list[str]) -> TrainOnlyScaler:
    """Fit median fill and standardization using only the Train matrix."""
    width = len(feature_columns)
    values = np.asarray(encoded_train[:, :width], dtype=np.float64)
    missing = np.asarray(encoded_train[:, width:], dtype=bool)
    valid = (~missing) & np.isfinite(values)
    medians: list[float] = []
    means: list[float] = []
    scales: list[float] = []
    for index in range(width):
        column = values[valid[:, index], index]
        if column.size == 0:
            median = mean = 0.0
            scale = 1.0
        else:
            median = float(np.median(column))
            filled = np.where(valid[:, index], values[:, index], median)
            mean = float(np.mean(filled))
            scale = float(np.std(filled))
            if not np.isfinite(scale) or scale < 1e-8:
                scale = 1.0
        medians.append(median)
        means.append(mean)
        scales.append(scale)
    return TrainOnlyScaler(tuple(feature_columns), tuple(medians), tuple(means), tuple(scales))


def transform_graph_features(graph, scaler: TrainOnlyScaler) -> None:
    graph.event_x = scaler.transform(graph.event_x)


def feature_schema_fingerprint(feature_columns: list[str], provenance) -> str:
    """Hash schema and non-label provenance for reproducibility."""
    rows = provenance[["event_id", "source_file", "source_row", "timestamp"]].copy()
    payload = {
        "feature_columns": list(feature_columns),
        "provenance_sha256": hashlib.sha256(
            rows.to_csv(index=False).encode("utf-8")
        ).hexdigest(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def preprocessing_diagnostics(encoded_train: np.ndarray, labels: np.ndarray, feature_columns: list[str]) -> dict[str, Any]:
    width = len(feature_columns)
    values = np.asarray(encoded_train[:, :width], dtype=np.float64)
    missing = np.asarray(encoded_train[:, width:], dtype=bool)
    finite = bool(np.isfinite(values).all())
    max_abs = float(np.nanmax(np.abs(values))) if values.size else 0.0
    missing_rows: list[dict[str, Any]] = []
    for index, name in enumerate(feature_columns):
        normal_mask = labels == 0
        attack_mask = labels == 1
        normal_rate = float(missing[normal_mask, index].mean()) if normal_mask.any() else float("nan")
        attack_rate = float(missing[attack_mask, index].mean()) if attack_mask.any() else float("nan")
        missing_rows.append({
            "feature": name,
            "normal_missing_rate": normal_rate,
            "attack_missing_rate": attack_rate,
            "absolute_rate_difference": abs(normal_rate - attack_rate) if np.isfinite(normal_rate) and np.isfinite(attack_rate) else float("nan"),
            "perfect_label_match": bool(np.array_equal(missing[:, index].astype(np.int8), labels.astype(np.int8))),
        })
    return {
        "finite_raw_values": finite,
        "max_abs_raw_value": max_abs,
        "missingness_rows": missing_rows,
        "target_columns_in_features": [],
        "status": "PASS" if finite and not any(row["perfect_label_match"] for row in missing_rows) else "FAIL",
    }

