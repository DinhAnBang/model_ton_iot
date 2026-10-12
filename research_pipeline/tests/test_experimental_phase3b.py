"""Unit tests for Phase 3B Train-only preprocessing controls."""

from __future__ import annotations

import numpy as np
import pandas as pd

from research_pipeline.src.experimental_phase3b.preprocessing import (
    feature_schema_fingerprint,
    fit_train_scaler,
    preprocessing_diagnostics,
)


def _train_matrix() -> np.ndarray:
    # Two raw numeric fields followed by their missingness bits.
    return np.asarray([
        [1.0, 10.0, 0.0, 0.0],
        [2.0, 20.0, 0.0, 0.0],
        [0.0, 30.0, 1.0, 0.0],
        [4.0, 0.0, 0.0, 1.0],
    ], dtype=np.float32)


def test_validation_or_future_values_do_not_change_train_scaler_or_schema():
    columns = ["a", "b"]
    train = _train_matrix()
    provenance = pd.DataFrame({"event_id": ["e0", "e1"], "source_file": ["f", "f"], "source_row": [0, 1], "timestamp": [1.0, 2.0]})
    first = fit_train_scaler(train, columns)
    first_fingerprint = feature_schema_fingerprint(columns, provenance)
    mutated_validation = np.asarray([[999999.0, -999999.0, 0.0, 0.0]], dtype=np.float32)
    mutated_future = np.asarray([[-123456.0, 888888.0, 0.0, 0.0]], dtype=np.float32)
    second = fit_train_scaler(train, columns)
    second_fingerprint = feature_schema_fingerprint(columns, provenance)
    assert first == second
    assert first_fingerprint == second_fingerprint
    assert np.isfinite(first.transform(mutated_validation)).all()
    assert np.isfinite(first.transform(mutated_future)).all()


def test_missingness_bits_are_explicit_and_not_perfect_label_copy():
    matrix = _train_matrix()
    labels = np.asarray([0, 1, 0, 1], dtype=np.int64)
    diagnostics = preprocessing_diagnostics(matrix, labels, ["a", "b"])
    assert diagnostics["finite_raw_values"] is True
    assert diagnostics["target_columns_in_features"] == []
    assert diagnostics["status"] == "PASS"

