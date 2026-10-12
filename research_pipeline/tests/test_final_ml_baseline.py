from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from research_pipeline.src.experimental_phase3_baseline.final_ml_baseline import (
    _canonical_type_series,
    _model_config,
    compute_metrics,
)
from research_pipeline.src.pipeline.feature_selector import final_features, make_group_plans
from research_pipeline.src.pipeline.models import RunState, SourceRecord
from research_pipeline.src.pipeline.preprocessor import fit_preprocessors, transform_baseline
from research_pipeline.src.pipeline.split_data import remediated_split_codes
from research_pipeline.src.validation.final_split_protocol import assign_calendar_split


def test_final_protocol_uses_half_open_boundaries() -> None:
    assert assign_calendar_split("2019-04-23T23:59:59Z") == "train"
    assert assign_calendar_split("2019-04-24T00:00:00Z") == "validation"
    assert assign_calendar_split("2019-04-26T00:00:00Z") == "test"
    assert assign_calendar_split("2019-04-26T00:00:00") == "quarantine"


def test_pipeline_remediated_split_quarantines_invalid_time() -> None:
    record = SourceRecord(
        platform="Linux",
        file_name="linux_disk_1.csv",
        file_path=None,  # type: ignore[arg-type]
        relative_path="x",
        columns=["ts", "feature", "label", "type"],
        rows_expected=3,
        timestamp_column="ts",
        timestamp_min="2019-04-02T00:00:00",
        timestamp_max="2019-04-27T00:00:00",
        split_mode="chronological_calendar_block",
        block_cut_1=pd.Timestamp("2019-04-24", tz="UTC").timestamp(),
        block_cut_2=pd.Timestamp("2019-04-26", tz="UTC").timestamp(),
    )
    from research_pipeline.src.pipeline.normalization import event_time_numeric

    frame = pd.DataFrame({"ts": [1554163200, 1556150400, "bad"], "feature": [1, 2, 3], "label": [0, 1, 1], "type": ["normal", "scanning", "scanning"]})
    codes = remediated_split_codes(event_time_numeric(frame, record), record)
    assert codes.tolist() == [0, 1, -1]


def test_feature_statistics_are_fit_on_train_only() -> None:
    path = Path.cwd() / ".final_ml_unit_fit.csv"
    frame = pd.DataFrame(
        {
            "ts": [1554218915, 1554218920, 1556150400],
            "feature": [1.0, 3.0, 1000.0],
            "label": [0, 1, 1],
            "type": ["normal", "scanning", "scanning"],
        }
    )
    frame.to_csv(path, index=False)
    try:
        record = SourceRecord(
            platform="Linux",
            file_name="linux_disk_1.csv",
            file_path=path,
            relative_path="linux_disk_1.csv",
            columns=list(frame.columns),
            rows_expected=len(frame),
            timestamp_column="ts",
            timestamp_min="2019-04-02T00:00:00",
            timestamp_max="2019-04-25T00:00:00",
            group_id="Linux::disk",
            split_mode="chronological_calendar_block",
            block_cut_1=pd.Timestamp("2019-04-24", tz="UTC").timestamp(),
            block_cut_2=pd.Timestamp("2019-04-26", tz="UTC").timestamp(),
        )
        risk = {
            ("Linux", "linux_disk_1.csv", "feature"): {
                "platform": "Linux",
                "file": "linux_disk_1.csv",
                "feature": "feature",
                "classification": "VALID_OBSERVABLE_FEATURE",
                "reason": "test",
            }
        }
        plan = make_group_plans([record], risk)[0]
        fit_preprocessors([plan], {"normal": "normal", "scanning": "scanning"}, 2, RunState(), risk=risk, quarantine_invalid=True)
        assert final_features(plan) == ["feature"]
        assert plan.feature_specs["feature"].median == 2.0
        validation_transformed = transform_baseline(frame.iloc[[2]], plan)
        assert validation_transformed.loc[2, "feature"] == 1000.0
    finally:
        path.unlink(missing_ok=True)


def test_known_mask_keeps_only_normal_and_scanning() -> None:
    frame = pd.DataFrame({"type": ["normal", "scanning", "DDoS", "password"]})
    canonical = _canonical_type_series(frame, {"normal": "normal", "scanning": "scanning", "DDoS": "ddos", "password": "password"})
    assert canonical.isin({"normal", "scanning"}).tolist() == [True, True, False, False]


def test_metrics_include_confusion_and_undefined_semantics() -> None:
    metrics = compute_metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.8, 0.7, 0.2]))
    assert (metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"]) == (1, 1, 1, 1)
    assert metrics["status"] == "PASS"
    single = compute_metrics(np.array([0, 0]), np.array([0.1, 0.2]))
    assert single["roc_auc"] == "UNDEFINED"
    assert single["pr_auc_attack"] == "UNDEFINED"


def test_model_configuration_is_reproducible_and_has_no_test_tuning() -> None:
    y = np.array([0, 0, 1], dtype=np.int8)
    first = _model_config("XGBoost", y)
    second = _model_config("XGBoost", y)
    assert first == second
    assert first["params"]["random_state"] == 42
    assert first["train_class_weight_source"].endswith("Train only")

