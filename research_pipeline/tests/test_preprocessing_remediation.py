"""Regression tests for the preprocessing remediation path."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from research_pipeline.src.common.config import (
    CORRECTIVE_TRAIN_END,
    CORRECTIVE_VALIDATION_END,
    QUARANTINE_CODE,
    SPLIT_CODE,
)
from research_pipeline.src.main import parse_args
from research_pipeline.src.pipeline.duplicate_handler import DuplicateRouter
from research_pipeline.src.pipeline.feature_selector import (
    final_features,
    infer_train_feature_specs,
    make_group_plans,
)
from research_pipeline.src.pipeline.models import FeatureSpec, GroupPlan, RunState, SourceRecord
from research_pipeline.src.pipeline.output_writer import transform_group
from research_pipeline.src.pipeline.report_generator import determine_pipeline_status
from research_pipeline.src.pipeline.split_data import remediated_split_codes


def make_record(path: Path, rows: int, *, platform: str = "Linux") -> SourceRecord:
    return SourceRecord(
        platform=platform,
        file_name=path.name,
        file_path=path,
        relative_path=f"Processed_datasets/{path.name}",
        columns=["ts", "feature", "type", "label"],
        rows_expected=rows,
        timestamp_column="ts",
        timestamp_min=str(CORRECTIVE_TRAIN_END - 100),
        timestamp_max=str(CORRECTIVE_VALIDATION_END + 100),
        group_id=f"{platform}::disk",
        split_mode="chronological_calendar_block",
        block_cut_1=CORRECTIVE_TRAIN_END,
        block_cut_2=CORRECTIVE_VALIDATION_END,
    )


def risk_for(feature: str = "feature") -> dict[tuple[str, str, str], dict[str, str]]:
    return {
        ("Linux", "events.csv", feature): {
            "platform": "Linux",
            "file": "events.csv",
            "feature": feature,
            "classification": "VALID_OBSERVABLE_FEATURE",
            "reason": "test feature",
        }
    }


class RemediationTests(unittest.TestCase):
    def test_isolated_output_arguments_are_available(self) -> None:
        args = parse_args([
            "--mode", "remediated",
            "--output-root", "research_pipeline/output_acceptance",
            "--report-dir", "research_pipeline/reports/acceptance",
        ])
        self.assertEqual(args.mode, "remediated")
        self.assertTrue(str(args.output_root).endswith("output_acceptance"))
        self.assertTrue(str(args.report_dir).replace("\\", "/").endswith("reports/acceptance"))

    def test_feature_kind_and_feature_list_ignore_validation_and_test_values(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "events.csv"
            base = pd.DataFrame({
                "ts": [
                    pd.Timestamp("2019-04-23", tz="UTC").timestamp(),
                    pd.Timestamp("2019-04-23", tz="UTC").timestamp() + 1,
                    pd.Timestamp("2019-04-24", tz="UTC").timestamp(),
                    pd.Timestamp("2019-04-26", tz="UTC").timestamp(),
                ],
                "feature": ["a", "b", "validation-only", "test-only"],
                "type": ["normal"] * 4,
                "label": [0] * 4,
            })
            base.to_csv(path, index=False)
            record = make_record(path, len(base))
            risk = risk_for()

            def learn(frame: pd.DataFrame) -> tuple[str, list[str]]:
                frame.to_csv(path, index=False)
                plan = make_group_plans([record], risk)[0]
                infer_train_feature_specs(plan, risk, {"normal": "normal"}, 2)
                return plan.feature_specs["feature"].kind, final_features(plan)

            first = learn(base)
            changed = base.copy()
            changed.loc[2:, "feature"] = [f"future-{index}" for index in range(100, 200)][:2]
            second = learn(changed)
            self.assertEqual(first, ("categorical", ["feature"]))
            self.assertEqual(second, first)

    def test_missing_timestamp_is_quarantined_and_record_count_reconciles(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            path = root / "events.csv"
            frame = pd.DataFrame({
                "ts": [
                    pd.Timestamp("2019-04-23", tz="UTC").timestamp(),
                    np.nan,
                    pd.Timestamp("2019-04-26", tz="UTC").timestamp(),
                ],
                "feature": [1.0, 2.0, 3.0],
                "type": ["normal", "normal", "DoS"],
                "label": [0, 0, 1],
            })
            frame.to_csv(path, index=False)
            record = make_record(path, len(frame))
            times = pd.Series(frame["ts"])
            codes = remediated_split_codes(times, record)
            self.assertEqual(codes.tolist(), [SPLIT_CODE["train"], QUARANTINE_CODE, SPLIT_CODE["test"]])
            self.assertEqual(len(codes), int((codes == 0).sum() + (codes == 1).sum() + (codes == 2).sum() + (codes == -1).sum()))

            spec = {"feature": FeatureSpec("feature", "numeric", "VALID_OBSERVABLE_FEATURE", "test")}
            plan = GroupPlan("Linux::disk", "Linux", [record], spec, ["feature"], record.columns, "linux_disk")
            state = RunState()
            manifest: list[dict[str, object]] = []
            with patch("research_pipeline.src.pipeline.output_writer.PROJECT_ROOT", root):
                transform_group(
                    plan,
                    {"normal": "normal", "DoS": "dos"},
                    2,
                    root / "output",
                    state,
                    manifest,
                    quarantine_invalid=True,
                )
            self.assertEqual(sum(state.output_rows.values()), len(frame))
            self.assertEqual(state.output_rows[("Linux", "Linux::disk", "events.csv", "quarantine")], 1)
            self.assertEqual(state.output_normal[("Linux", "Linux::disk", "events.csv", "quarantine")], 1)
            self.assertEqual(state.output_attack[("Linux", "Linux::disk", "events.csv", "quarantine")], 0)

    def test_equal_cutoff_boundaries_are_half_open_and_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            record = make_record(Path(temp_dir) / "events.csv", 4)
            times = pd.Series([
                CORRECTIVE_TRAIN_END - 1,
                CORRECTIVE_TRAIN_END,
                CORRECTIVE_VALIDATION_END - 1,
                CORRECTIVE_VALIDATION_END,
            ])
            self.assertEqual(
                remediated_split_codes(times, record).tolist(),
                [SPLIT_CODE["train"], SPLIT_CODE["validation"], SPLIT_CODE["validation"], SPLIT_CODE["test"]],
            )

    def test_quarantine_router_never_registers_invalid_rows_as_train(self) -> None:
        router = DuplicateRouter()
        fingerprints = np.asarray([10, 10, 11], dtype="uint64")
        codes = np.asarray([0, QUARANTINE_CODE, 2], dtype="int8")
        final = router.route_masked(fingerprints, codes, codes != QUARANTINE_CODE, QUARANTINE_CODE)
        self.assertEqual(final.tolist(), [0, -1, 2])
        self.assertEqual(router.routed, 0)

    def test_audit_fail_changes_terminal_status(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            leakage = Path(temp_dir) / "leakage.csv"
            pd.DataFrame([{"status": "FAIL"}]).to_csv(leakage, index=False)
            with patch("research_pipeline.src.pipeline.report_generator.REPORT_DIR", Path(temp_dir)):
                self.assertEqual(determine_pipeline_status("remediated", leakage), "blocked_by_audit")
                audit_dir = Path(temp_dir) / "split_audit_v2"
                audit_dir.mkdir()
                pd.DataFrame([{"status": "FAIL"}]).to_csv(audit_dir / "leakage_risks.csv", index=False)
                self.assertEqual(determine_pipeline_status("v1", Path(temp_dir) / "clean.csv"), "blocked_by_audit")


if __name__ == "__main__":
    unittest.main()
