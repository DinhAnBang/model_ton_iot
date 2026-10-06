"""Small checks for the chronological split and train-only fit contract."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


from research_pipeline.src.common.config import CORRECTIVE_TRAIN_END, CORRECTIVE_VALIDATION_END
from research_pipeline.src.common.errors import PipelineError
from research_pipeline.src.pipeline.models import FeatureSpec, GroupPlan, RunState, SourceRecord
from research_pipeline.src.pipeline.preprocessor import fit_group, transform_baseline
from research_pipeline.src.pipeline.split_data import (
    assign_split_plan,
    base_split_codes,
    calendar_split_description,
    platform_class_coverage,
    scenario_id_series,
)


def record(path: Path, *, split_mode: str = "chronological_calendar_block") -> SourceRecord:
    return SourceRecord(
        platform="Linux",
        file_name=path.name,
        file_path=path,
        relative_path=f"Processed_datasets/{path.name}",
        columns=["ts", "RDDSK", "type", "label"],
        rows_expected=0,
        timestamp_column="ts",
        timestamp_min="",
        timestamp_max="",
        group_id="Linux::disk",
        split_mode=split_mode,
        block_cut_1=CORRECTIVE_TRAIN_END,
        block_cut_2=CORRECTIVE_VALIDATION_END,
    )


class SplitTests(unittest.TestCase):
    def test_network_file_holdout_counts(self) -> None:
        network_files = []
        for index in range(23):
            source = record(Path(f"Network_dataset_{index + 1}.csv"))
            source.platform = "Network"
            source.group_id = "Network"
            source.timestamp_min = str(index)
            network_files.append(source)
        assign_split_plan(network_files)
        self.assertEqual(
            [sum(source.fixed_split == split for source in network_files) for split in ("train", "validation", "test")],
            [16, 3, 4],
        )

    def test_complete_days_and_missing_time(self) -> None:
        dates = ["2019-04-23", "2019-04-24", "2019-04-25", "2019-04-26"]
        times = pd.Series([pd.Timestamp(day, tz="UTC").timestamp() for day in dates] + [np.nan])
        source = record(Path("disk.csv"))

        self.assertEqual(base_split_codes(times, source).tolist(), [0, 1, 1, 2, 0])
        blocks = scenario_id_series(times, source).tolist()
        self.assertEqual(len(set(blocks)), 5)
        self.assertTrue(blocks[-1].endswith("MISSING_TIME"))

    def test_source_holdout_keeps_all_rows_together(self) -> None:
        source = record(Path("network.csv"), split_mode="source_file_holdout")
        source.fixed_split = "validation"
        times = pd.Series([1.0, 2.0, np.nan])
        self.assertEqual(base_split_codes(times, source).tolist(), [1, 1, 1])

    def test_missing_boundaries_fail_instead_of_row_split(self) -> None:
        source = record(Path("disk.csv"))
        source.block_cut_1 = None
        with self.assertRaises(PipelineError):
            base_split_codes(pd.Series([1.0, 2.0]), source)

    def test_report_description_uses_configured_boundaries(self) -> None:
        self.assertEqual(
            calendar_split_description(),
            "train through 2019-04-23, validation on 2019-04-24 through 2019-04-25, test from 2019-04-26 onward",
        )


class CoverageTests(unittest.TestCase):
    def test_train_and_test_require_both_labels(self) -> None:
        rows = pd.DataFrame([
            {"platform": "IoT", "split": "train", "normal_count": 3, "attack_count": 1},
            {"platform": "IoT", "split": "validation", "normal_count": 2, "attack_count": 0},
            {"platform": "IoT", "split": "test", "normal_count": 2, "attack_count": 1},
            {"platform": "Linux", "split": "train", "normal_count": 3, "attack_count": 0},
            {"platform": "Linux", "split": "test", "normal_count": 2, "attack_count": 0},
        ])
        iot, linux = platform_class_coverage(rows)
        self.assertTrue(iot.eligible)
        self.assertFalse(iot.validation_both)
        self.assertFalse(linux.eligible)
        self.assertEqual(linux.missing_requirements, ["train attack", "test attack"])


class PreprocessorTests(unittest.TestCase):
    def test_fit_uses_train_rows_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "disk.csv"
            days = ["2019-04-23"] * 3 + ["2019-04-24", "2019-04-26"]
            frame = pd.DataFrame({
                "ts": [pd.Timestamp(day, tz="UTC").timestamp() for day in days],
                "RDDSK": [1.0, 3.0, np.nan, 1000.0, 2000.0],
                "type": ["normal", "DoS", "normal", "DoS", "DoS"],
                "label": [0, 1, 0, 1, 1],
            })
            frame.to_csv(path, index=False)
            source = record(path)
            source.rows_expected = len(frame)
            spec = FeatureSpec("RDDSK", "numeric", "VALID_OBSERVABLE_FEATURE", "test")
            plan = GroupPlan("Linux::disk", "Linux", [source], {"RDDSK": spec}, ["RDDSK"], source.columns, "linux_disk")

            fit_group(plan, {"normal": "normal", "DoS": "dos"}, chunk_size=2, state=RunState())

            self.assertEqual(spec.median, 2.0)
            self.assertEqual(spec.train_nonmissing, 2)
            transformed = transform_baseline(frame.iloc[[2, 3]], plan)
            self.assertEqual(transformed["RDDSK"].tolist(), [2.0, 1000.0])
            self.assertEqual(transformed.columns.tolist(), ["RDDSK", "label"])


if __name__ == "__main__":
    unittest.main()
