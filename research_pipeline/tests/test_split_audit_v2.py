"""Small, data-free regression tests for the streaming split audit."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from research_pipeline.src.common.config import CORRECTIVE_TRAIN_END, CORRECTIVE_VALIDATION_END
from research_pipeline.src.pipeline.duplicate_handler import DuplicateRouter
from research_pipeline.src.pipeline.models import SourceRecord
from research_pipeline.src.validation import split_audit_v2 as audit


def _record(name: str, platform: str, start: float, end: float) -> SourceRecord:
    return SourceRecord(
        platform=platform,
        file_name=name,
        file_path=Path(name),
        relative_path=name,
        columns=["ts", "type", "label", "x"],
        rows_expected=0,
        timestamp_column="ts",
        timestamp_min=str(start),
        timestamp_max=str(end),
        group_id=platform,
        split_mode="source_file_holdout" if platform == "Network" else "chronological_calendar_block",
        fixed_split="train" if platform == "Network" else None,
        block_cut_1=CORRECTIVE_TRAIN_END,
        block_cut_2=CORRECTIVE_VALIDATION_END,
    )


class SplitAuditV2Tests(unittest.TestCase):
    def test_rare_attack_type_is_marked_unseen_from_train(self) -> None:
        captured = []
        with patch.object(audit, "write_csv_rows", side_effect=lambda path, rows, columns: captured.extend(rows)):
            audit._write_coverage(
                Path("coverage.csv"),
                {("IoT", "xss", "validation"): 1},
                {("IoT", "xss"): {"iot.csv"}},
            )
            frame = pd.DataFrame(captured)
            row = frame[(frame.platform == "IoT") & (frame.attack_type == "xss")].iloc[0]
            self.assertEqual(row["seen_unseen_from_train"], "UNSEEN")
            self.assertEqual(row["coverage_status"], "UNSEEN_IN_TRAIN_VALIDATION_ONLY")

    def test_missing_timestamp_and_positive_network_overlap_are_flagged(self) -> None:
        records = [
            _record("Network_dataset_1.csv", "Network", 0, 10),
            _record("Network_dataset_2.csv", "Network", 9, 20),
        ]
        result = {
            "daily": {},
            "quality": {("Network", "Network_dataset_1.csv"): {"invalid": 2}},
            "router_summary": {"Network_dataset_1.csv": {"routed": 0, "future_to_train": 0}},
            "cross_totals": {"fingerprints": 0, "rows": 0, "conflicts": 0},
            "envelopes": {(platform, split): [None, None] for platform in ("Network", "IoT", "Linux", "Windows") for split in ("train", "validation", "test")},
        }
        captured = []
        with patch.object(audit, "write_csv_rows", side_effect=lambda path, rows, columns: captured.extend(rows)):
            rows = audit._write_leakage_risks(Path("risks.csv"), records, result)
        statuses = {row["check"]: row["status"] for row in rows}
        self.assertEqual(statuses["network_source_interval_overlap"], "FAIL")
        self.assertEqual(statuses["missing_timestamp_forced_to_train"], "FAIL")

    def test_duplicate_router_can_route_future_duplicate_to_train(self) -> None:
        router = DuplicateRouter()
        final = router.route(np.array([123, 123], dtype=np.uint64), np.array([0, 2], dtype=np.int8))
        self.assertEqual(final.tolist(), [0, 0])
        self.assertEqual(router.routed, 1)

    def test_cross_file_duplicate_partition_detects_split_conflict(self) -> None:
        records = [_record("a.csv", "Network", 0, 1), _record("b.csv", "Network", 2, 3)]
        path = MagicMock()
        path.exists.return_value = True
        path.stat.return_value.st_size = 1
        data = np.array([(99, 0, 0, 0), (99, 0, 1, 2)], dtype=audit._DUP_DTYPE)
        with patch("numpy.fromfile", return_value=data):
            pairs, totals = audit._process_duplicate_partitions([path], records, {0: "Network|schema"})
        self.assertEqual(totals["fingerprints"], 1)
        self.assertEqual(totals["conflicts"], 1)
        self.assertEqual(len(pairs), 1)


if __name__ == "__main__":
    unittest.main()
