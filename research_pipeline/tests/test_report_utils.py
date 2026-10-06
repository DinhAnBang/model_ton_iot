"""Check that shared report helpers keep CSV and rerun behavior stable."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path


from research_pipeline.src.common.utils import update_report_section, write_csv_rows


class ReportUtilsTests(unittest.TestCase):
    def test_csv_columns_and_phase_rerun(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            report = root / "research_pipeline" / "reports" / "table.csv"
            write_csv_rows(report, [{"source": "IoT", "rows": 2, "extra": "ignored"}], ["source", "rows"])
            with report.open(encoding="utf-8", newline="") as handle:
                self.assertEqual(list(csv.DictReader(handle)), [{"source": "IoT", "rows": "2"}])

            summary = report.parent / "audit_summary.md"
            summary.write_text("# Audit\n", encoding="utf-8")
            update_report_section(root, "## Phase 1.1", "## Phase 1.1\n\nFirst")
            update_report_section(root, "## Phase 1.2", "## Phase 1.2\n\nLater")
            update_report_section(root, "## Phase 1.1", "## Phase 1.1\n\nUpdated")
            self.assertEqual(
                summary.read_text(encoding="utf-8"),
                "# Audit\n## Phase 1.1\n\nUpdated\n## Phase 1.2\n\nLater\n",
            )


if __name__ == "__main__":
    unittest.main()
