"""Run the modular CLI on three tiny Network source files."""

from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from research_pipeline.src import main as pipeline_main
from research_pipeline.src.pipeline import data_reader, feature_selector, output_writer, report_generator


class PipelineSmokeTests(unittest.TestCase):
    def test_main_writes_unchanged_source_and_train_only_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            reports = root / "research_pipeline" / "reports"
            reports.mkdir(parents=True)
            output = root / "research_pipeline" / "output"
            sources = root / "Processed_datasets" / "Processed_Network_dataset"
            sources.mkdir(parents=True)

            inventory = []
            risks = []
            original_bytes = {}
            for index, day in enumerate(("2019-04-23", "2019-04-24", "2019-04-26"), start=1):
                name = f"Network_dataset_{index}.csv"
                path = sources / name
                timestamp = pd.Timestamp(day, tz="UTC").timestamp()
                pd.DataFrame({
                    "ts": [timestamp, timestamp + 1],
                    "duration": [float(index), float(index + 1)],
                    "type": ["normal", "DoS"],
                    "label": [0, 1],
                }).to_csv(path, index=False)
                original_bytes[path] = path.read_bytes()
                inventory.append({
                    "platform": "Network",
                    "file_name": name,
                    "file_path": path.relative_to(root).as_posix(),
                    "rows": 2,
                    "columns": 4,
                    "timestamp_column": "ts",
                    "timestamp_min": str(timestamp),
                    "timestamp_max": str(timestamp + 1),
                    "duplicate_count": 0,
                })
                risks.append({
                    "platform": "Network",
                    "file": name,
                    "feature": "duration",
                    "classification": "VALID_OBSERVABLE_FEATURE",
                    "reason": "observable measurement",
                })
            pd.DataFrame(inventory).to_csv(reports / "dataset_inventory.csv", index=False)
            pd.DataFrame(risks).to_csv(reports / "feature_risk.csv", index=False)

            with ExitStack() as stack:
                for module, attribute, value in (
                    (pipeline_main, "PROJECT_ROOT", root),
                    (pipeline_main, "DEFAULT_OUTPUT_ROOT", output),
                    (data_reader, "REPORT_DIR", reports),
                    (feature_selector, "REPORT_DIR", reports),
                    (output_writer, "PROJECT_ROOT", root),
                    (report_generator, "PROJECT_ROOT", root),
                    (report_generator, "REPORT_DIR", reports),
                ):
                    stack.enter_context(patch.object(module, attribute, value))
                self.assertEqual(pipeline_main.main(["--chunk-size", "2"]), 0)

            summary = json.loads((output / "manifests" / "run_summary.json").read_text(encoding="utf-8"))
            artifact = json.loads((output / "preprocessors" / "network.json").read_text(encoding="utf-8"))
            distribution = pd.read_csv(reports / "final_split_distribution.csv")
            self.assertEqual(summary["source_rows"], 6)
            self.assertEqual(summary["output_rows"], 6)
            self.assertEqual(artifact["fit_split"], "train")
            self.assertEqual(distribution.groupby("split")["rows"].sum().to_dict(), {
                "train": 2, "validation": 2, "test": 2,
            })
            self.assertTrue(all(path.read_bytes() == content for path, content in original_bytes.items()))


if __name__ == "__main__":
    unittest.main()
