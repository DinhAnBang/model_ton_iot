"""Run a non-destructive real-data acceptance sample for two IoT files."""

from __future__ import annotations

import time

from ..common.config import raw_type_mapping
from ..common.paths import PROJECT_ROOT
from ..pipeline import feature_selector, report_generator
from ..pipeline.data_reader import make_source_records
from ..pipeline.feature_selector import make_group_plans, read_feature_risk
from ..pipeline.models import RunState
from ..pipeline.output_writer import (
    clear_generated_output,
    export_dataset,
    validate_final_outputs,
    write_preprocessor,
)
from ..pipeline.preprocessor import fit_preprocessors
from ..pipeline.report_generator import generate_final_reports
from ..pipeline.split_data import assign_split_plan


SAMPLE_FILES = {"IoT_Garage_Door.csv", "IoT_Thermostat.csv"}
SAMPLE_OUTPUT = PROJECT_ROOT / "research_pipeline" / "output_remediation_acceptance_iot"
SAMPLE_REPORTS = PROJECT_ROOT / "research_pipeline" / "reports" / "remediation_acceptance"


def run(chunk_size: int = 50_000) -> dict[str, object]:
    if SAMPLE_OUTPUT.exists() or SAMPLE_REPORTS.exists():
        raise RuntimeError("Acceptance sample output/report already exists; refusing to overwrite it")
    records = [record for record in make_source_records(PROJECT_ROOT) if record.file_name in SAMPLE_FILES]
    if {record.file_name for record in records} != SAMPLE_FILES:
        raise RuntimeError("Inventory does not contain exactly the requested IoT sample files")
    assign_split_plan(records)
    risk = feature_selector.read_feature_risk()
    plans = make_group_plans(records, risk, legacy_sampling=False)
    clear_generated_output(SAMPLE_OUTPUT, overwrite=False)
    state = RunState()
    started = time.time()
    type_mapping = raw_type_mapping()
    fit_preprocessors(
        plans,
        type_mapping,
        chunk_size,
        state,
        risk=risk,
        quarantine_invalid=True,
    )
    preprocessors = [
        write_preprocessor(plan, SAMPLE_OUTPUT, chunk_size, quarantine_invalid=True)
        for plan in plans
    ]
    manifest_path = export_dataset(
        plans,
        type_mapping,
        chunk_size,
        SAMPLE_OUTPUT,
        state,
        quarantine_invalid=True,
    )
    SAMPLE_REPORTS.mkdir(parents=True, exist_ok=False)
    report_generator.REPORT_DIR = SAMPLE_REPORTS
    reports, elapsed, status = generate_final_reports(
        records,
        plans,
        state,
        risk,
        SAMPLE_OUTPUT,
        manifest_path,
        preprocessors,
        started,
        chunk_size,
        mode="remediated",
        quarantine_invalid=True,
    )
    validate_final_outputs(state, records)
    return {
        "status": status,
        "source_rows": sum(record.rows_expected for record in records),
        "output_rows": sum(state.output_rows.values()),
        "invalid_timestamps": dict(state.invalid_timestamps),
        "output_rows_by_split": {
            split: sum(value for key, value in state.output_rows.items() if key[3] == split)
            for split in ("train", "validation", "test", "quarantine")
        },
        "sample_output": str(SAMPLE_OUTPUT),
        "sample_reports": str(SAMPLE_REPORTS),
        "reports": {name: str(path) for name, path in reports.items()},
        "elapsed_seconds": elapsed,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(run(), ensure_ascii=False, indent=2))
