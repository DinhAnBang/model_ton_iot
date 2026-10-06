"""Run Phase-1 validation, splitting, preprocessing and export.

From the project root: python -m research_pipeline.src.main
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

from .common.config import DEFAULT_CHUNK_SIZE, raw_type_mapping
from .common.errors import PipelineError
from .common.paths import DEFAULT_OUTPUT_ROOT, PROJECT_ROOT
from .common.utils import log
from .pipeline.data_reader import make_source_records
from .pipeline.feature_selector import make_group_plans, read_feature_risk
from .pipeline.models import RunState
from .pipeline.output_writer import (
    clear_generated_output,
    export_dataset,
    validate_final_outputs,
    write_preprocessor,
)
from .pipeline.preprocessor import fit_preprocessors
from .pipeline.report_generator import generate_final_reports
from .pipeline.split_data import assign_split_plan


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--overwrite", action="store_true", help="replace only generated output directories")
    parser.add_argument("--max-files", type=int, default=None, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Coordinate the existing Phase-1 scientific workflow."""
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.chunk_size <= 0:
        raise PipelineError("--chunk-size must be positive")
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S", stream=sys.stdout)
    started = time.time()
    log(f"Project root: {PROJECT_ROOT}")
    log("Loading inventory, label mapping and feature-risk policy")
    type_mapping = raw_type_mapping()
    records = make_source_records(PROJECT_ROOT)
    if args.max_files is not None:
        records = records[: args.max_files]
    assign_split_plan(records)
    risk = read_feature_risk()
    clear_generated_output(DEFAULT_OUTPUT_ROOT, args.overwrite)
    plans = make_group_plans(records, risk)
    log(f"Input: {len(records)} files / {sum(r.rows_expected for r in records):,} rows")
    log(f"Groups: {', '.join(p.group_id for p in plans)}")

    state = RunState()
    fit_preprocessors(plans, type_mapping, args.chunk_size, state)
    preprocessors = [write_preprocessor(plan, DEFAULT_OUTPUT_ROOT, args.chunk_size) for plan in plans]
    manifest_path = export_dataset(plans, type_mapping, args.chunk_size, DEFAULT_OUTPUT_ROOT, state)
    reports, elapsed = generate_final_reports(
        records, plans, state, risk, DEFAULT_OUTPUT_ROOT, manifest_path,
        preprocessors, started, args.chunk_size,
    )
    validate_final_outputs(state, records)

    log("SUCCESS")
    for name, path in reports.items():
        log(f"  {name}: {path.relative_to(PROJECT_ROOT)}")
    log(f"  elapsed: {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PipelineError as error:
        logging.getLogger("research_pipeline").error("ERROR: %s", error)
        raise SystemExit(2) from error
