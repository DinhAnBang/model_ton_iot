"""Compare a completed Phase-1 run with the saved pre-refactor reference.

Run from the project root after a full rebuild:
    python -m research_pipeline.tests.check_full_regression
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_ROOT = PROJECT_ROOT / "research_pipeline"
REFERENCE = Path(__file__).with_name("regression_reference.json")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_metadata_sha256() -> str:
    source_root = PROJECT_ROOT / "Processed_datasets"
    entries = sorted(
        f"{path.relative_to(source_root).as_posix()}|{path.stat().st_size}|{path.stat().st_mtime_ns}"
        for path in source_root.rglob("*.csv")
    )
    return hashlib.sha256("\n".join(entries).encode()).hexdigest()


def output_paths_sha256() -> tuple[str, int, int]:
    output_root = PIPELINE_ROOT / "output"
    baseline = list((output_root / "baseline").rglob("*.csv.gz"))
    semantic = list((output_root / "semantic").rglob("*.csv.gz"))
    paths = sorted(path.relative_to(PIPELINE_ROOT).as_posix() for path in baseline + semantic)
    return hashlib.sha256("\n".join(paths).encode()).hexdigest(), len(baseline), len(semantic)


def compare() -> list[str]:
    expected = json.loads(REFERENCE.read_text(encoding="utf-8"))
    summary = json.loads((PIPELINE_ROOT / "output/manifests/run_summary.json").read_text(encoding="utf-8"))
    distribution = pd.read_csv(PIPELINE_ROOT / "reports/final_split_distribution.csv")
    leakage = pd.read_csv(PIPELINE_ROOT / "reports/split_leakage_check.csv")
    preprocessors = list((PIPELINE_ROOT / "output/preprocessors").glob("*.json"))
    output_paths_hash, baseline_files, semantic_files = output_paths_sha256()

    actual_counts = {
        f"{platform}/{split}": [int(row["rows"]), int(row["normal_count"]), int(row["attack_count"])]
        for (platform, split), row in distribution.groupby(["platform", "split"])[
            ["rows", "normal_count", "attack_count"]
        ].sum().iterrows()
    }
    feature_counts = {group["group_id"]: group["baseline_feature_count"] for group in summary["groups"]}
    fit_splits = {
        json.loads(path.read_text(encoding="utf-8"))["fit_split"] for path in preprocessors
    }
    checks = {
        "source file count": summary["source_file_count"] == expected["source_file_count"],
        "source rows": summary["source_rows"] == expected["source_rows"],
        "output rows": summary["output_rows"] == expected["output_rows"],
        "platform split and label counts": actual_counts == expected["platform_split_counts"],
        "baseline features by group": feature_counts == expected["baseline_feature_counts"],
        "Network 32 features": feature_counts.get("Network") == 32,
        "duplicate crossing": all(count == 0 for count in summary["duplicate_cross_split"].values()),
        "leakage statuses": leakage.status.value_counts().to_dict() == expected["leakage_status_counts"],
        "no failing leakage checks": leakage.status.isin(["PASS", "EXPECTED_LIMITATION"]).all(),
        "train-only preprocessors": len(preprocessors) == expected["preprocessor_count"] and fit_splits == {expected["preprocessor_fit_split"]},
        "output paths": output_paths_hash == expected["output_paths_sha256"],
        "baseline file count": baseline_files == expected["baseline_file_count"],
        "semantic file count": semantic_files == expected["semantic_file_count"],
        "source metadata": source_metadata_sha256() == expected["source_metadata_sha256"],
    }
    for relative_path, expected_hash in expected["artifact_sha256"].items():
        checks[f"artifact: {relative_path}"] = sha256(PIPELINE_ROOT / relative_path) == expected_hash

    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'} {name}")
    return [name for name, passed in checks.items() if not passed]


if __name__ == "__main__":
    failures = compare()
    if failures:
        raise SystemExit(f"Regression failed: {', '.join(failures)}")
    print("REGRESSION PASS")
