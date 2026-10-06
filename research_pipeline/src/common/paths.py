"""Resolve all project paths from the package location."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PIPELINE_ROOT = PROJECT_ROOT / "research_pipeline"
REPORT_DIR = PIPELINE_ROOT / "reports"
CONFIG_DIR = PIPELINE_ROOT / "config"
DEFAULT_OUTPUT_ROOT = PIPELINE_ROOT / "output"
