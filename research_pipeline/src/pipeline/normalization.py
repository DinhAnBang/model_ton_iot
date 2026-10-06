"""Normalization for the Phase-1 research pipeline."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from ..common.config import UNKNOWN_CANONICAL
from ..common.errors import PipelineError
from .models import RunState, SourceRecord


def parse_time_bound(value: Any, platform: str) -> float:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return float("nan")
    text = str(value)
    if platform == "IoT":
        parsed = pd.to_datetime(text, errors="coerce", utc=True)
        if pd.isna(parsed):
            return float("nan")
        return float(parsed.value / 1_000_000_000)
    try:
        return float(text)
    except ValueError:
        parsed = pd.to_datetime(text, errors="coerce", utc=True)
        if pd.isna(parsed):
            return float("nan")
        return float(parsed.value / 1_000_000_000)


def iot_date_time_parts(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Return epoch seconds and normalized local timestamp text for IoT rows."""
    date_raw = frame["date"].astype("string")
    time_raw = frame["time"].astype("string").str.strip()
    parsed_date = pd.to_datetime(date_raw, format="%d-%b-%y", errors="coerce", utc=True)
    date_seconds = parsed_date.dt.as_unit("ns").astype("int64").astype("float64") / 1_000_000_000
    date_valid = parsed_date.notna()
    parts = time_raw.str.extract(r"^(?P<hour>\d{1,2}):(?P<minute>\d{1,2}):(?P<second>\d{1,2})$")
    hour = pd.to_numeric(parts["hour"], errors="coerce")
    minute = pd.to_numeric(parts["minute"], errors="coerce")
    second = pd.to_numeric(parts["second"], errors="coerce")
    time_valid = (
        hour.notna() & minute.notna() & second.notna()
        & hour.between(0, 23) & minute.between(0, 59) & second.between(0, 59)
    )
    valid = date_valid & time_valid
    values = date_seconds + hour * 3600 + minute * 60 + second
    values[~valid] = np.nan
    text = parsed_date.dt.strftime("%Y-%m-%d").astype("string") + "T" + time_raw + ""
    text[~valid] = pd.NA
    return values, text


def event_time_numeric(frame: pd.DataFrame, record: SourceRecord) -> pd.Series:
    if record.timestamp_column == "date + time":
        if "date" not in frame.columns or "time" not in frame.columns:
            raise PipelineError(f"{record.file_name}: missing date/time columns")
        values, _ = iot_date_time_parts(frame)
        return values
    if record.timestamp_column not in frame.columns:
        raise PipelineError(f"{record.file_name}: missing timestamp column {record.timestamp_column}")
    numeric = pd.to_numeric(frame[record.timestamp_column], errors="coerce")
    parsed = pd.to_datetime(numeric, errors="coerce", unit="s", utc=True)
    # A NaT cast to int64 is the minimum int64; keep invalid rows as NaN.
    values = parsed.dt.as_unit("ns").astype("int64").astype("float64") / 1_000_000_000
    values[parsed.isna()] = np.nan
    return values


def event_time_text(frame: pd.DataFrame, record: SourceRecord) -> pd.Series:
    if record.timestamp_column == "date + time":
        _, text = iot_date_time_parts(frame)
        return text
    numeric = pd.to_numeric(frame[record.timestamp_column], errors="coerce")
    parsed = pd.to_datetime(numeric, errors="coerce", unit="s", utc=True)
    return parsed.dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ").astype("string")


def raw_timestamp_text(frame: pd.DataFrame, record: SourceRecord) -> pd.Series:
    if record.timestamp_column == "date + time":
        return (frame["date"].astype("string") + " " + frame["time"].astype("string")).astype("string")
    return frame[record.timestamp_column].astype("string")


def canonical_type_series(frame: pd.DataFrame, type_mapping: dict[str, str]) -> tuple[pd.Series, pd.Series]:
    raw = frame["type"].astype("string")
    canonical = raw.map(type_mapping).fillna(UNKNOWN_CANONICAL).astype("string")
    return raw, canonical


def validate_chunk(
    frame: pd.DataFrame,
    rec: SourceRecord,
    type_mapping: dict[str, str],
    state: RunState,
    validation_only: bool = False,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    required = {"label", "type"}
    if rec.timestamp_column == "date + time":
        required.update({"date", "time"})
    else:
        required.add(rec.timestamp_column)
    missing = required - set(frame.columns)
    if missing:
        raise PipelineError(f"{rec.file_name}: missing required columns {sorted(missing)}")

    labels = pd.to_numeric(frame["label"], errors="coerce")
    bad_labels = int((labels.isna() | ~labels.isin([0, 1])).sum())
    if bad_labels:
        state.invalid_labels[rec.file_name] += bad_labels
        raise PipelineError(f"{rec.file_name}: {bad_labels} invalid labels; expected only 0/1")
    raw_type, canonical = canonical_type_series(frame, type_mapping)
    unknown = int((canonical == UNKNOWN_CANONICAL).sum())
    if unknown and not validation_only:
        state.unknown_types[rec.file_name] += unknown
    times = event_time_numeric(frame, rec)
    invalid_times = int(times.isna().sum())
    if invalid_times:
        if not validation_only:
            state.invalid_timestamps[rec.file_name] += invalid_times
    return labels.astype("int64"), raw_type, canonical, times


def format_event_range(value: float, platform: str) -> str:
    if not np.isfinite(value):
        return ""
    stamp = pd.to_datetime(value, unit="s", utc=True)
    if platform == "IoT":
        return stamp.strftime("%Y-%m-%dT%H:%M:%SZ")
    return stamp.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
