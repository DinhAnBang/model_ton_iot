from datetime import datetime, timezone

import pandas as pd

from research_pipeline.src.validation.final_split_protocol import assign_calendar_split
from research_pipeline.src.validation.final_timestamp_validation import (
    history_before_target,
    parse_absolute_iso,
    parse_processed_time,
    parse_source_local_iso,
    timestamp_offset_seconds,
)


def test_unix_seconds_are_seconds_not_milliseconds():
    chunk = pd.DataFrame({"ts": [1554198358, 1554198359]})
    parsed = parse_processed_time(chunk, "Network")
    assert parsed.iloc[0] == pd.Timestamp("2019-04-02T09:45:58")


def test_explicit_iso_timezone_and_unverified_naive_iso_are_separate():
    assert parse_absolute_iso("2019-04-24T00:00:00Z") == datetime(2019, 4, 24, tzinfo=timezone.utc)
    assert parse_absolute_iso("2019-04-24T00:00:00") is None
    assert parse_source_local_iso("2019-04-24T00:00:00") == datetime(2019, 4, 24)
    assert parse_source_local_iso("2019-04-24T00:00:00Z") is None


def test_iot_date_time_parses_and_missing_value_is_invalid():
    chunk = pd.DataFrame({"date": ["24-Apr-19", "bad"], "time": ["03:20:32", "03:20:32"]})
    parsed = parse_processed_time(chunk, "IoT")
    assert parsed.notna().tolist() == [True, False]


def test_offset_formula_is_explicit_and_reproducible():
    epoch = datetime(2019, 4, 24, 3, 20, 32, tzinfo=timezone.utc).timestamp()
    local = datetime(2019, 4, 23, 20, 20, 32)
    assert timestamp_offset_seconds(epoch, local) == 25200.0


def test_snapshot_history_excludes_equal_and_future_events():
    target = datetime(2019, 4, 24, 0, 0, 0)
    values = [datetime(2019, 4, 23, 23, 59, 59), target, datetime(2019, 4, 24, 0, 0, 1)]
    assert history_before_target(values, target) == [datetime(2019, 4, 23, 23, 59, 59)]


def test_half_open_split_boundary_remains_deterministic():
    assert assign_calendar_split("2019-04-24T00:00:00Z") == "validation"
    assert assign_calendar_split("2019-04-26T00:00:00Z") == "test"
