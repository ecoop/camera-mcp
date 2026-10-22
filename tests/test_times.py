# Copyright (c) 2026 Eric Cooper.
"""Times in one place: the log's UTC ISO format, local display, and reading "24h" / "yesterday"."""

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from camera_mcp.history import HistoryError, parse_when as history_parse_when
from camera_mcp.times import TimeError, epoch, iso, local, parse_when

LA = ZoneInfo("America/Los_Angeles")


def test_stored_times_round_trip_and_show_locally():
    t = datetime(2026, 10, 9, 13, 0, 0, 250000, tzinfo=ZoneInfo("UTC")).timestamp()
    assert iso(t) == "2026-10-09T13:00:00.250Z"  # what the log stores: UTC, milliseconds, Z
    assert epoch(iso(t)) == t
    assert local(iso(t), LA) == "2026-10-09T06:00:00-07:00" and local(None, LA) is None


def test_parse_when_reads_durations_days_and_iso_times():
    now = datetime(2026, 10, 9, 12, 0, tzinfo=LA).timestamp()
    assert parse_when("4h", now, LA) == now - 4 * 3600 and parse_when("now", now, LA) == now
    assert parse_when("yesterday", now, LA) == datetime(2026, 10, 8, tzinfo=LA).timestamp()
    assert parse_when("2026-10-09T06:00", now, LA) == datetime(2026, 10, 9, 6, 0, tzinfo=LA).timestamp()
    with pytest.raises(TimeError, match="can't read the time"):
        parse_when("last tuesday", now, LA)
    with pytest.raises(HistoryError, match="can't read the time"):  # history answers keep raising HistoryError
        history_parse_when("last tuesday", now, LA)
