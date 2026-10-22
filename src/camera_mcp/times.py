# Copyright (c) 2026 Eric Cooper.
"""Times, in one place. The log stores UTC ISO strings (`iso`); answers show local times (`local`);
people and models ask with "24h", "yesterday" or an ISO time (`parse_when`)."""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

DURATION = re.compile(r"^(\d+(?:\.\d+)?)\s*(s|m|h|d|w)$")
UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}


class TimeError(ValueError):
    """A time that can't be read."""


def iso(t: float) -> str:
    """Epoch seconds as the log stores times: UTC ISO 8601 to the millisecond, with a Z."""
    return datetime.fromtimestamp(t, timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def epoch(text: str) -> float:
    """A stored ISO time as epoch seconds."""
    return datetime.fromisoformat(text).timestamp()


def local(text: str | None, tz) -> str | None:
    """A stored ISO time in the site's time zone, to the second (None stays None)."""
    return datetime.fromisoformat(text).astimezone(tz).isoformat(timespec="seconds") if text else None


def parse_when(text: str | None, now: float, tz) -> float:
    """An ISO time (local if it has no zone), a duration ago ("15m", "4h", "3d", "2w"),
    "now", "today" or "yesterday", as epoch seconds. Raises TimeError."""
    if text is None or text.strip().lower() in ("", "now"):
        return now
    t = text.strip().lower()
    if m := DURATION.match(t):
        return now - float(m.group(1)) * UNITS[m.group(2)]
    if t in ("today", "yesterday"):
        midnight = datetime.fromtimestamp(now, tz).replace(hour=0, minute=0, second=0, microsecond=0)
        return (midnight - timedelta(days=1 if t == "yesterday" else 0)).timestamp()
    try:
        when = datetime.fromisoformat(text.strip())
    except ValueError:
        raise TimeError(f"can't read the time {text!r}: use an ISO time like 2026-10-09T06:00, a "
                        "duration ago like 15m, 4h, 3d or 2w, or now, today, yesterday") from None
    return (when if when.tzinfo else when.replace(tzinfo=tz)).timestamp()
