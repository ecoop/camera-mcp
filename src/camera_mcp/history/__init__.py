# Copyright (c) 2026 Eric Cooper.
"""Questions about the past, answered from the monitor's visit log.

The log is opened read-only: the monitor is its only writer. Every answer about a stretch
of time says how much of it each camera was actually watched, because "no skunks this week"
is only true if the camera was watching all week.

History is assembled from one module per kind of question: base (shared plumbing and coverage),
visits, health, tags and media. Rules about boxes live in camera_mcp.boxtags."""

from camera_mcp.history.base import GROUP_BY, HistoryError, LogReader, parse_when
from camera_mcp.history.health import HealthQueries
from camera_mcp.history.media import MediaQueries
from camera_mcp.history.tags import TagQueries
from camera_mcp.history.visits import VisitQueries


class History(VisitQueries, TagQueries, MediaQueries, HealthQueries, LogReader):
    """The visit log, read-only, answering questions about the past (see the modules for each kind)."""


__all__ = ["History", "HistoryError", "GROUP_BY", "parse_when"]
