# Copyright (c) 2026 Eric Cooper.
"""The plumbing every history query shares: opening the visit log read-only, filters (cameras,
tags, taxa, time), hidden-visit handling, schema checks for older logs, local times, and coverage
(how much of a stretch each camera was actually watched). LogReader is the base of History."""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from contextlib import closing
from pathlib import Path

from camera_mcp import times
from camera_mcp.config import Camera, Site, find_camera
from camera_mcp.taxa import CLASS_NAMES, GROUPS, UNIONS, expand_taxon
from camera_mcp.times import iso

GROUP_NAMES = {g for g, _ in GROUPS} | {"bird", "mammal", "animal", "person", "vehicle"}
GROUP_BY = ("species", "group", "camera", "hour", "day")
WATCHING_LAG = 60.0  # a camera whose last frame is older than this isn't being watched now


class HistoryError(ValueError):
    pass


def parse_when(text: str | None, now: float, tz) -> float:
    """times.parse_when, raising HistoryError (what every history answer's caller handles)."""
    try:
        return times.parse_when(text, now, tz)
    except times.TimeError as e:
        raise HistoryError(str(e)) from None


def _get(row: sqlite3.Row | None, key: str, default=None):
    """A column that an older monitor's log may not have yet (the server never alters the log)."""
    return row[key] if row is not None and key in row.keys() else default


def _tables(db: sqlite3.Connection) -> set[str]:
    return {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}


def _has_column(db: sqlite3.Connection, name: str) -> bool:
    return name in {r[1] for r in db.execute("PRAGMA table_info(visits)")}


def _fav_cols(db: sqlite3.Connection) -> str:
    """SQL for a visit row's favourite (kind, clip moment), NULL when the log has no favorites table."""
    if "favorites" not in _tables(db):
        return "NULL AS fav_kind, NULL AS fav_at"
    return ("(SELECT kind FROM favorites f WHERE f.visit_id = v.id) AS fav_kind,"
            " (SELECT clip_at FROM favorites f WHERE f.visit_id = v.id) AS fav_at")


def _gated(db: sqlite3.Connection) -> bool:
    """Whether the log records the unnamed-visit gate's verdicts (an older monitor's doesn't)."""
    return "suppressed" in {r[1] for r in db.execute("PRAGMA table_info(visits)")}


class LogReader:
    """The shared plumbing; History adds the query mixins on top."""

    def __init__(self, db_path: Path, cameras: dict[str, Camera], site: Site | None = None):
        self.db_path = Path(db_path)
        self.cameras = cameras
        self.tz = (site or Site()).tz()

    def _db(self) -> sqlite3.Connection:
        if not self.db_path.exists():
            raise HistoryError(f"there's no visit log yet: the monitor hasn't run ({self.db_path} doesn't exist)")
        db = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        db.row_factory = sqlite3.Row
        if "visits" not in _tables(db):
            raise HistoryError("the visit log has no visits yet: it was written by a monitor without animal "
                               "identification; run the current monitor (with SpeciesNet) to start logging visits")
        return db

    def _local(self, iso: str | None) -> str | None:
        return times.local(iso, self.tz)

    @staticmethod
    def _iso(t: float) -> str:
        return iso(t)

    def _camera_names(self, cameras: list[str] | None, tag: str | None) -> list[str]:
        names = list(self.cameras)
        if cameras:
            names = []
            for query in cameras:
                cam = find_camera(self.cameras, query)
                if cam is None:
                    raise HistoryError(f"unknown camera {query!r}; known: {', '.join(sorted(self.cameras))}")
                names.append(cam.name)
        if tag:
            tagged = [c.name for c in self.cameras.values() if tag in c.tags]
            if not tagged:
                tags = sorted({t for c in self.cameras.values() for t in c.tags})
                raise HistoryError(f"no camera has the tag {tag!r}; tags in use: {', '.join(tags) or 'none'}")
            names = [n for n in names if n in tagged]
        return names

    @staticmethod
    def _taxon(taxon: str, col: str = "") -> tuple[str, list]:
        """SQL matching a taxon: a group ("songbirds"), a union ("night-critters"), a class
        ("bird"), a common name ("striped skunk") or a taxonomy prefix ("mammalia;rodentia")."""
        q = taxon.strip().lower()
        if q in UNIONS:
            groups = list(UNIONS[q])
            return f"{col}grp IN ({', '.join('?' * len(groups))})", groups
        if q in GROUP_NAMES - {"bird", "mammal"}:
            return f"{col}grp = ?", [q]  # groups are already decided per visit, in order (jays aren't songbirds)
        clauses, params = [f"lower({col}label) = ?"], [q]
        for prefix in expand_taxon(q):
            if ";" in prefix or prefix in CLASS_NAMES:
                clauses.append(f"{col}taxonomy LIKE ?")
                params.append(prefix + "%")
        return "(" + " OR ".join(clauses) + ")", params

    def _where(self, t0: float, t1: float, names: list[str], taxon: str | None, min_score: float,
               col: str = "") -> tuple[str, list]:
        """Visits overlapping [t0, t1] on these cameras, optionally of a taxon / above a score.
        Shown and hidden (suppressed) visits alike: callers add `_shown`."""
        sql = [f"{col}started_at <= ?", f"{col}ended_at >= ?", f"{col}camera IN ({', '.join('?' * len(names))})"]
        params: list = [self._iso(t1), self._iso(t0), *names]
        if taxon:
            clause, p = self._taxon(taxon, col)
            sql.append(clause)
            params += p
        if min_score:
            sql.append(f"{col}score >= ?")
            params.append(min_score)
        return " AND ".join(sql), params

    @staticmethod
    def _shown(db: sqlite3.Connection, where: str, col: str = "") -> str:
        """`where` narrowed to the visits the gate didn't hide."""
        return f"{where} AND {col}suppressed IS NULL" if _gated(db) else where

    @staticmethod
    def _hidden(db: sqlite3.Connection, where: str, params: list, col: str = "") -> int:
        """How many visits matching `where` the unnamed-visit gate hid (0 for an older log)."""
        if not _gated(db):
            return 0
        alias = f" {col.rstrip('.')}" if col else ""
        return db.execute(f"SELECT COUNT(*) FROM visits{alias} WHERE {where} AND {col}suppressed IS NOT NULL",
                          params).fetchone()[0]

    def coverage(self, names: list[str], t0: float, t1: float, db: sqlite3.Connection | None = None) -> list[dict]:
        """For each camera: the share of [t0, t1] the monitor was receiving its frames."""
        if db is None:  # standalone call: open our own connection and close it when done
            with closing(self._db()) as own:
                return self.coverage(names, t0, t1, own)
        span = max(t1 - t0, 1e-9)
        watched: dict[str, float] = defaultdict(float)
        marks = ", ".join("?" * len(names))
        known = "camera_uptime" in _tables(db)  # an older monitor didn't record when it was watching
        if known:
            for row in db.execute(f"SELECT camera, started_at, ended_at FROM camera_uptime WHERE camera IN ({marks})"
                                  " AND started_at <= ? AND ended_at >= ?", (*names, self._iso(t1), self._iso(t0))):
                watched[row["camera"]] += max(0.0, min(times.epoch(row["ended_at"]), t1) - max(times.epoch(row["started_at"]), t0))
        skipped: dict[str, float] = defaultdict(float)
        if "analysis_skipped" in _tables(db):
            for row in db.execute(f"SELECT camera, started_at, ended_at FROM analysis_skipped WHERE camera IN ({marks})"
                                  " AND started_at <= ? AND ended_at >= ?", (*names, self._iso(t1), self._iso(t0))):
                skipped[row["camera"]] += max(0.0, min(times.epoch(row["ended_at"]), t1) - max(times.epoch(row["started_at"]), t0))
        status = {r["camera"]: r for r in db.execute(f"SELECT * FROM camera_status WHERE camera IN ({marks})", names)}
        out = []
        for name in names:
            st = status.get(name)
            out.append({
                "camera": name,
                "watched": round(min(watched[name] / span, 1.0), 3) if known else None,  # None: unknown
                "unanalyzed_motion_seconds": round(skipped[name]),  # motion the detector was too far behind to see
                "last_frame": self._local(_get(st, "last_frame_at")),
                "looks_moved_since": self._local(_get(st, "view_moved_since")),
            })
        return out
