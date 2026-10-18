# Copyright (c) 2026 Eric Cooper.
"""Questions about the past, answered from the monitor's visit log.

The log is opened read-only: the monitor is its only writer. Every answer about a stretch
of time says how much of it each camera was actually watched, because "no skunks this week"
is only true if the camera was watching all week.
"""

from __future__ import annotations

import re
import sqlite3
import time
from collections import defaultdict
from contextlib import closing
from datetime import datetime, timedelta
from pathlib import Path

from camera_mcp.config import Camera, Site, find_camera
from camera_mcp.store import iso
from camera_mcp.taxa import CLASS_NAMES, GROUPS, UNIONS, expand_taxon

DURATION = re.compile(r"^(\d+(?:\.\d+)?)\s*(s|m|h|d|w)$")
UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}
GROUP_NAMES = {g for g, _ in GROUPS} | {"bird", "mammal", "animal", "person", "vehicle"}
GROUP_BY = ("species", "group", "camera", "hour", "day")
WATCHING_LAG = 60.0  # a camera whose last frame is older than this isn't being watched now


class HistoryError(ValueError):
    pass


def parse_when(text: str | None, now: float, tz) -> float:
    """An ISO time (local if it has no zone), a duration ago ("15m", "4h", "3d", "2w"),
    "now", "today" or "yesterday", as epoch seconds."""
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
        raise HistoryError(f"can't read the time {text!r}: use an ISO time like 2026-10-09T06:00, a "
                           "duration ago like 15m, 4h, 3d or 2w, or now, today, yesterday") from None
    return (when if when.tzinfo else when.replace(tzinfo=tz)).timestamp()


def _epoch(iso: str) -> float:
    return datetime.fromisoformat(iso).timestamp()


def _get(row: sqlite3.Row | None, key: str, default=None):
    """A column that an older monitor's log may not have yet (the server never alters the log)."""
    return row[key] if row is not None and key in row.keys() else default


def _tables(db: sqlite3.Connection) -> set[str]:
    return {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}


def _gated(db: sqlite3.Connection) -> bool:
    """Whether the log records the unnamed-visit gate's verdicts (an older monitor's doesn't)."""
    return "suppressed" in {r[1] for r in db.execute("PRAGMA table_info(visits)")}


class History:
    def __init__(self, db_path: Path, cameras: dict[str, Camera], site: Site | None = None):
        self.db_path = Path(db_path)
        self.cameras = cameras
        self.tz = (site or Site()).tz()

    # ---- plumbing -----------------------------------------------------------------

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
        return datetime.fromisoformat(iso).astimezone(self.tz).isoformat(timespec="seconds") if iso else None

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
                watched[row["camera"]] += max(0.0, min(_epoch(row["ended_at"]), t1) - max(_epoch(row["started_at"]), t0))
        skipped: dict[str, float] = defaultdict(float)
        if "analysis_skipped" in _tables(db):
            for row in db.execute(f"SELECT camera, started_at, ended_at FROM analysis_skipped WHERE camera IN ({marks})"
                                  " AND started_at <= ? AND ended_at >= ?", (*names, self._iso(t1), self._iso(t0))):
                skipped[row["camera"]] += max(0.0, min(_epoch(row["ended_at"]), t1) - max(_epoch(row["started_at"]), t0))
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

    # ---- questions ----------------------------------------------------------------

    def find_visits(self, since: str = "24h", until: str | None = None, cameras: list[str] | None = None,
                    tag: str | None = None, taxon: str | None = None, overlapping: str | None = None,
                    min_score: float = 0.0, limit: int = 50, include_suppressed: bool = False) -> dict:
        """Visits overlapping the window, newest first. Visits the unnamed-visit gate hid (likely
        leaves, see gate.py) are left out and counted in `hidden`, unless `include_suppressed`."""
        now = time.time()
        t0, t1 = parse_when(since, now, self.tz), parse_when(until, now, self.tz)
        names = self._camera_names(cameras, tag)
        with closing(self._db()) as db:
            where, params = self._where(t0, t1, names, taxon, min_score, col="v.")
            if overlapping:
                clause, p = self._taxon(overlapping, "o.")
                where += (" AND EXISTS (SELECT 1 FROM visits o WHERE o.camera = v.camera AND o.id != v.id"
                          f" AND o.started_at <= v.ended_at AND o.ended_at >= v.started_at AND {clause})")
                params += p
            hidden = 0 if include_suppressed else self._hidden(db, where, params, "v.")
            if not include_suppressed:
                where = self._shown(db, where, "v.")
            total = db.execute(f"SELECT COUNT(*) FROM visits v WHERE {where}", params).fetchone()[0]
            verdict = ("(SELECT verdict FROM visit_labels l WHERE l.visit_id = v.id)"  # a human's label, if any
                       if "visit_labels" in _tables(db) else "NULL")
            rows = db.execute(f"SELECT v.*, {verdict} AS verdict FROM visits v WHERE {where}"
                              " ORDER BY v.started_at DESC LIMIT ?",
                              (*params, max(1, min(limit, 500)))).fetchall()
            return {
                "from": self._local(self._iso(t0)), "to": self._local(self._iso(t1)),
                "total": total, "shown": len(rows), "hidden": hidden,
                "visits": [{
                    "id": r["id"], "camera": r["camera"],
                    "started": self._local(r["started_at"]), "last_seen": self._local(r["ended_at"]),
                    "still_there": bool(_get(r, "open", 0)), "label": r["label"], "group": r["grp"],
                    "score": r["score"], "max_at_once": r["max_count"], "frames": r["frames"],
                    "has_thumbnail": r["thumbnail"] is not None, "clip": r["clip"],
                    "suppressed": _get(r, "suppressed"), "verdict": r["verdict"],
                } for r in rows],
                "coverage": self.coverage(names, t0, t1, db),
            }

    def activity_summary(self, since: str = "7d", until: str | None = None, group_by: str = "species",
                         cameras: list[str] | None = None, tag: str | None = None,
                         taxon: str | None = None) -> dict:
        if group_by not in GROUP_BY:
            raise HistoryError(f"group_by must be one of {', '.join(GROUP_BY)}")
        now = time.time()
        t0, t1 = parse_when(since, now, self.tz), parse_when(until, now, self.tz)
        names = self._camera_names(cameras, tag)
        with closing(self._db()) as db:
            where, params = self._where(t0, t1, names, taxon, 0.0)
            hidden = self._hidden(db, where, params)
            where = self._shown(db, where)
            rows = db.execute(f"SELECT label, grp, camera, started_at, max_count FROM visits WHERE {where} LIMIT 100000",
                              params).fetchall()

            def key(r) -> str:
                if group_by in ("hour", "day"):
                    local = datetime.fromisoformat(r["started_at"]).astimezone(self.tz)
                    return f"{local:%H}:00" if group_by == "hour" else f"{local:%Y-%m-%d}"
                return r[{"species": "label", "group": "grp", "camera": "camera"}[group_by]]

            agg: dict[str, dict] = {}
            for r in rows:
                a = agg.setdefault(key(r), {"key": key(r), "visits": 0, "max_at_once": 0, "first": None, "last": None})
                a["visits"] += 1
                a["max_at_once"] = max(a["max_at_once"], r["max_count"])
                a["first"] = min(a["first"] or r["started_at"], r["started_at"])
                a["last"] = max(a["last"] or r["started_at"], r["started_at"])
            out = sorted(agg.values(), key=(lambda a: a["key"]) if group_by in ("hour", "day") else (lambda a: -a["visits"]))
            for a in out:
                a["first"], a["last"] = self._local(a["first"]), self._local(a["last"])
            return {
                "from": self._local(self._iso(t0)), "to": self._local(self._iso(t1)), "group_by": group_by,
                "total_visits": len(rows), "hidden": hidden, "rows": out, "coverage": self.coverage(names, t0, t1, db),
            }

    def observed_taxa(self, restrict: tuple[str, ...] = ()) -> list[str]:
        """The taxon filter values worth offering in the UI, as taxa `find_visits` accepts.

        With `restrict` set (the site's `menu_species`), exactly those, in that order — a curated
        menu the operator pins. Otherwise derived from the log: the distinct `grp` values actually
        logged, most frequent first, plus any applicable UNIONS (e.g. "night-critters" once a
        raccoon, skunk or opossum has been seen). Empty when there's no visit log yet."""
        if restrict:
            return list(restrict)
        try:
            with closing(self._db()) as db:
                rows = db.execute("SELECT grp, COUNT(*) n FROM visits WHERE grp IS NOT NULL AND grp != ''"
                                  " GROUP BY grp ORDER BY n DESC, grp").fetchall()
        except HistoryError:
            return []  # no log yet: the monitor hasn't run
        observed = [r["grp"] for r in rows]
        seen = set(observed)
        unions = [u for u, members in UNIONS.items() if seen.intersection(members)]
        return unions + observed

    def visit_thumbnail(self, visit_id: int, data_dir: Path) -> tuple[bytes, str]:
        with closing(self._db()) as db:
            row = db.execute("SELECT * FROM visits WHERE id = ?", (visit_id,)).fetchone()
        if row is None:
            raise HistoryError(f"no visit {visit_id}")
        caption = (f"visit {visit_id}: {row['label']} ({row['grp']}) on {row['camera']}, "
                   f"{self._local(row['started_at'])} to {self._local(row['ended_at'])}, up to {row['max_count']} at once")
        path = data_dir / row["thumbnail"] if row["thumbnail"] else None
        if path is None or not path.exists():
            raise HistoryError(f"no thumbnail was kept for visit {visit_id} ({caption}): images of people may be "
                               "turned off for that camera, or the visit is still going on")
        return path.read_bytes(), caption

    def analysis_health(self, since: str = "24h", until: str | None = None) -> dict:
        """How the shared analyzer kept up over the window: throughput (frames analysed and mean
        cost), backlog (peak queue depth and lag), and loss (segments dropped on overload)."""
        now = time.time()
        t0, t1 = parse_when(since, now, self.tz), parse_when(until, now, self.tz)
        window = {"from": self._local(self._iso(t0)), "to": self._local(self._iso(t1))}
        with closing(self._db()) as db:
            if "analysis_stats" not in _tables(db):
                return {**window, "samples": 0, "note": "no analysis stats yet (older monitor, or detection is off)"}
            rows = db.execute("SELECT * FROM analysis_stats WHERE at >= ? AND at <= ? ORDER BY at",
                              (self._iso(t0), self._iso(t1))).fetchall()
        if not rows:
            return {**window, "samples": 0, "note": "no samples in this window yet"}
        frames = sum(r["analyzed"] for r in rows)
        worst = max(rows, key=lambda r: r["max_lag"])
        return {
            **window,
            "samples": len(rows),
            "frames_analyzed": frames,
            "segments_dropped": sum(r["dropped"] for r in rows),  # dropped on sustained overload
            "mean_seconds_per_frame": round(sum(r["infer_seconds"] for r in rows) / frames, 3) if frames else None,
            "peak_lag_seconds": round(worst["max_lag"], 1),  # furthest the analyzer fell behind
            "peak_lag_at": self._local(worst["at"]),
            "peak_queue_depth": max(r["queue_depth"] for r in rows),
        }

    def analysis_series(self, since: str = "24h", until: str | None = None, max_points: int = 300,
                        camera: str | None = None) -> dict:
        """The analysis-stats samples over a window, for charting the backlog/throughput timeline.
        With `camera`, just that camera's load (from analysis_camera_stats, which only has data from
        when per-camera recording started); otherwise the aggregate across all cameras. Downsampled
        to at most `max_points` points."""
        now = time.time()
        t0, t1 = parse_when(since, now, self.tz), parse_when(until, now, self.tz)
        window = {"from": self._local(self._iso(t0)), "to": self._local(self._iso(t1)), "camera": camera}
        with closing(self._db()) as db:
            tables = _tables(db)
            if camera is not None:
                cam = find_camera(self.cameras, camera)
                name = cam.name if cam else camera
                if "analysis_camera_stats" not in tables:
                    return {**window, "points": []}
                rows = db.execute(
                    "SELECT at, queue_depth, max_lag, analyzed, dropped FROM analysis_camera_stats "
                    "WHERE camera = ? AND at >= ? AND at <= ? ORDER BY at",
                    (name, self._iso(t0), self._iso(t1))).fetchall()
            else:
                if "analysis_stats" not in tables:
                    return {**window, "points": []}
                # sum across cameras per sample, so legacy single rows and new per-camera rows both read right
                rows = db.execute(
                    "SELECT at, SUM(queue_depth) queue_depth, MAX(max_lag) max_lag, "
                    "SUM(analyzed) analyzed, SUM(dropped) dropped FROM analysis_stats "
                    "WHERE at >= ? AND at <= ? GROUP BY at ORDER BY at", (self._iso(t0), self._iso(t1))).fetchall()
        step = max(1, len(rows) // max_points)
        points = [{"at": self._local(r["at"]), "depth": r["queue_depth"], "lag": round(r["max_lag"]),
                   "frames": r["analyzed"], "dropped": r["dropped"]} for r in rows[::step]]
        return {**window, "points": points}

    def analysis_series_all(self, since: str = "24h", until: str | None = None, max_points: int = 200) -> dict:
        """Each camera's analysis series over the window, keyed by camera, for a small-multiples view
        (which cameras drive the congestion). Only cameras with per-camera data appear. Downsampled
        per camera to at most `max_points` points."""
        now = time.time()
        t0, t1 = parse_when(since, now, self.tz), parse_when(until, now, self.tz)
        window = {"from": self._local(self._iso(t0)), "to": self._local(self._iso(t1))}
        with closing(self._db()) as db:
            if "analysis_camera_stats" not in _tables(db):
                return {**window, "cameras": {}}
            rows = db.execute("SELECT camera, at, queue_depth, max_lag, analyzed, dropped "
                              "FROM analysis_camera_stats WHERE at >= ? AND at <= ? ORDER BY camera, at",
                              (self._iso(t0), self._iso(t1))).fetchall()
        by: dict[str, list] = defaultdict(list)
        for r in rows:
            by[r["camera"]].append(r)
        cameras = {}
        for cam, rs in by.items():
            step = max(1, len(rs) // max_points)
            cameras[cam] = [{"at": self._local(r["at"]), "depth": r["queue_depth"], "lag": round(r["max_lag"]),
                             "frames": r["analyzed"], "dropped": r["dropped"]} for r in rs[::step]]
        return {**window, "cameras": cameras}

    def stats_heatmap(self, since: str = "7d", until: str | None = None) -> dict:
        """Analyzer load by local hour-of-day (0-23) × camera over the window, for a heatmap of when
        and where the congestion is. Each cell totals frames analysed and segments dropped in that
        hour across the window (so a multi-day window shows the recurring daily pattern)."""
        now = time.time()
        t0, t1 = parse_when(since, now, self.tz), parse_when(until, now, self.tz)
        window = {"from": self._local(self._iso(t0)), "to": self._local(self._iso(t1))}
        with closing(self._db()) as db:
            if "analysis_camera_stats" not in _tables(db):
                return {**window, "cameras": [], "cells": {}, "max_frames": 0, "max_dropped": 0}
            rows = db.execute("SELECT camera, at, analyzed, dropped FROM analysis_camera_stats "
                              "WHERE at >= ? AND at <= ?", (self._iso(t0), self._iso(t1))).fetchall()
        cells: dict[str, list] = defaultdict(lambda: [{"frames": 0, "dropped": 0} for _ in range(24)])
        for r in rows:
            try:
                hour = datetime.fromisoformat(r["at"]).astimezone(self.tz).hour
            except (ValueError, TypeError):
                continue
            c = cells[r["camera"]][hour]
            c["frames"] += r["analyzed"]
            c["dropped"] += r["dropped"]
        cams = sorted(cells)
        allc = [c for cam in cells.values() for c in cam]
        return {**window, "cameras": cams, "cells": {cam: cells[cam] for cam in cams},
                "max_frames": max((c["frames"] for c in allc), default=0),
                "max_dropped": max((c["dropped"] for c in allc), default=0)}

    def label_map(self, camera: str, since: str = "30d", until: str | None = None, taxon: str | None = None,
                  unlabelled: bool = False, limit: int = 3000) -> dict:
        """One entry per visit on a camera, placed where its boxes were: the mean box centre and size,
        as fractions of the FULL frame (boxes are stored as fractions of the camera's roi crop and
        mapped back here, using its current roi). With each visit's label, time and human verdict, so
        the dashboard can plot past labels over the camera's picture and open any one of them.
        Labelled visits only, unless `unlabelled`. Raises HistoryError for an unknown camera."""
        now = time.time()
        t0, t1 = parse_when(since, now, self.tz), parse_when(until, now, self.tz)
        cam = find_camera(self.cameras, camera)
        if cam is None:
            raise HistoryError(f"unknown camera {camera!r}; known: {', '.join(sorted(self.cameras))}")
        rx, ry, rw, rh = cam.roi or (0.0, 0.0, 1.0, 1.0)
        with closing(self._db()) as db:
            has_labels = "visit_labels" in _tables(db)
            if not has_labels and not unlabelled:
                return {"camera": cam.name, "from": self._local(self._iso(t0)), "to": self._local(self._iso(t1)), "visits": []}
            where, params = self._where(t0, t1, [cam.name], taxon, 0.0, col="v.")
            verdict = "l.verdict" if has_labels else "NULL"
            join = "LEFT JOIN visit_labels l ON l.visit_id = v.id" if has_labels else ""
            if not unlabelled:
                where += " AND l.verdict IS NOT NULL"
            sup = "v.suppressed" if _gated(db) else "NULL"
            rows = db.execute(f"""
                SELECT v.id, v.label, v.grp, v.started_at, v.frames, v.max_count, v.clip, v.thumbnail,
                       {sup} AS suppressed, {verdict} AS verdict,
                       avg(json_extract(je.value, '$.bbox[0]') + json_extract(je.value, '$.bbox[2]') / 2) AS cx,
                       avg(json_extract(je.value, '$.bbox[1]') + json_extract(je.value, '$.bbox[3]') / 2) AS cy,
                       avg(json_extract(je.value, '$.bbox[2]')) AS bw, avg(json_extract(je.value, '$.bbox[3]')) AS bh
                FROM visits v {join} JOIN visit_frames f ON f.visit_id = v.id, json_each(f.boxes) je
                WHERE {where} GROUP BY v.id ORDER BY v.started_at DESC LIMIT ?""", (*params, max(1, min(limit, 5000)))).fetchall()
        return {
            "camera": cam.name, "from": self._local(self._iso(t0)), "to": self._local(self._iso(t1)),
            "visits": [{
                "id": r["id"], "camera": cam.name, "label": r["label"], "group": r["grp"],
                "started": self._local(r["started_at"]), "frames": r["frames"], "max_at_once": r["max_count"],
                "clip": r["clip"], "has_thumbnail": r["thumbnail"] is not None,
                "suppressed": r["suppressed"], "verdict": r["verdict"],
                "x": round(rx + r["cx"] * rw, 4), "y": round(ry + r["cy"] * rh, 4),
                "size": round(max(r["bw"] * rw, r["bh"] * rh), 4),
            } for r in rows if r["cx"] is not None],
        }

    def visits_timeline(self, since: str = "24h", until: str | None = None,
                        cameras: list[str] | None = None, taxon: str | None = None, bins: int = 48) -> dict:
        """Visit counts over the window, binned into `bins` time buckets and split by group (grp),
        for a stacked timeline of what visited and when. The web layer maps groups to display
        classes/colors."""
        now = time.time()
        t0, t1 = parse_when(since, now, self.tz), parse_when(until, now, self.tz)
        window = {"from": self._local(self._iso(t0)), "to": self._local(self._iso(t1))}
        names = [find_camera(self.cameras, c).name for c in (cameras or []) if find_camera(self.cameras, c)]
        q = "SELECT started_at, grp FROM visits WHERE started_at >= ? AND started_at <= ?"
        params: list = [self._iso(t0), self._iso(t1)]
        if names:
            q += f" AND camera IN ({','.join('?' * len(names))})"
            params += names
        with closing(self._db()) as db:
            rows = db.execute(self._shown(db, q), params).fetchall()
        span = max(1e-6, t1 - t0)
        width = span / max(1, bins)
        buckets: list[dict] = [defaultdict(int) for _ in range(bins)]
        total: dict[str, int] = defaultdict(int)
        for r in rows:
            try:
                ts = datetime.fromisoformat(r["started_at"]).timestamp()
            except (ValueError, TypeError):
                continue
            bi = min(bins - 1, max(0, int((ts - t0) / width)))
            buckets[bi][r["grp"]] += 1
            total[r["grp"]] += 1
        out = [{"t": self._local(self._iso(t0 + (i + 0.5) * width)), "grps": dict(b)} for i, b in enumerate(buckets)]
        return {**window, "bins": out, "bin_seconds": round(width), "total": dict(total), "visits": len(rows)}

    def visit_boxes(self, visit_id: int) -> dict:
        """A visit's per-frame detector boxes (the visit_frames table), with its camera, start time
        and whether a clip was kept. Each frame's `boxes` are fractions of the analysis frame — i.e.
        the camera's roi crop if it has one; the web layer maps them to the full frame. Raises
        HistoryError if the visit is unknown."""
        import json
        with closing(self._db()) as db:
            v = db.execute("SELECT camera, started_at, clip, label, taxonomy FROM visits WHERE id = ?",
                           (visit_id,)).fetchone()
            if v is None:
                raise HistoryError(f"no visit {visit_id}")
            rows = db.execute("SELECT ts, label, taxonomy, score, boxes FROM visit_frames"
                              " WHERE visit_id = ? ORDER BY ts", (visit_id,)).fetchall()
        frames = []
        for r in rows:
            try:
                boxes = json.loads(r["boxes"])
            except (ValueError, TypeError):
                boxes = []
            frames.append({"ts": r["ts"], "label": r["label"], "taxonomy": r["taxonomy"],
                           "score": r["score"], "boxes": boxes})
        return {"camera": v["camera"], "started_at": v["started_at"], "has_clip": bool(v["clip"]),
                "label": v["label"], "taxonomy": v["taxonomy"], "frames": frames}

    def visit_label(self, visit_id: int) -> dict | None:
        """The human correctness label on a visit, or None if it hasn't been labelled (or the log
        is from an older monitor with no visit_labels table)."""
        with closing(self._db()) as db:
            if "visit_labels" not in _tables(db):
                return None
            r = db.execute("SELECT verdict, note, at FROM visit_labels WHERE visit_id = ?",
                           (visit_id,)).fetchone()
        if r is None:
            return None
        return {"verdict": r["verdict"], "note": r["note"], "at": self._local(r["at"])}

    def label_counts(self) -> dict:
        """How many visits carry each verdict, and the total labelled. Zeros (and an empty
        by_verdict) when nothing's been labelled or the log predates the visit_labels table."""
        with closing(self._db()) as db:
            if "visit_labels" not in _tables(db):
                return {"total": 0, "by_verdict": {}}
            rows = db.execute("SELECT verdict, COUNT(*) AS n FROM visit_labels GROUP BY verdict").fetchall()
        by_verdict = {r["verdict"]: r["n"] for r in rows}
        return {"total": sum(by_verdict.values()), "by_verdict": by_verdict}

    def visit_still(self, visit_id: int, data_dir: Path) -> Path:
        """The full-res still saved at detection time (aligned to the box). Raises HistoryError if the
        visit has no stored still (older visit, or its segment had aged out) — caller falls back."""
        with closing(self._db()) as db:
            has = "still" in {r[1] for r in db.execute("PRAGMA table_info(visits)")}
            row = db.execute("SELECT still FROM visits WHERE id = ?", (visit_id,)).fetchone() if has else None
        if row is None or not row["still"]:
            raise HistoryError(f"no stored still for visit {visit_id}")
        path = Path(data_dir) / row["still"]
        if not path.exists():
            raise HistoryError(f"still file missing for visit {visit_id}")
        return path

    def visit_clip(self, visit_id: int, data_dir: Path) -> Path:
        """The path to a visit's saved clip. Raises HistoryError if the visit or its clip is missing."""
        with closing(self._db()) as db:
            row = db.execute("SELECT clip FROM visits WHERE id = ?", (visit_id,)).fetchone()
        if row is None:
            raise HistoryError(f"no visit {visit_id}")
        if not row["clip"]:
            raise HistoryError(f"no clip was kept for visit {visit_id}")
        path = Path(data_dir) / row["clip"]
        if not path.exists():
            raise HistoryError(f"clip file missing for visit {visit_id}")
        return path

    def monitor_status(self) -> list[dict]:
        now = time.time()
        with closing(self._db()) as db:
            status = {r["camera"]: r for r in db.execute("SELECT * FROM camera_status")}
            has_open = "open" in {r[1] for r in db.execute("PRAGMA table_info(visits)")}
            open_counts = dict(db.execute("SELECT camera, COUNT(*) FROM visits WHERE open = 1 GROUP BY camera").fetchall()
                               if has_open else [])
        out = []
        for name in self.cameras:
            st = status.get(name)
            last = _epoch(st["last_frame_at"]) if st and st["last_frame_at"] else None
            out.append({
                "camera": name,
                "watching_now": bool(st and st["state"] == "watching" and last and now - last < WATCHING_LAG),
                "state": st["state"] if st else "never seen by the monitor",
                "last_frame": self._local(st["last_frame_at"]) if st else None,
                "last_motion": self._local(st["last_motion_at"]) if st else None,
                "looks_moved_since": self._local(_get(st, "view_moved_since")),
                "view_score": _get(st, "view_score"),
                "open_visits": open_counts.get(name, 0),
                "error": st["error"] if st else None,
            })
        return out
