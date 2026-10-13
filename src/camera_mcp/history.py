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
                    min_score: float = 0.0, limit: int = 50) -> dict:
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
            total = db.execute(f"SELECT COUNT(*) FROM visits v WHERE {where}", params).fetchone()[0]
            rows = db.execute(f"SELECT * FROM visits v WHERE {where} ORDER BY v.started_at DESC LIMIT ?",
                              (*params, max(1, min(limit, 500)))).fetchall()
            return {
                "from": self._local(self._iso(t0)), "to": self._local(self._iso(t1)),
                "total": total, "shown": len(rows),
                "visits": [{
                    "id": r["id"], "camera": r["camera"],
                    "started": self._local(r["started_at"]), "last_seen": self._local(r["ended_at"]),
                    "still_there": bool(_get(r, "open", 0)), "label": r["label"], "group": r["grp"],
                    "score": r["score"], "max_at_once": r["max_count"], "frames": r["frames"],
                    "has_thumbnail": r["thumbnail"] is not None, "clip": r["clip"],
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
                "total_visits": len(rows), "rows": out, "coverage": self.coverage(names, t0, t1, db),
            }

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

    def analysis_series(self, since: str = "24h", until: str | None = None, max_points: int = 300) -> dict:
        """The analysis-stats samples over a window, for charting the backlog/throughput timeline.
        Downsampled to at most `max_points` points."""
        now = time.time()
        t0, t1 = parse_when(since, now, self.tz), parse_when(until, now, self.tz)
        window = {"from": self._local(self._iso(t0)), "to": self._local(self._iso(t1))}
        with closing(self._db()) as db:
            if "analysis_stats" not in _tables(db):
                return {**window, "points": []}
            rows = db.execute("SELECT at, queue_depth, max_lag, analyzed, dropped FROM analysis_stats "
                              "WHERE at >= ? AND at <= ? ORDER BY at", (self._iso(t0), self._iso(t1))).fetchall()
        step = max(1, len(rows) // max_points)
        points = [{"at": self._local(r["at"]), "depth": r["queue_depth"], "lag": round(r["max_lag"]),
                   "frames": r["analyzed"], "dropped": r["dropped"]} for r in rows[::step]]
        return {**window, "points": points}

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
