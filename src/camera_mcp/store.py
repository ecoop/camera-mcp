# Copyright (c) 2026 Eric Cooper.
"""The monitor's SQLite store: motion windows and per-camera status.

The monitor is the only writer. Readers (the MCP server, from phase 2c) open the file
read-only; WAL mode lets them read while the monitor writes. Times are stored as UTC
ISO 8601 strings, so they sort and compare as text.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

VERDICTS = ("real", "false", "wrong_species")  # a human's correctness label on a visit

SCHEMA = """
CREATE TABLE IF NOT EXISTS motion_windows (
  id          INTEGER PRIMARY KEY,
  camera      TEXT NOT NULL,
  started_at  TEXT NOT NULL,
  ended_at    TEXT NOT NULL,
  peak        REAL NOT NULL,      -- largest fraction of pixels changed
  frames      INTEGER NOT NULL,   -- motion frames in the window
  clip        TEXT                -- path relative to the data directory; NULL if none was saved
);
CREATE INDEX IF NOT EXISTS motion_windows_camera_time ON motion_windows(camera, started_at);

CREATE TABLE IF NOT EXISTS visits (
  id          INTEGER PRIMARY KEY,
  camera      TEXT NOT NULL,
  started_at  TEXT NOT NULL,
  ended_at    TEXT NOT NULL,       -- the last frame the animal was detected in
  label       TEXT NOT NULL,       -- common name: 'domestic cat', 'bird', 'person'
  taxonomy    TEXT NOT NULL,       -- 'class;order;family;genus;species', '' if unknown
  grp         TEXT NOT NULL,       -- what the visit is keyed by: 'songbirds', 'cat', 'mammal'
  score       REAL NOT NULL,       -- the winning label's share of the vote, 0-1
  frames      INTEGER NOT NULL,    -- analysed frames it was detected in
  max_count   INTEGER NOT NULL,    -- most of them in any one frame
  thumbnail   TEXT,                -- paths relative to the data directory; NULL if not kept
  clip        TEXT,
  model       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS visits_camera_time ON visits(camera, started_at);
CREATE INDEX IF NOT EXISTS visits_grp_time ON visits(grp, started_at);
CREATE INDEX IF NOT EXISTS visits_taxonomy ON visits(taxonomy);

CREATE TABLE IF NOT EXISTS visit_frames (
  visit_id  INTEGER NOT NULL REFERENCES visits(id),
  ts        TEXT NOT NULL,
  label     TEXT NOT NULL,
  taxonomy  TEXT NOT NULL,
  score     REAL NOT NULL,
  boxes     TEXT NOT NULL           -- JSON: [{"conf": 0.94, "bbox": [x, y, w, h]}], fractions of the frame
);
CREATE INDEX IF NOT EXISTS visit_frames_visit ON visit_frames(visit_id);

CREATE TABLE IF NOT EXISTS visit_labels (  -- a human's correctness label on a visit (training data for the re-scorer)
  visit_id  INTEGER PRIMARY KEY REFERENCES visits(id),  -- one label per visit; a re-label replaces it
  verdict   TEXT NOT NULL,       -- 'real' | 'false' | 'wrong_species'
  note      TEXT,                -- optional free text; NULL if none
  at        TEXT NOT NULL        -- when the label was set (UTC ISO)
);

CREATE TABLE IF NOT EXISTS camera_uptime (
  camera      TEXT NOT NULL,
  started_at  TEXT NOT NULL,       -- a stretch of time the monitor was receiving this camera's frames
  ended_at    TEXT NOT NULL,
  PRIMARY KEY (camera, started_at)
);

CREATE TABLE IF NOT EXISTS analysis_skipped (
  camera      TEXT NOT NULL,       -- motion the detector never looked at, because it was too far behind
  started_at  TEXT NOT NULL,
  ended_at    TEXT NOT NULL,
  PRIMARY KEY (camera, started_at)
);

CREATE TABLE IF NOT EXISTS camera_status (
  camera          TEXT PRIMARY KEY,
  state           TEXT NOT NULL,  -- 'starting', 'watching', 'retrying', 'stopped'
  last_frame_at   TEXT,
  last_motion_at  TEXT,
  error           TEXT,           -- last error, credentials redacted
  updated_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS analysis_stats (  -- one sample per interval: how the shared analyzer is keeping up
  at             TEXT PRIMARY KEY,  -- sample time
  queue_depth    INTEGER NOT NULL,  -- segments waiting at the sample
  max_lag        REAL NOT NULL,     -- oldest segment age (s) analysed in the interval: how far behind it fell
  analyzed       INTEGER NOT NULL,  -- frames (inferences) run in the interval: throughput / "images processed"
  dropped        INTEGER NOT NULL,  -- segments dropped (deferred past the limit) in the interval
  infer_seconds  REAL NOT NULL      -- total analysis time in the interval (mean per frame = infer_seconds / analyzed)
);

CREATE TABLE IF NOT EXISTS analysis_camera_stats (  -- the same sample, split per camera (which cameras drive the load)
  at             TEXT NOT NULL,     -- sample time (shared with the analysis_stats row for this interval)
  camera         TEXT NOT NULL,
  queue_depth    INTEGER NOT NULL,  -- this camera's segments waiting at the sample
  max_lag        REAL NOT NULL,     -- this camera's oldest segment age (s) analysed in the interval
  analyzed       INTEGER NOT NULL,  -- this camera's frames analysed in the interval
  dropped        INTEGER NOT NULL,  -- this camera's segments dropped in the interval
  infer_seconds  REAL NOT NULL,     -- this camera's analysis time in the interval
  PRIMARY KEY (at, camera)
);
"""

# Columns added after the first release, added to an existing database when it's opened.
ADDED_COLUMNS = {
    "visits": {
        "open": "INTEGER NOT NULL DEFAULT 0",  # 1 while the animal is still there (ended_at = last seen so far)
    },
    "camera_status": {
        "view_score": "REAL",  # last moved-camera check: how well the view matched its reference, 0-1
        "view_moved_since": "TEXT",  # set while the camera looks moved; NULL otherwise
    },
}


def iso(t: float) -> str:
    return datetime.fromtimestamp(t, timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._lock = threading.Lock()  # one connection, shared by the camera threads
        with self._lock:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("PRAGMA busy_timeout=5000")  # wait out the other writer (web labels vs. monitor)
            self._db.executescript(SCHEMA)
            for table, columns in ADDED_COLUMNS.items():
                have = {row[1] for row in self._db.execute(f"PRAGMA table_info({table})")}
                for name, kind in columns.items():
                    if name not in have:
                        self._db.execute(f"ALTER TABLE {table} ADD COLUMN {name} {kind}")

    def add_window(self, camera: str, start: float, end: float, peak: float, frames: int,
                   clip: str | None) -> int:
        with self._lock:
            cur = self._db.execute(
                "INSERT INTO motion_windows (camera, started_at, ended_at, peak, frames, clip)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (camera, iso(start), iso(end), peak, frames, clip),
            )
            return int(cur.lastrowid)

    def open_visit(self, visit, *, model: str) -> int:
        """Record a visit as soon as its animal arrives, so "is anything there now?" sees it."""
        label = visit.label
        with self._lock:
            cur = self._db.execute(
                "INSERT INTO visits (camera, started_at, ended_at, label, taxonomy, grp, score, frames,"
                " max_count, model, open) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)",
                (visit.camera, iso(visit.start), iso(visit.last_seen), label.name, label.taxonomy, visit.key,
                 round(label.score, 3), visit.frames, visit.max_count, model),
            )
            return int(cur.lastrowid)

    def update_open_visit(self, visit) -> None:
        label = visit.label
        with self._lock:
            self._db.execute(
                "UPDATE visits SET ended_at = ?, label = ?, taxonomy = ?, grp = ?, score = ?, frames = ?,"
                " max_count = ? WHERE id = ? AND open = 1",
                (iso(visit.last_seen), label.name, label.taxonomy, visit.key, round(label.score, 3),
                 visit.frames, visit.max_count, visit.db_id),
            )

    def close_stale_visits(self, camera: str) -> int:
        """Visits left open by a monitor that stopped without closing them."""
        with self._lock:
            return self._db.execute("UPDATE visits SET open = 0 WHERE camera = ? AND open = 1", (camera,)).rowcount

    def extend_uptime(self, camera: str, start: float, end: float) -> None:
        with self._lock:
            self._db.execute(
                "INSERT INTO camera_uptime VALUES (?, ?, ?) ON CONFLICT(camera, started_at)"
                " DO UPDATE SET ended_at = excluded.ended_at",
                (camera, iso(start), iso(end)),
            )

    def add_skipped(self, camera: str, start: float, end: float) -> None:
        """Record motion that was never analysed; extends the camera's last span if they touch."""
        with self._lock:
            last = self._db.execute("SELECT started_at, ended_at FROM analysis_skipped WHERE camera = ?"
                                    " ORDER BY started_at DESC LIMIT 1", (camera,)).fetchone()
            if last and last[1] >= iso(start - 1.0):
                self._db.execute("UPDATE analysis_skipped SET ended_at = ? WHERE camera = ? AND started_at = ?",
                                 (max(last[1], iso(end)), camera, last[0]))
            else:
                self._db.execute("INSERT INTO analysis_skipped VALUES (?, ?, ?)", (camera, iso(start), iso(end)))

    def add_analysis_stats(self, at: float, queue_depth: int, max_lag: float, analyzed: int,
                           dropped: int, infer_seconds: float) -> None:
        """One sample of how the shared analyzer is keeping up (throughput, backlog, loss)."""
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO analysis_stats VALUES (?, ?, ?, ?, ?, ?)",
                             (iso(at), queue_depth, round(max_lag, 1), analyzed, dropped, round(infer_seconds, 3)))

    def add_camera_stats(self, at: float, rows: list[tuple]) -> None:
        """Per-camera analyzer samples for one interval: rows of
        (camera, queue_depth, max_lag, analyzed, dropped, infer_seconds). Shares `at` with the
        aggregate analysis_stats row, so the two line up on a chart."""
        if not rows:
            return
        at_iso = iso(at)
        values = [(at_iso, cam, depth, round(lag, 1), an, dr, round(inf, 3))
                  for cam, depth, lag, an, dr, inf in rows]
        with self._lock:
            self._db.executemany("INSERT OR REPLACE INTO analysis_camera_stats VALUES (?, ?, ?, ?, ?, ?, ?)", values)

    def add_visit(self, visit, *, model: str, thumbnail: str | None, clip: str | None) -> int:
        """Store a closed visits.Visit and its per-frame detections (completing its open row, if any)."""
        label = visit.label
        values = (visit.camera, iso(visit.start), iso(visit.last_seen), label.name, label.taxonomy, visit.key,
                  round(label.score, 3), visit.frames, visit.max_count, thumbnail, clip, model)
        with self._lock:
            if getattr(visit, "db_id", None):
                self._db.execute(
                    "UPDATE visits SET camera = ?, started_at = ?, ended_at = ?, label = ?, taxonomy = ?, grp = ?,"
                    " score = ?, frames = ?, max_count = ?, thumbnail = ?, clip = ?, model = ?, open = 0"
                    " WHERE id = ?", (*values, visit.db_id))
                visit_id = visit.db_id
            else:
                cur = self._db.execute(
                    "INSERT INTO visits (camera, started_at, ended_at, label, taxonomy, grp, score, frames,"
                    " max_count, thumbnail, clip, model) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", values)
                visit_id = int(cur.lastrowid)
            rows = []
            for t, dets in visit.log:
                best = max(dets, key=lambda d: d.box.conf)
                boxes = [{"conf": round(d.box.conf, 3), "bbox": [round(v, 4) for v in (d.box.x, d.box.y, d.box.w, d.box.h)]}
                         for d in dets]
                rows.append((visit_id, iso(t), best.label.name, best.label.taxonomy, round(best.label.score, 3),
                             json.dumps(boxes)))
            self._db.executemany("INSERT INTO visit_frames VALUES (?, ?, ?, ?, ?, ?)", rows)
            return visit_id

    def set_visit_label(self, visit_id: int, verdict: str, note: str | None = None) -> None:
        """Record a human's correctness verdict on a visit, upserting on visit_id so a re-label
        replaces the old one. Raises ValueError for an unknown verdict or a visit that isn't there."""
        if verdict not in VERDICTS:
            raise ValueError(f"verdict must be one of {', '.join(VERDICTS)}")
        with self._lock:
            if self._db.execute("SELECT 1 FROM visits WHERE id = ?", (visit_id,)).fetchone() is None:
                raise ValueError(f"no visit {visit_id}")
            self._db.execute(
                "INSERT INTO visit_labels (visit_id, verdict, note, at) VALUES (?, ?, ?, ?)"
                " ON CONFLICT(visit_id) DO UPDATE SET"
                " verdict = excluded.verdict, note = excluded.note, at = excluded.at",
                (visit_id, verdict, note, iso(time.time())),
            )

    def visits(self, camera: str | None = None) -> list[sqlite3.Row]:
        with self._lock:
            self._db.row_factory = sqlite3.Row
            sql = "SELECT * FROM visits" + (" WHERE camera = ?" if camera else "") + " ORDER BY started_at"
            return self._db.execute(sql, (camera,) if camera else ()).fetchall()

    def set_status(self, camera: str, state: str, *, now: float, last_frame: float | None = None,
                   last_motion: float | None = None, error: str | None = None) -> None:
        """Upsert a camera's status. Times left as None keep their stored value."""
        with self._lock:
            self._db.execute(
                """INSERT INTO camera_status (camera, state, last_frame_at, last_motion_at, error, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(camera) DO UPDATE SET
                     state = excluded.state,
                     last_frame_at = COALESCE(excluded.last_frame_at, last_frame_at),
                     last_motion_at = COALESCE(excluded.last_motion_at, last_motion_at),
                     error = excluded.error,
                     updated_at = excluded.updated_at""",
                (camera, state, last_frame and iso(last_frame), last_motion and iso(last_motion),
                 error, iso(now)),
            )

    def set_view(self, camera: str, score: float | None, moved: bool, *, now: float) -> None:
        """Record a moved-camera check. `view_moved_since` keeps the time it first looked moved."""
        with self._lock:
            self._db.execute(
                """UPDATE camera_status SET
                     view_score = COALESCE(?, view_score),
                     view_moved_since = CASE WHEN ? THEN COALESCE(view_moved_since, ?) ELSE NULL END,
                     updated_at = ?
                   WHERE camera = ?""",
                (score, moved, iso(now), iso(now), camera),
            )

    def windows(self, camera: str | None = None) -> list[sqlite3.Row]:
        with self._lock:
            self._db.row_factory = sqlite3.Row
            sql = "SELECT * FROM motion_windows" + (" WHERE camera = ?" if camera else "") + " ORDER BY started_at"
            return self._db.execute(sql, (camera,) if camera else ()).fetchall()

    def status(self) -> dict[str, sqlite3.Row]:
        with self._lock:
            self._db.row_factory = sqlite3.Row
            return {r["camera"]: r for r in self._db.execute("SELECT * FROM camera_status")}

    def close(self) -> None:
        with self._lock:
            self._db.close()
