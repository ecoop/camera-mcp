# Copyright (c) 2026 Eric Cooper.
"""The monitor's SQLite store: motion windows and per-camera status.

The monitor is the only writer. Readers (the MCP server, from phase 2c) open the file
read-only; WAL mode lets them read while the monitor writes. Times are stored as UTC
ISO 8601 strings, so they sort and compare as text.
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

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

CREATE TABLE IF NOT EXISTS camera_status (
  camera          TEXT PRIMARY KEY,
  state           TEXT NOT NULL,  -- 'starting', 'watching', 'retrying', 'stopped'
  last_frame_at   TEXT,
  last_motion_at  TEXT,
  error           TEXT,           -- last error, credentials redacted
  updated_at      TEXT NOT NULL
);
"""


def iso(t: float) -> str:
    return datetime.fromtimestamp(t, timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._lock = threading.Lock()  # one connection, shared by the camera threads
        with self._lock:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.executescript(SCHEMA)

    def add_window(self, camera: str, start: float, end: float, peak: float, frames: int,
                   clip: str | None) -> int:
        with self._lock:
            cur = self._db.execute(
                "INSERT INTO motion_windows (camera, started_at, ended_at, peak, frames, clip)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (camera, iso(start), iso(end), peak, frames, clip),
            )
            return int(cur.lastrowid)

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
