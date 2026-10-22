# Copyright (c) 2026 Eric Cooper.
"""How the detector kept up: throughput, backlog and lag, per camera and over time."""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import closing
from datetime import datetime

from camera_mcp.config import find_camera
from camera_mcp.history.base import _tables, parse_when


class HealthQueries:
    """History's health queries (a mixin over LogReader)."""

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
