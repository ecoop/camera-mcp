# Copyright (c) 2026 Eric Cooper.
"""Visits and activity: the MCP tools' core questions, plus which cameras are being watched now."""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import closing
from datetime import datetime

from camera_mcp.config import find_camera
from camera_mcp.history.base import (
    GROUP_BY,
    WATCHING_LAG,
    HistoryError,
    _fav_cols,
    _get,
    _tables,
    parse_when,
)
from camera_mcp.taxa import UNIONS
from camera_mcp.times import epoch


class VisitQueries:
    """History's visits queries (a mixin over LogReader)."""

    def find_visits(self, since: str = "24h", until: str | None = None, cameras: list[str] | None = None,
                    tag: str | None = None, taxon: str | None = None, overlapping: str | None = None,
                    min_score: float = 0.0, limit: int = 50, include_suppressed: bool = False,
                    favorites_only: bool = False) -> dict:
        """Visits overlapping the window, newest first. Visits the unnamed-visit gate hid (likely
        leaves, see gate.py) are left out and counted in `hidden`, unless `include_suppressed`.
        `favorites_only`: just the starred ones, from any time (the window is ignored) and hidden
        or not: a favourite is a keeper wherever it is."""
        now = time.time()
        t0, t1 = parse_when(since, now, self.tz), parse_when(until, now, self.tz)
        if favorites_only:
            t0, include_suppressed = 0.0, True
        names = self._camera_names(cameras, tag)
        with closing(self._db()) as db:
            where, params = self._where(t0, t1, names, taxon, min_score, col="v.")
            if overlapping:
                clause, p = self._taxon(overlapping, "o.")
                where += (" AND EXISTS (SELECT 1 FROM visits o WHERE o.camera = v.camera AND o.id != v.id"
                          f" AND o.started_at <= v.ended_at AND o.ended_at >= v.started_at AND {clause})")
                params += p
            if favorites_only:
                where += (" AND v.id IN (SELECT visit_id FROM favorites)" if "favorites" in _tables(db) else " AND 0")
            hidden = 0 if include_suppressed else self._hidden(db, where, params, "v.")
            if not include_suppressed:
                where = self._shown(db, where, "v.")
            total = db.execute(f"SELECT COUNT(*) FROM visits v WHERE {where}", params).fetchone()[0]
            verdict = ("(SELECT verdict FROM visit_labels l WHERE l.visit_id = v.id)"  # a human's label, if any
                       if "visit_labels" in _tables(db) else "NULL")
            rows = db.execute(f"SELECT v.*, {verdict} AS verdict, {_fav_cols(db)} FROM visits v WHERE {where}"
                              " ORDER BY v.started_at DESC LIMIT ?",
                              (*params, max(1, min(limit, 500)))).fetchall()
            return {
                "from": self._local(self._iso(t0)), "to": self._local(self._iso(t1)),
                "total": total, "shown": len(rows), "hidden": hidden,
                "visits": [self._visit_out(r) for r in rows],
                "coverage": self.coverage(names, t0, t1, db),
            }

    def _visit_out(self, r) -> dict:
        """One visits row (with its `verdict`) as find_visits returns it."""
        return {
            "id": r["id"], "camera": r["camera"],
            "started": self._local(r["started_at"]), "last_seen": self._local(r["ended_at"]),
            "still_there": bool(_get(r, "open", 0)), "label": r["label"], "group": r["grp"],
            "score": r["score"], "max_at_once": r["max_count"], "frames": r["frames"],
            "has_thumbnail": r["thumbnail"] is not None, "clip": r["clip"],
            "suppressed": _get(r, "suppressed"), "verdict": r["verdict"],
            "favorite": {"kind": _get(r, "fav_kind"), "clip_at": _get(r, "fav_at")} if _get(r, "fav_kind") else None,
        }

    def visits_by_id(self, ids: list[int]) -> list[dict]:
        """The given visits, in that order, as find_visits returns them (unknown ids are skipped)."""
        if not ids:
            return []
        with closing(self._db()) as db:
            verdict = ("(SELECT verdict FROM visit_labels l WHERE l.visit_id = v.id)"
                       if "visit_labels" in _tables(db) else "NULL")
            rows = {r["id"]: r for r in db.execute(
                f"SELECT v.*, {verdict} AS verdict, {_fav_cols(db)} FROM visits v WHERE v.id IN ({','.join('?' * len(ids))})",
                ids)}
        return [self._visit_out(rows[i]) for i in ids if i in rows]

    def activity_summary(self, since: str = "7d", until: str | None = None, group_by: str = "species",
                         cameras: list[str] | None = None, tag: str | None = None,
                         taxon: str | None = None) -> dict:
        """Visits in the window counted by `group_by`, as the owner tagged them: hidden ones are
        counted in `hidden`, ones tagged false in `tagged_false`, and one the owner named counts
        under that name (see _as_tagged)."""
        if group_by not in GROUP_BY:
            raise HistoryError(f"group_by must be one of {', '.join(GROUP_BY)}")
        now = time.time()
        t0, t1 = parse_when(since, now, self.tz), parse_when(until, now, self.tz)
        names = self._camera_names(cameras, tag)
        with closing(self._db()) as db:
            where, params = self._where(t0, t1, names, taxon, 0.0)
            hidden = self._hidden(db, where, params)
            where = self._shown(db, where)
            tagged_false = self._tagged_false(db, where, params)
            tagged, tagged_params = self._as_tagged(db)
            rows = db.execute(f"SELECT label, grp, camera, started_at, max_count FROM {tagged} WHERE {where}"
                              " LIMIT 100000", (*tagged_params, *params)).fetchall()

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
                "total_visits": len(rows), "hidden": hidden, "tagged_false": tagged_false, "rows": out,
                "coverage": self.coverage(names, t0, t1, db),
            }

    def visits_timeline(self, since: str = "24h", until: str | None = None,
                        cameras: list[str] | None = None, taxon: str | None = None, bins: int = 48) -> dict:
        """Visit counts over the window, binned into `bins` time buckets and split by group (grp),
        for a stacked timeline of what visited and when. Counted as the owner tagged them, like
        activity_summary: ones tagged false are left out (`tagged_false`), and a named one counts in
        its name's group. The web layer maps groups to display classes/colors."""
        now = time.time()
        t0, t1 = parse_when(since, now, self.tz), parse_when(until, now, self.tz)
        window = {"from": self._local(self._iso(t0)), "to": self._local(self._iso(t1))}
        names = [find_camera(self.cameras, c).name for c in (cameras or []) if find_camera(self.cameras, c)]
        where = "started_at >= ? AND started_at <= ?"
        params: list = [self._iso(t0), self._iso(t1)]
        if names:
            where += f" AND camera IN ({','.join('?' * len(names))})"
            params += names
        if taxon:
            clause, p = self._taxon(taxon)
            where += f" AND {clause}"
            params += p
        with closing(self._db()) as db:
            where = self._shown(db, where)
            tagged_false = self._tagged_false(db, where, params)
            tagged, tagged_params = self._as_tagged(db)
            rows = db.execute(f"SELECT started_at, grp FROM {tagged} WHERE {where}",
                              (*tagged_params, *params)).fetchall()
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
        return {**window, "bins": out, "bin_seconds": round(width), "total": dict(total), "visits": len(rows),
                "tagged_false": tagged_false}

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
            last = epoch(st["last_frame_at"]) if st and st["last_frame_at"] else None
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
