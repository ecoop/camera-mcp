# Copyright (c) 2026 Eric Cooper.
"""What people recorded: visit and box tags with species, the labels map, and the visits whose
boxes still need tags."""

from __future__ import annotations

import time
from contextlib import closing

from camera_mcp.boxtags import best_frame, box_tags
from camera_mcp.config import find_camera
from camera_mcp.history.base import HistoryError, _gated, _has_column, _tables, parse_when


class TagQueries:
    """History's tags queries (a mixin over LogReader)."""

    def tagged_species(self) -> list[str]:
        """Species named in tags so far (visit or box), most used first: the picker's first suggestions."""
        with closing(self._db()) as db:
            tables = _tables(db)
            parts = []
            if "visit_labels" in tables and "species" in {r[1] for r in db.execute("PRAGMA table_info(visit_labels)")}:
                parts.append("SELECT species FROM visit_labels WHERE species IS NOT NULL")
            if "box_labels" in tables:
                parts.append("SELECT species FROM box_labels WHERE species IS NOT NULL")
            if not parts:
                return []
            rows = db.execute(f"SELECT species, COUNT(*) n FROM ({' UNION ALL '.join(parts)})"
                              " GROUP BY species ORDER BY n DESC, species").fetchall()
        return [r["species"] for r in rows]

    def label_map(self, camera: str, since: str = "30d", until: str | None = None, taxon: str | None = None,
                  unlabelled: bool = False, limit: int = 3000) -> dict:
        """One entry per visit on a camera, placed at its most confident box (the box the visit viewer
        and thumbnail show), as fractions of the FULL frame (boxes are stored as fractions of the
        camera's roi crop and mapped back here, using its current roi). An unnamed visit can collect
        boxes on different things, so `spread` says how far its box centres ranged; averaging them
        would place the dot where nothing was. With each visit's label, time and human verdict, so
        the dashboard can plot past labels over the camera's picture and open any one of them; `night`
        says whether it was seen under infrared (None: unknown), for showing day or night apart.
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
            night = "v.night" if _has_column(db, "night") else "NULL"
            rows = db.execute(f"""
                WITH vs AS (SELECT v.id FROM visits v {join} WHERE {where} ORDER BY v.started_at DESC LIMIT ?),
                b AS (SELECT f.visit_id AS vid, json_extract(je.value, '$.conf') AS c,
                             json_extract(je.value, '$.bbox[0]') AS x, json_extract(je.value, '$.bbox[1]') AS y,
                             json_extract(je.value, '$.bbox[2]') AS w, json_extract(je.value, '$.bbox[3]') AS h
                      FROM visit_frames f JOIN vs ON vs.id = f.visit_id, json_each(f.boxes) je),
                best AS (SELECT vid, x, y, w, h, max(c) AS c FROM b GROUP BY vid),  -- bare columns: the max(c) row
                spread AS (SELECT vid, max(x + w / 2) - min(x + w / 2) AS sx, max(y + h / 2) - min(y + h / 2) AS sy
                           FROM b GROUP BY vid)
                SELECT v.id, v.label, v.grp, v.started_at, v.frames, v.max_count, v.clip, v.thumbnail,
                       {sup} AS suppressed, {verdict} AS verdict, {night} AS night,
                       best.x + best.w / 2 AS cx, best.y + best.h / 2 AS cy, best.w AS bw, best.h AS bh,
                       spread.sx, spread.sy
                FROM vs JOIN visits v ON v.id = vs.id {join} JOIN best ON best.vid = v.id JOIN spread ON spread.vid = v.id
                ORDER BY v.started_at DESC""", (*params, max(1, min(limit, 5000)))).fetchall()
        return {
            "camera": cam.name, "from": self._local(self._iso(t0)), "to": self._local(self._iso(t1)),
            "visits": [{
                "id": r["id"], "camera": cam.name, "label": r["label"], "group": r["grp"],
                "started": self._local(r["started_at"]), "frames": r["frames"], "max_at_once": r["max_count"],
                "clip": r["clip"], "has_thumbnail": r["thumbnail"] is not None,
                "suppressed": r["suppressed"], "verdict": r["verdict"],
                "night": None if r["night"] in (None, -1) else bool(r["night"]),  # under infrared; None: unknown
                "x": round(rx + r["cx"] * rw, 4), "y": round(ry + r["cy"] * rh, 4),
                "size": round(max(r["bw"] * rw, r["bh"] * rh), 4),
                "spread": round(max(r["sx"] * rw, r["sy"] * rh), 4),
            } for r in rows if r["cx"] is not None],
        }

    def _label(self, db, visit_id: int) -> dict | None:
        if "visit_labels" not in _tables(db):
            return None
        species = "species" if "species" in {r[1] for r in db.execute("PRAGMA table_info(visit_labels)")} else "NULL"
        r = db.execute(f"SELECT verdict, note, at, {species} AS species FROM visit_labels WHERE visit_id = ?",
                       (visit_id,)).fetchone()
        if r is None:
            return None
        return {"verdict": r["verdict"], "species": r["species"], "note": r["note"], "at": self._local(r["at"])}

    def visit_label(self, visit_id: int) -> dict | None:
        """The human correctness label on a visit (with the species they gave, if any), or None if it
        hasn't been labelled (or the log is from an older monitor with no visit_labels table)."""
        with closing(self._db()) as db:
            return self._label(db, visit_id)

    def needs_box_tags(self, camera: str | None = None, limit: int = 500) -> list[int]:
        """Visits tagged real whose still (best frame) shows a box with no tag of its own or
        inherited (see box_tags): the ones to revisit so every box on a training still is known.
        Newest first."""
        import json
        with closing(self._db()) as db:
            if "visit_labels" not in _tables(db):
                return []
            q = "SELECT v.id FROM visits v JOIN visit_labels l ON l.visit_id = v.id WHERE l.verdict = 'real'"
            params: list = []
            if camera:
                cam = find_camera(self.cameras, camera)
                if cam is None:
                    raise HistoryError(f"unknown camera {camera!r}; known: {', '.join(sorted(self.cameras))}")
                q += " AND v.camera = ?"
                params.append(cam.name)
            ids = [r["id"] for r in db.execute(q + " ORDER BY v.started_at DESC", params)]
            has_box = "box_labels" in _tables(db)
            out = []
            for vid in ids:
                frames = [{"ts": r["ts"], "boxes": json.loads(r["boxes"] or "[]")}
                          for r in db.execute("SELECT ts, boxes FROM visit_frames WHERE visit_id = ? ORDER BY ts", (vid,))]
                if not frames:
                    continue
                explicit = {(r["frame_ts"], r["box"]): {"verdict": r["verdict"], "species": r["species"]}
                            for r in db.execute("SELECT frame_ts, box, verdict, species FROM box_labels WHERE visit_id = ?",
                                                (vid,))} if has_box else {}
                tags = box_tags(frames, self._label(db, vid), explicit)[best_frame(frames)]
                if any(t is None for t in tags):
                    out.append(vid)
                    if len(out) >= limit:
                        break
        return out

    def label_counts(self) -> dict:
        """How many visits carry each verdict, and the total labelled. Zeros (and an empty
        by_verdict) when nothing's been labelled or the log predates the visit_labels table."""
        with closing(self._db()) as db:
            if "visit_labels" not in _tables(db):
                return {"total": 0, "by_verdict": {}}
            rows = db.execute("SELECT verdict, COUNT(*) AS n FROM visit_labels GROUP BY verdict").fetchall()
        by_verdict = {r["verdict"]: r["n"] for r in rows}
        return {"total": sum(by_verdict.values()), "by_verdict": by_verdict}
