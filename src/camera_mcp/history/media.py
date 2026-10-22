# Copyright (c) 2026 Eric Cooper.
"""Pictures and video for a visit: its thumbnail, aligned still, clip, and per-frame boxes with their tags."""

from __future__ import annotations

from contextlib import closing
from pathlib import Path

from camera_mcp.boxtags import best_frame, box_tags
from camera_mcp.history.base import HistoryError, _tables


class MediaQueries:
    """History's media queries (a mixin over LogReader)."""

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

    def visit_boxes(self, visit_id: int) -> dict:
        """A visit's per-frame detector boxes (the visit_frames table), with its camera, start time
        and whether a clip was kept. Each frame's `boxes` are fractions of the analysis frame — i.e.
        the camera's roi crop if it has one; the web layer maps them to the full frame. Each frame
        also has `tags`, one per box (see box_tags), and `best` is the frame the still shows. Raises
        HistoryError if the visit is unknown."""
        import json
        with closing(self._db()) as db:
            v = db.execute("SELECT camera, started_at, clip, label, taxonomy FROM visits WHERE id = ?",
                           (visit_id,)).fetchone()
            if v is None:
                raise HistoryError(f"no visit {visit_id}")
            rows = db.execute("SELECT ts, label, taxonomy, score, boxes FROM visit_frames"
                              " WHERE visit_id = ? ORDER BY ts", (visit_id,)).fetchall()
            label = self._label(db, visit_id)
            explicit = {(r["frame_ts"], r["box"]): {"verdict": r["verdict"], "species": r["species"]}
                        for r in db.execute("SELECT frame_ts, box, verdict, species FROM box_labels WHERE visit_id = ?",
                                            (visit_id,))} if "box_labels" in _tables(db) else {}
        frames = []
        for r in rows:
            try:
                boxes = json.loads(r["boxes"])
            except (ValueError, TypeError):
                boxes = []
            frames.append({"ts": r["ts"], "label": r["label"], "taxonomy": r["taxonomy"],
                           "score": r["score"], "boxes": boxes})
        for f, tags in zip(frames, box_tags(frames, label, explicit)):
            f["tags"] = tags
        return {"camera": v["camera"], "started_at": v["started_at"], "has_clip": bool(v["clip"]),
                "label": v["label"], "taxonomy": v["taxonomy"], "frames": frames, "best": best_frame(frames),
                "visit_tag": label}

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
