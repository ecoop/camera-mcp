# Copyright (c) 2026 Eric Cooper.
"""Visits: one kind of animal on one camera, from first detection until it's gone.

The tracker is fed every analysed frame's boxes. Boxes are grouped by visit key (a group
such as "songbirds" or "cat", else the class), so a flock is one visit with a head count,
and a cat arriving among the birds opens a second, overlapping visit. A visit closes after
`close_after` seconds without its kind being detected.

The visit's label is decided by a score-weighted vote over all its boxes. A plain "bird"
box that couldn't be sorted joins whatever bird visit is open, rather than opening its own.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from camera_mcp.taxa import BIRD_GROUPS, Label

UNNAMED = ("animal", "person", "vehicle")  # keys that aren't an identified animal


@dataclass(frozen=True)
class Box:
    x: float  # left, as a fraction of the frame width
    y: float
    w: float
    h: float
    conf: float  # detector confidence


@dataclass(frozen=True)
class Detection:
    box: Box
    label: Label
    key: str  # the visit it belongs to (see taxa.visit_key)


@dataclass
class Visit:
    camera: str
    key: str
    start: float
    last_seen: float
    frames: int = 0
    max_count: int = 0
    votes: dict[tuple[str, str], float] = field(default_factory=lambda: defaultdict(float))
    best_conf: float = 0.0
    thumbnail: np.ndarray | None = None  # RGB crop of the best box so far
    best_t: float = 0.0  # time of the best-box frame (so the monitor can save the full frame)
    best_seg: str | None = None  # the ring-buffer segment that frame came from, and its offset within it
    best_seg_off: float = 0.0
    log: list[tuple[float, list[Detection]]] = field(default_factory=list)  # per frame
    db_id: int | None = None  # its row in the store, written when it opened

    @property
    def label(self) -> Label:
        """The winning (name, taxonomy) by summed score; score = its share of the votes."""
        named = {k: v for k, v in self.votes.items() if k[0] != "animal"} or self.votes  # "animal": no opinion
        if not named:
            return Label(self.key)
        (name, taxonomy), weight = max(named.items(), key=lambda kv: kv[1])
        return Label(name, taxonomy, weight / sum(named.values()))


def crop(frame: np.ndarray, box: Box, margin: float = 0.15) -> np.ndarray:
    h, w = frame.shape[:2]
    mx, my = box.w * margin, box.h * margin
    x0, y0 = max(0, int((box.x - mx) * w)), max(0, int((box.y - my) * h))
    x1, y1 = min(w, int((box.x + box.w + mx) * w)), min(h, int((box.y + box.h + my) * h))
    return frame[y0:y1, x0:x1].copy()


class VisitTracker:
    def __init__(self, camera: str, close_after: float = 30.0):
        self.camera = camera
        self.close_after = close_after
        self.open: dict[str, Visit] = {}

    def feed(self, t: float, detections: list[Detection], frame: np.ndarray | None = None) -> list[Visit]:
        """Add one analysed frame at time `t`; return the visits this frame closed."""
        by_key: dict[str, list[Detection]] = defaultdict(list)
        named = {d.key for d in detections if d.key not in UNNAMED}
        for d in detections:
            key = d.key
            if key == "bird":  # unsorted: join an open bird visit if there is one
                key = next((k for k in self.open if k in BIRD_GROUPS or k == "bird"), "bird")
            elif key == "animal":  # unidentified: most likely one of the animals already here
                key = next(iter(sorted(named)), None) or next(
                    (k for k in self.open if k not in UNNAMED), "animal")
            by_key[key].append(d)
        for key, dets in by_key.items():
            if key in BIRD_GROUPS and key not in self.open and "bird" in self.open:
                self.open[key] = self.open.pop("bird")  # the unsorted bird visit now has a group
                self.open[key].key = key
            if key not in UNNAMED and key not in self.open and "animal" in self.open:
                self.open[key] = self.open.pop("animal")  # the unidentified animal now has a name
                self.open[key].key = key
            visit = self.open.get(key)
            if visit is None:
                visit = self.open[key] = Visit(self.camera, key, start=t, last_seen=t)
            visit.last_seen = t
            visit.frames += 1
            visit.max_count = max(visit.max_count, len(dets))
            for d in dets:
                visit.votes[(d.label.name, d.label.taxonomy)] += max(d.label.score, 0.01) * d.box.conf
            best = max(dets, key=lambda d: d.box.conf)
            if frame is not None and best.box.conf > visit.best_conf:
                visit.best_conf = best.box.conf
                visit.thumbnail = crop(frame, best.box)
                visit.best_t = t  # the monitor tags this with its segment and saves the full frame
            visit.log.append((t, dets))
        return self._close(lambda v: t - v.last_seen >= self.close_after)

    def close_all(self) -> list[Visit]:
        return self._close(lambda v: True)

    def _close(self, done) -> list[Visit]:
        closed = [v for v in self.open.values() if done(v)]
        for v in closed:
            del self.open[v.key]
        return closed
