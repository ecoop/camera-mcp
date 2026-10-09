# Copyright (c) 2026 Eric Cooper.
"""Learned scenery: detections that are really part of the view.

MegaDetector sometimes boxes a fixed object as an animal or a person: in the first feeder
trial, an empty tray feeder ("animal"), a cloth over the deck rail and clothespins on the line
("person"). Nothing the classifier recognises, so they become visits over and over.
Camera-trap tools call the fix repeat-detection elimination: a box that keeps turning up in
the same place at the same size, over a long enough stretch, is scenery, and is ignored from
then on. Real people and animals move, so their boxes don't line up that closely.

Only boxes the classifier didn't confirm (unidentified "animal", and the detector's "person"
and "vehicle") are ever learned or dropped. A named animal holding still (a cat stalking the
feeder) is never treated as scenery. And a learned spot keeps a small picture of itself: a box
there is dropped only while the spot still looks the same, so a bird sitting on a learned
feeder roof changes the picture and is kept.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from camera_mcp.detect import iou
from camera_mcp.visits import Box, Detection

LEARNABLE = ("animal", "person", "vehicle")  # visit keys the classifier didn't confirm
MATCH_IOU = 0.6  # the same spot: boxes overlap this much (small boxes jitter by a few pixels)
MIN_HITS = 5  # seen at least this many times...
MIN_SPAN = 600.0  # ...over at least this many seconds (separate motion events, not one animal pausing)
FORGET_AFTER = 86400.0  # a spot not seen for a day is forgotten (the object was moved)
PATCH = 24  # the spot's picture: this many pixels square, grayscale
SAME_LOOK = 0.6  # correlation with that picture above which the spot "looks the same" (light-adjusted)


def patch(frame: np.ndarray | None, box: Box) -> list[float] | None:
    """The box's contents as a small, light-normalised grayscale picture (mean 0, std 1)."""
    if frame is None:
        return None
    h, w = frame.shape[:2]
    x0, y0 = int(box.x * w), int(box.y * h)
    x1, y1 = max(x0 + 2, int((box.x + box.w) * w)), max(y0 + 2, int((box.y + box.h) * h))
    region = frame[y0:y1, x0:x1]
    if region.size == 0:
        return None
    gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY) if region.ndim == 3 else region
    small = cv2.resize(gray, (PATCH, PATCH), interpolation=cv2.INTER_AREA).astype(np.float32)
    std = float(small.std())
    return ((small - small.mean()) / (std if std > 1e-3 else 1.0)).ravel().round(3).tolist()


def looks_same(a: list[float] | None, b: list[float] | None) -> bool:
    """Correlation of two normalised patches; unknown (no picture) counts as the same."""
    if a is None or b is None:
        return True
    return float(np.dot(a, b)) / len(a) >= SAME_LOOK


@dataclass
class Spot:
    x: float
    y: float
    w: float
    h: float
    hits: int
    first: float
    last: float
    kind: str = "animal"  # the visit key it was detected as
    look: list[float] | None = field(default=None, repr=False)  # its picture (see patch())

    @property
    def box(self) -> Box:
        return Box(self.x, self.y, self.w, self.h, 1.0)

    @property
    def learned(self) -> bool:
        return self.hits >= MIN_HITS and self.last - self.first >= MIN_SPAN


class Scenery:
    def __init__(self, path: Path):
        self.path = path
        try:
            self.spots = [Spot(**s) for s in json.loads(path.read_text())]
        except (FileNotFoundError, ValueError, TypeError):
            self.spots = []

    def filter(self, t: float, detections: list[Detection],
               frame: np.ndarray | None = None) -> tuple[list[Detection], list[Spot]]:
        """Drop unconfirmed boxes on learned scenery. Returns the rest, and any spots that
        became scenery with this frame (to log)."""
        kept: list[Detection] = []
        newly: list[Spot] = []
        changed = False
        for d in detections:
            if d.key not in LEARNABLE:
                kept.append(d)
                continue
            spot = max((s for s in self.spots if s.kind == d.key), key=lambda s: iou(s.box, d.box), default=None)
            if spot is None or iou(spot.box, d.box) < MATCH_IOU:
                self.spots.append(Spot(d.box.x, d.box.y, d.box.w, d.box.h, 1, t, t, d.key, patch(frame, d.box)))
                kept.append(d)
                changed = True
                continue
            now = patch(frame, spot.box)
            if spot.learned and not looks_same(now, spot.look):
                kept.append(d)  # something is on the spot (a bird on the feeder roof): keep it
                continue
            was = spot.learned
            spot.hits += 1
            spot.last = t
            if spot.look is None or not spot.learned:
                spot.look = now or spot.look  # while learning, keep the latest look of the empty spot
            changed = True
            if spot.learned and not was:
                newly.append(spot)
            if not spot.learned:
                kept.append(d)
        before = len(self.spots)
        self.spots = [s for s in self.spots if t - s.last < FORGET_AFTER]
        if changed or len(self.spots) != before:
            self._save()
        return kept, newly

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps([asdict(s) for s in self.spots]))
        tmp.replace(self.path)

    def reset(self) -> int:
        """Forget every spot (the camera was re-aimed). Returns how many were learned."""
        learned = sum(s.learned for s in self.spots)
        self.spots = []
        self.path.unlink(missing_ok=True)
        return learned
