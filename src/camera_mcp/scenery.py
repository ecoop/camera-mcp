# Copyright (c) 2026 Eric Cooper.
"""Learned scenery: detections that are really part of the view.

MegaDetector sometimes boxes a fixed object as an animal or a person: in the first feeder
trial, an empty tray feeder ("animal"), a cloth over the deck rail and clothespins on the line
("person"). Nothing the classifier recognises, so they become visits over and over.
Camera-trap tools call the fix repeat-detection elimination: a box that keeps turning up in
the same place at the same size, over a long enough stretch, is scenery, and is ignored from
then on. Real people and animals move, so their boxes don't line up that closely.

Only unconfirmed detections ("animal"/"person"/"vehicle") ever *train* a spot, so a real animal
(which gets a species label) never becomes scenery. But once a spot is learned, ANY detection on
it is judged by appearance, not its label: SpeciesNet sometimes labels the same fixed object a
species (an empty feeder cap read as "opossum"), and that must be dropped too. A learned spot
keeps a small picture of itself; a detection on it is dropped only when it matches that picture.
A species-labelled box is dropped only on a positive match (so a real animal on the spot, which
looks different, is kept — never missed for lack of evidence).
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
SINGLE_HIT_TTL = 1800.0  # a one-off detection that never recurs is dropped after this (prunes noise)
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


def matches_reference(now: list[float] | None, reference: list[float] | None) -> bool:
    """True only when both patches exist and correlate: positive evidence the spot looks like its
    learned reference. Unknown (no picture on either side) is NOT a match — so nothing is dropped
    without evidence."""
    if now is None or reference is None:
        return False
    return float(np.dot(now, reference)) / len(now) >= SAME_LOOK


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
        """Drop detections that are part of the learned scenery. Returns the rest, and any spots
        that became scenery with this frame (to log)."""
        kept: list[Detection] = []
        newly: list[Spot] = []
        changed = False
        for d in detections:
            learned = [s for s in self.spots if s.learned and iou(s.box, d.box) >= MATCH_IOU]
            if learned:
                spot = max(learned, key=lambda s: iou(s.box, d.box))
                same = matches_reference(patch(frame, spot.box), spot.look)
                if d.key in LEARNABLE:
                    drop = same or patch(frame, spot.box) is None  # unconfirmed: scenery unless it looks different
                else:
                    drop = same  # species-labelled: drop only on positive evidence it's the fixed object
                if drop:
                    spot.hits += 1
                    spot.last = t
                    changed = True
                    continue
                kept.append(d)  # something really is on the spot
                continue
            if d.key not in LEARNABLE:
                kept.append(d)  # identified species off any learned spot: a real animal
                continue
            spot = max((s for s in self.spots if s.kind == d.key and not s.learned),
                       key=lambda s: iou(s.box, d.box), default=None)
            if spot is None or iou(spot.box, d.box) < MATCH_IOU:
                self.spots.append(Spot(d.box.x, d.box.y, d.box.w, d.box.h, 1, t, t, d.key, patch(frame, d.box)))
                kept.append(d)
                changed = True
                continue
            was = spot.learned
            spot.hits += 1
            spot.last = t
            spot.look = patch(frame, spot.box) or spot.look  # keep the latest look while learning
            changed = True
            if spot.learned and not was:
                newly.append(spot)
            if not spot.learned:
                kept.append(d)
        if self._prune(t):
            changed = True
        if changed:
            self._save()
        return kept, newly

    def _prune(self, t: float) -> bool:
        """Drop spots not seen for a day, and one-off detections that never recurred."""
        before = len(self.spots)
        self.spots = [s for s in self.spots
                      if t - s.last < FORGET_AFTER and not (s.hits <= 1 and t - s.first > SINGLE_HIT_TTL)]
        return len(self.spots) != before

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
