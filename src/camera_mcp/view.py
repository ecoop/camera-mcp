# Copyright (c) 2026 Eric Cooper.
"""Has a camera been moved? (Raccoons tilt cameras.)

Every few minutes, while nothing is moving, the watcher compares the scene's edges (fence
lines, posts, the house) with a reference picture from the same camera. Edges survive
changes of light far better than brightness does. Day (colour) and night (infrared) get
separate references, because the two images look nothing alike.

If several checks in a row find the edges somewhere else, the camera is flagged as moved.
The flag clears by itself once the view matches again (the camera was put back), or when
the reference is reset after re-aiming on purpose (`camera-monitor --reset-view NAME`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

DAY, NIGHT = "day", "night"
MIN_EDGES = 0.004  # a frame with fewer edge pixels than this (dark, fog, a blocked lens) can't be judged
REFRESH_SECONDS = 3600.0  # a matching view replaces the reference at most this often (seasons, furniture)
REFRESH_SCORE = 0.6  # ...and only if it matches at least this well


def edges(gray: np.ndarray) -> np.ndarray:
    """The frame's strong edges as a boolean mask. The heavy blur and high thresholds keep
    structure (fence lines, posts, the house) and drop texture (grass, gravel), which would
    otherwise line up by chance after a shift. Calibrated on our cameras: the same view scores
    0.92-0.95, a raccoon-tilted fence camera 0.03, a 12-pixel shift 0.48, 40% less light 0.68."""
    return cv2.Canny(cv2.GaussianBlur(gray, (7, 7), 0), 60, 160) > 0


def similarity(current: np.ndarray, reference: np.ndarray, slack: int = 1) -> float | None:
    """How well two edge masks line up, 0 (not at all) to 1 (exactly), allowing `slack` pixels
    of wobble. None if the current frame has too few edges to judge."""
    if current.shape != reference.shape or current.mean() < MIN_EDGES:
        return None
    kernel = np.ones((2 * slack + 1, 2 * slack + 1), np.uint8)
    near_ref = cv2.dilate(reference.astype(np.uint8), kernel) > 0
    near_cur = cv2.dilate(current.astype(np.uint8), kernel) > 0
    precision = (current & near_ref).sum() / max(current.sum(), 1)
    recall = (reference & near_cur).sum() / max(reference.sum(), 1)
    return 0.0 if precision + recall == 0 else float(2 * precision * recall / (precision + recall))


def is_colour(u: np.ndarray, v: np.ndarray, threshold: float = 3.0) -> bool:
    """Day or night: an infrared image has no colour, so its chroma sits at the neutral 128."""
    return float(np.abs(u.astype(np.int16) - 128).mean() + np.abs(v.astype(np.int16) - 128).mean()) > threshold


def is_infrared(image: np.ndarray) -> bool:
    """Whether a picture (RGB or BGR, any size: a whole frame or a visit's thumbnail) was taken under
    the camera's infrared light. The camera sends pure grey then: chroma exactly neutral. A daylight
    crop, however dull (grey bark, shade), keeps at least ~1; on 4,794 thumbnails every infrared one
    measured 0 and every daylight one 1 or more, so 0.5 splits them with nothing in between."""
    if image is None or image.size == 0 or image.ndim != 3:
        return False
    yuv = cv2.cvtColor(np.ascontiguousarray(image[..., :3]), cv2.COLOR_RGB2YUV)
    return not is_colour(yuv[..., 1], yuv[..., 2], threshold=0.5)


@dataclass(frozen=True)
class ViewResult:
    score: float | None  # None: this was the first look in this light (the reference was just taken)
    moved: bool
    changed: bool  # the moved flag flipped on this check


class ViewCheck:
    def __init__(self, ref_dir: Path, camera: str, every: float = 300.0, threshold: float = 0.5,
                 strikes: int = 3):
        self.ref_dir = ref_dir
        self.camera = camera
        self.every = every
        self.threshold = threshold
        self.strikes_needed = strikes
        self.strikes = 0
        self.moved = False
        self.last_check = 0.0
        self.refreshed: dict[str, float] = {}

    def due(self, now: float) -> bool:
        return self.every > 0 and now - self.last_check >= self.every

    def _path(self, light: str) -> Path:
        return self.ref_dir / f"{self.camera}-{light}.png"

    def _save(self, light: str, mask: np.ndarray, now: float) -> None:
        self.ref_dir.mkdir(parents=True, exist_ok=True)
        # imencode/imdecode, not imwrite/imread: SpeciesNet's yolov5 replaces those in cv2.
        self._path(light).write_bytes(cv2.imencode(".png", mask.astype(np.uint8) * 255)[1].tobytes())
        self.refreshed[light] = now

    def check(self, now: float, gray: np.ndarray, colour: bool) -> ViewResult | None:
        """Compare one still frame with the reference. None if the frame can't be judged."""
        self.last_check = now
        light = DAY if colour else NIGHT
        current = edges(gray)
        path = self._path(light)
        stored = (cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_GRAYSCALE)
                  if path.exists() else None)
        if stored is None or stored.shape != current.shape:
            if current.mean() < MIN_EDGES:
                return None
            self._save(light, current, now)
            return ViewResult(None, self.moved, False)
        score = similarity(current, stored > 0)
        if score is None:
            return None
        if score < self.threshold:
            self.strikes += 1
        else:
            self.strikes = 0
            if score >= REFRESH_SCORE and now - self.refreshed.get(light, 0.0) >= REFRESH_SECONDS:
                self._save(light, current, now)
        was = self.moved
        self.moved = self.strikes >= self.strikes_needed
        return ViewResult(score, self.moved, self.moved != was)

    def reset(self) -> int:
        """Forget the reference views (after re-aiming the camera on purpose). Returns how many."""
        removed = 0
        for light in (DAY, NIGHT):
            if self._path(light).exists():
                self._path(light).unlink()
                removed += 1
        self.strikes, self.moved = 0, False
        return removed
