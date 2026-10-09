# Copyright (c) 2026 Eric Cooper.
"""Tier 1: motion detection on small grayscale frames, and the windows it opens.

Tuned for recall, not precision. A false alarm (a branch, a cloud) costs a little work
for the detector downstream; a missed animal is lost. So the thresholds are low, and
there are no exclusion zones by default.

The background adapts slowly, so a scene that changes and then holds still (a parked
chair, a sleeping cat) stops counting as motion after a while. That is why, from phase
2b on, an open visit keeps sampling for the detector whether or not there is motion.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from camera_mcp.config import MonitorSettings


class MotionDetector:
    """Fraction of pixels that differ from a running-average background."""

    def __init__(self, settings: MonitorSettings, adapt: float = 0.05):
        self.threshold = settings.pixel_threshold
        self.min_changed = settings.min_changed
        self.adapt = adapt  # how fast the background follows the scene, per frame
        self.background: np.ndarray | None = None

    def update(self, gray: np.ndarray) -> float:
        """Feed one grayscale frame; return the fraction of pixels that changed."""
        frame = cv2.GaussianBlur(gray, (5, 5), 0).astype(np.float32)
        if self.background is None or self.background.shape != frame.shape:
            self.background = frame
            return 0.0
        changed = float((cv2.absdiff(frame, self.background) > self.threshold).mean())
        cv2.accumulateWeighted(frame, self.background, self.adapt)
        return changed

    def is_motion(self, changed: float) -> bool:
        return changed >= self.min_changed


@dataclass
class Window:
    """A stretch of time with motion: from the first moving frame to the last."""

    start: float
    last_motion: float
    peak: float  # the largest changed fraction seen
    frames: int = 0  # frames with motion


class MotionWindows:
    """Opens a window on motion and closes it after `quiet_seconds` without any."""

    def __init__(self, quiet_seconds: float):
        self.quiet = quiet_seconds
        self.open: Window | None = None

    def feed(self, t: float, changed: float, moving: bool) -> Window | None:
        """Feed one frame's result at time `t`; return a window if this frame closed it."""
        if moving:
            if self.open is None:
                self.open = Window(start=t, last_motion=t, peak=changed)
            self.open.last_motion = t
            self.open.peak = max(self.open.peak, changed)
            self.open.frames += 1
            return None
        if self.open is not None and t - self.open.last_motion >= self.quiet:
            return self.close()
        return None

    def close(self) -> Window | None:
        """Close the open window now (end of input, shutdown)."""
        window, self.open = self.open, None
        return window
