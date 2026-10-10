# Copyright (c) 2026 Eric Cooper.
"""Tiers 2 and 3: find animals in a frame and name them.

`SpeciesNetDetector` runs Google's SpeciesNet in three steps, called separately so that
every box gets classified (SpeciesNet's own pipeline classifies only the most confident
box per image, so a cat among birds could go unnamed):

1. MegaDetector finds animal, person and vehicle boxes. On wide-view cameras it also looks
   at overlapping tiles of the frame, since it shrinks the whole frame to 1280 px first
   and a sparrow across the yard would be a few pixels.
2. The species classifier names each animal box, from a crop of the full-resolution frame,
   all boxes in one batch.
3. The ensemble applies the location filter (species not found in the configured country
   and state are dropped) and rolls uncertain answers up ("bird" rather than a guess).

SpeciesNet is optional (`pip install camera-mcp[detect]`); it's imported on first use, so
the package works without it and the monitor falls back to motion only.
"""

from __future__ import annotations

import logging
import threading
from typing import Protocol

import numpy as np

from camera_mcp.config import Site
from camera_mcp.taxa import ANIMAL, PERSON, VEHICLE, Label, allowed, bird_group_from_scores, parse_prediction, visit_key
from camera_mcp.visits import Box, Detection

log = logging.getLogger("camera_mcp.detect")


class Detector(Protocol):
    model: str  # recorded with every visit, so labels can be recomputed after a model change

    def analyze(self, frame: np.ndarray, *, tiles: bool | tuple[int, int] = False, min_conf: float = 0.2,
                max_boxes: int = 8, vehicles: bool = False) -> list[Detection]:
        """Boxes in an RGB frame, each with a label and visit key. `tiles`: False/True (whole frame
        or a 2x2 grid), or an explicit (rows, cols) grid; rows*cols <= 1 means the whole frame."""
        ...


def available() -> bool:
    try:
        import speciesnet  # noqa: F401
    except ImportError:
        return False
    return True


def iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a.x + a.w, b.x + b.w) - max(a.x, b.x))
    iy = max(0.0, min(a.y + a.h, b.y + b.h) - max(a.y, b.y))
    inter = ix * iy
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union > 0 else 0.0


def contained(inner: Box, outer: Box) -> float:
    """The share of `inner` that lies inside `outer`."""
    ix = max(0.0, min(inner.x + inner.w, outer.x + outer.w) - max(inner.x, outer.x))
    iy = max(0.0, min(inner.y + inner.h, outer.y + outer.h) - max(inner.y, outer.y))
    return ix * iy / (inner.w * inner.h) if inner.w * inner.h > 0 else 0.0


def merge_boxes(boxes: list[tuple[Box, str]], overlap: float = 0.5, inside: float = 0.7) -> list[tuple[Box, str]]:
    """Drop boxes that repeat a more confident one of the same kind: overlapping it, or mostly
    inside it (a tile's edge can cut an animal into a partial box within the full-frame one)."""
    kept: list[tuple[Box, str]] = []
    for box, kind in sorted(boxes, key=lambda bk: -bk[0].conf):
        if all(iou(box, k) < overlap and not (kk == kind and (contained(box, k) >= inside or contained(k, box) >= inside))
               for k, kk in kept):
            kept.append((box, kind))
    return kept


def tile_windows(rows: int = 2, cols: int = 2, overlap: float = 0.2) -> list[tuple[float, float, float, float]]:
    """(x, y, w, h) fractions of the frame for overlapping tiles."""
    tw, th = (1 + overlap * (cols - 1)) / cols, (1 + overlap * (rows - 1)) / rows
    return [(c * (1 - tw) / max(cols - 1, 1), r * (1 - th) / max(rows - 1, 1), tw, th)
            for r in range(rows) for c in range(cols)]


class SpeciesNetDetector:
    def __init__(self, site: Site, model_name: str | None = None):
        self.site = site
        self.model_name = model_name
        self.model = "speciesnet (not loaded)"
        self._lock = threading.Lock()
        self._parts = None

    def _load(self):
        if self._parts is None:
            from speciesnet import DEFAULT_MODEL
            from speciesnet.classifier import SpeciesNetClassifier
            from speciesnet.detector import SpeciesNetDetector as MegaDetector
            from speciesnet.ensemble import SpeciesNetEnsemble

            name = self.model_name or DEFAULT_MODEL
            log.info("loading SpeciesNet %s (first use downloads the weights)", name)
            self._parts = (MegaDetector(name), SpeciesNetClassifier(name), SpeciesNetEnsemble(name, geofence=True))
            from importlib.metadata import version

            self.model = f"speciesnet {version('speciesnet')} / {name.rsplit('/', 2)[-2] if '/' in name else name}"
        return self._parts

    def analyze(self, frame: np.ndarray, *, tiles: bool | tuple[int, int] = False, min_conf: float = 0.2,
                max_boxes: int = 8, vehicles: bool = False) -> list[Detection]:
        from PIL import Image
        from speciesnet.utils import BBox

        rows, cols = (2, 2) if tiles is True else (1, 1) if tiles is False else tiles
        with self._lock:  # one model, shared by every camera
            mega, classifier, ensemble = self._load()
            image = Image.fromarray(frame)
            found = self._boxes(mega, image, (0.0, 0.0, 1.0, 1.0))
            if rows * cols > 1:
                for window in tile_windows(rows, cols):
                    found += self._boxes(mega, image, window)
            found = [(b, k) for b, k in merge_boxes(found) if b.conf >= min_conf and (vehicles or k != "vehicle")]

            animals = [b for b, k in found if k == "animal"][:max_boxes]
            labels: dict[Box, Label] = {}
            top5s: dict[Box, list[tuple[str, float]]] = {}
            if animals:
                crops = [classifier.preprocess(image, bboxes=[BBox(b.x, b.y, b.w, b.h)]) for b in animals]
                keys = [f"box{i}" for i in range(len(animals))]
                results = classifier.batch_predict(keys, crops)
                geo = {"country": self.site.country, **({"admin1_region": self.site.admin1_region}
                                                        if self.site.admin1_region else {})}
                for key, box, result in zip(keys, animals, results):
                    det = {"category": "1", "label": "animal", "conf": box.conf, "bbox": [box.x, box.y, box.w, box.h]}
                    pred = ensemble.combine([key], {key: result}, {key: {"detections": [det]}}, {key: geo}, {})[0]
                    labels[box] = parse_prediction(pred.get("prediction", ""), pred.get("prediction_score", 0.0))
                    cls = result.get("classifications", {})
                    top5s[box] = [(";".join(c.split(";")[1:6]), s) for c, s in zip(cls.get("classes", []), cls.get("scores", []))]

        detections = []
        for box, kind in found:
            if kind == "person":
                label = Label(PERSON.name, PERSON.taxonomy, box.conf)
            elif kind == "vehicle":
                label = Label(VEHICLE.name, "", box.conf)
            elif box in labels:
                label = allowed(labels[box], self.site.allowlist)
            else:  # past max_boxes: take the label of the nearest classified box
                nearest = min(labels, key=lambda b: (b.x - box.x) ** 2 + (b.y - box.y) ** 2, default=None)
                label = labels[nearest] if nearest else ANIMAL
            bird_group = bird_group_from_scores(top5s.get(box, [])) if label.taxonomy.startswith("aves;;") else None
            detections.append(Detection(box, label, visit_key(label, bird_group)))
        return detections

    @staticmethod
    def _boxes(mega, image, window: tuple[float, float, float, float]) -> list[tuple[Box, str]]:
        """MegaDetector on one window of the image; boxes mapped back to full-frame fractions."""
        x0, y0, ww, wh = window
        W, H = image.size
        part = image if window == (0.0, 0.0, 1.0, 1.0) else image.crop(
            (int(x0 * W), int(y0 * H), int((x0 + ww) * W), int((y0 + wh) * H)))
        result = mega.predict("frame", mega.preprocess(part))
        out = []
        for d in result.get("detections", []):
            bx, by, bw, bh = d["bbox"]
            kind = {"human": "person"}.get(d["label"], d["label"])
            out.append((Box(x0 + bx * ww, y0 + by * wh, bw * ww, bh * wh, float(d["conf"])), kind))
        return out
